import sys
import os
import functools
import datetime
from typing import Optional, Tuple, Iterable, Union

import scipy.sparse

import xarray as xr
import numpy as np
import numpy.typing as npt
import cftime
from mpi4py import MPI
import mpi4py.util.dtlib
import h5py

import pygetm
from pygetm.constants import CENTERS, INTERFACES
from .. import simulator

_comm = MPI.COMM_WORLD.Dup()
size = _comm.Get_size()
rank = _comm.Get_rank()

verbose = True
use_root = True

matrix_types = ("periodic", "time_dependent", "constant")

class MatArray(pygetm.input.LazyArray):
    def __init__(self, path: str, name: str):
        transpose = False
        try:
            # First try MATLAB 7.3 format (=HDF5)
            vardict = self.file = h5py.File(path)
        except Exception:
            # Now try older MATLAB formats
            vardict = scipy.io.loadmat(path)
            transpose = True

        if name not in vardict:
            raise KeyError(
                f"Variable {name} not found in {path}."
                f" Available: {', '.join(vardict.keys())}"
            )

        self.var = vardict[name]
        if transpose:
            self.var = self.var.T

        name = f"MatArray({path!r}, {name!r})"
        super().__init__(self.var.shape, self.var.dtype, name)

    def __getitem__(self, slices) -> np.ndarray:
        return self.var[slices]


def get_mat_array(
    path: str, name: str, grid_file: str, times: Optional[npt.ArrayLike] = None
) -> xr.DataArray:
    """This routine should ultimately read .mat files in HDF5
    as well as proprietary MATLAB formats"""
    data = MatArray(path, name)
    bathy, ideep, x, y, z, dz, delta_t = _read_grid(grid_file)
    dims = ["y", "x"]
    coords = {}
    coords["lon"] = xr.DataArray(x[0, :], dims=("x",), attrs=dict(units="degrees_east"))
    coords["lat"] = xr.DataArray(
        y[0, :], dims=("y",), attrs=dict(units="degrees_north")
    )
    assert data.shape[-1] == x.size
    assert data.shape[-2] == y.size
    if data.ndim > 2 and data.shape[-3] == dz.shape[0]:
        z_if = np.zeros((dz.shape[0] + 1,) + dz.shape[1:], dz.dtype)
        z_if[1:] = dz.cumsum(axis=0)
        coords["zc"] = xr.DataArray(
            0.5 * (z_if[:-1, ...] + z_if[1:, ...]),
            dims=("z", "y", "x"),
            attrs=dict(standard_name="depth"),
        )
        dims.insert(0, "z")
    if times is not None:
        assert len(times) == data.shape[0]
        dims.insert(0, "time")
        coords["time"] = times
    dims = [f"dim_{i}" for i in range(data.ndim - len(dims))] + dims
    return xr.DataArray(data, coords=coords, dims=dims)


def map_input_to_compressed_grid(
    value: xr.DataArray,
    global_shape: Tuple[int],
    indices: Iterable[np.ndarray],
) -> Optional[xr.DataArray]:
    if value.shape[-2:] != global_shape:
        return

    # First select the region (x and y slice) that encompasses all points
    # in the local subdomain
    region_slices = tuple([slice(ind.min(), ind.max() + 1) for ind in indices])
    local_shape = value.shape[:-2] + tuple([s.stop - s.start for s in region_slices])
    s_region = pygetm.input.Slice(
        pygetm.input._as_lazyarray(value),
        shape=local_shape,
        passthrough=range(value.ndim - 2),
    )
    src_slices = (slice(None),) * (value.ndim - 2) + region_slices
    tgt_slices = (slice(None),) * value.ndim
    s_region._slices.append((src_slices, tgt_slices))

    # In the extracted region, index the individual points
    indices = tuple([ind - ind.min() for ind in indices])
    local_shape = value.shape[:-2] + (1,) + indices[0].shape
    s = pygetm.input.Slice(
        s_region,
        shape=local_shape,
        passthrough=range(value.ndim - 2),
    )
    src_slices = (slice(None),) * (value.ndim - 2) + indices
    tgt_slices = (slice(None),) * (value.ndim - 2) + (0, slice(None))
    s._slices.append((src_slices, tgt_slices))
    s.passthrough_own_slices = False

    # Keep only coordinates without any horizontal coordinate dimension (x, y)
    alldims = frozenset(value.dims[-2:])
    coords = {}
    for n, c in value.coords.items():
        if not alldims.intersection(c.dims):
            coords[n] = c

    return xr.DataArray(
        s, dims=value.dims, coords=coords, attrs=value.attrs, name=s.name
    )


class CompressedToFullGrid(pygetm.output.operators.UnivariateTransformWithData):
    def __init__(
        self,
        source: pygetm.output.operators.Base,
        grid: Optional[pygetm.domain.Grid],
        mapping: np.ndarray,
        global_array: Optional[pygetm.core.Array] = None,
    ):
        self._grid = grid
        self._slice = (mapping,)
        shape = source.shape[:-2] + (grid.ny, grid.nx)
        self._z = None
        if source.ndim > 2:
            self._z = CENTERS if source.shape[0] == grid.nz else INTERFACES
            self._slice = (slice(None),) + self._slice
        dims = pygetm.output.operators.grid2dims(grid, self._z)
        expression = f"{self.__class__.__name__}({source.expression})"
        super().__init__(source, shape=shape, dims=dims, expression=expression)
        self.values.fill(self.fill_value)
        self._global_values = None if global_array is None else global_array.values

    def get(
        self,
        out: Optional[npt.ArrayLike] = None,
        slice_spec: Tuple[int, ...] = (),
    ) -> npt.ArrayLike:
        compressed_values = self._source.get()
        if self._global_values is not None:
            if out is None:
                return self._global_values
            else:
                out[slice_spec] = self._global_values
                return out[slice_spec]
        self.values[self._slice] = compressed_values[..., 0, :]
        return super().get(out, slice_spec)

    @property
    def grid(self) -> pygetm.domain.Grid:
        return self._grid

    @property
    def coords(self):
        global_x = self._grid.lon if self._grid.domain.spherical else self._grid.x
        global_y = self._grid.lat if self._grid.domain.spherical else self._grid.y
        global_z = self._grid.zc if self._z == CENTERS else self._grid.zf
        globals_arrays = (global_x, global_y, global_z)
        for c, g in zip(self._source.coords, globals_arrays):
            yield CompressedToFullGrid(c, self._grid, self._slice[-1], g)


class TransportMatrix(pygetm.input.LazyArray):
    def __init__(
        self,
        file_list: list,
        name: str,
        offsets: np.ndarray,
        counts: np.ndarray,
        rank: int,
        comm: MPI.Comm,
        order: Optional[np.ndarray] = None,
        logger=None,
        power=1,
        scale_factor=1.0,
    ):
        if logger:
            logger.info(f"Initializing {name} arrays")
        self._file_list = file_list
        self._group_name = name

        # Obtain global sparse array indices and data type
        # Currently these are determined by reading the first global sparse matrix,
        # but in the future, they could might be read directly from file if the arrays
        # are stored (or have been transformed to) CSR format.
        self.global_indices = None
        self.global_indptr = None
        self.global_dtype = None
        if rank == 0:
            self._get_matrix_metadata(file_list[0], order=order)
        dtype = comm.bcast(self.global_dtype)
        indptr = comm.bcast(self.global_indptr)

        # Collect information for MPI scatter of sparse array indices and data
        # For this, we need arrays with sparse data offsets and counts for every rank
        self._comm = comm
        self._rank = rank
        self._counts = []
        self._offsets = []
        ioffset = 0
        for o, c in zip(offsets, counts):
            self._offsets.append(ioffset)
            self._counts.append(indptr[o + c] - indptr[o])
            ioffset += self._counts[-1]

        # Get local sparse array indices
        istartrow = offsets[rank]
        istoprow = istartrow + counts[rank]
        self.indptr = np.asarray(indptr[istartrow : istoprow + 1])
        self.indptr -= self.indptr[0]
        self.indices = np.empty((self.indptr[-1],), indptr.dtype)
        mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.indices.dtype)
        comm.Scatterv(
            [self.global_indices, self._counts, self._offsets, mpi_dtype], self.indices
        )
        self.dense_shape = (counts[rank], counts.sum())
        self._power = power
        self._scale_factor = scale_factor

        shape = (self.indices.size,)
        if len(file_list) > 1:
            shape = (len(file_list),) + shape
        super().__init__(shape, dtype.newbyteorder("="), name)

    def __getitem__(self, slices) -> np.ndarray:
        itime = 0 if len(self._file_list) == 1 else slices[0]
        assert isinstance(itime, (int, np.integer))
        if self._rank == 0:
            d = self._get_matrix_data(self._file_list[itime]).newbyteorder("=")
            if self._power != 1:
                csr = scipy.sparse.csr_matrix(
                    (d, self.global_indices, self.global_indptr)
                )
                d = (csr**self._power).data
        else:
            d = None
        values = np.empty((self.shape[-1],), self.dtype)
        self._comm.Scatterv([d, self._counts, self._offsets, MPI.DOUBLE], values)
        result = values[slices[1:]] if len(self._file_list) > 1 else values[slices]
        if self._scale_factor != 1.0:
            result *= self._scale_factor
        return result

    def _get_matrix_metadata(self, fn: str, order: Optional[np.ndarray] = None):
        try:
            # MATLAB >= 7.3 format (HDF5)
            with h5py.File(fn) as ds:
                group = ds[self._group_name]
                A_ir = np.asarray(group["ir"], dtype=np.intp)
                A_jc = np.asarray(group["jc"], dtype=np.intp)
                dtype = group["data"].dtype
        except OSError:
            # MATLAB < 7.3 format
            vardict = scipy.io.loadmat(fn)
            A = vardict[self._group_name]
            A_ir = A.indices
            A_jc = A.indptr
            dtype = A.dtype
        n = A_jc.size - 1

        if order is None:
            old2newindex = np.arange(n)
        else:
            assert order.shape == (n,)
            old2newindex = np.argsort(order)
            A_ir = old2newindex[A_ir]

        # Determine CSC to CSR mapping
        items_per_row = np.zeros((n,), dtype=int)
        np.add.at(items_per_row, A_ir, 1)

        self.index_map = np.empty_like(A_ir)
        indices = np.empty_like(A_ir)
        indptr = np.empty_like(A_jc)
        indptr[0] = 0
        indptr[1:] = items_per_row.cumsum()
        current_items_per_row = np.zeros_like(items_per_row)
        for icol, (colstart, colstop) in enumerate(zip(A_jc[:-1], A_jc[1:])):
            irows = A_ir[colstart:colstop]
            newi = indptr[irows] + current_items_per_row[irows]
            indices[newi] = old2newindex[icol]
            self.index_map[newi] = np.arange(colstart, colstop)
            current_items_per_row[irows] += 1
        self.global_indices = indices
        self.global_indptr = indptr
        self.global_dtype = dtype
        if False:
            assert (current_items_per_row == (indptr[1:] - indptr[:-1])).all()
            A_data = np.random.random(indices.shape)
            csr = scipy.sparse.csr_array((A_data[self.index_map], indices, indptr))
            csr2 = scipy.sparse.csc_array((A_data, A_ir, A_jc), (n, n)).tocsr()
            assert csr.nnz == csr2.nnz
            assert csr.shape == csr2.shape
            assert (csr.indices == csr2.indices).all()
            assert (csr.indptr == csr2.indptr).all()
            assert (csr.data == csr2.data).all()

    def _get_matrix_data(self, fn: str) -> np.ndarray:
        try:
            # MATLAB >= 7.3 format (HDF5)
            with h5py.File(fn) as ds:
                data = np.asarray(ds[self._group_name]["data"])
        except OSError:
            # MATLAB < 7.3 format
            vardict = scipy.io.loadmat(fn)
            data = vardict[self._group_name].data
        return data[self.index_map]

    def create_sparse_array(self) -> scipy.sparse.csr_array:
        return scipy.sparse.csr_array(
            (
                np.empty((self.shape[-1],), self.dtype),
                self.indices,
                self.indptr,
            ),
            self.dense_shape,
        )


def _read_grid(fn, logger=None):
    if rank == 0 and logger:
        logger.info(f"Reading grid information from: {fn}")
    if not os.path.isfile(fn):
        raise Exception(f"Grid file {fn} does not exist")
    with h5py.File(fn) as ds:
        bathy = np.asarray(ds["bathy"], dtype=int)
        ideep = np.asarray(ds["ideep"], dtype=int)
        x = np.asarray(ds["x"])
        y = np.asarray(ds["y"])
        z = np.asarray(ds["z"])
        dz = np.asarray(ds["dz"])
        delta_t = np.asarray(ds["deltaT"][0, 0])
    return bathy, ideep, x, y, z, dz, delta_t


def create_domain(path: str, logger=None) -> pygetm.domain.Domain:
    bathy, ideep, lon, lat, z, dz, delta_t = _read_grid(path, logger=logger)

    # If x,y coordinates are effectively 1D, transpose latitude to ensure
    # its singleton dimension comes last (x)
    if lon.shape[0] == 1 and lat.shape[0] == 1:
        lat = lat.T

    mask_hz = ideep != 0
    lon, lat, ideep = np.broadcast_arrays(lon, lat, ideep)
    H = dz.sum(axis=0)

    # Squeeze out columns with only land. This maps the horizontal from 2D to 1D
    lon_packed = lon[mask_hz]
    lat_packed = lat[mask_hz]
    ideep_packed = ideep[mask_hz]
    H_packed = H[mask_hz]

    # Simple subdomain division along x dimension
    tiling = pygetm.parallel.Tiling(nrow=1, ncol=_comm.size, comm=_comm)

    nx, ny, nz = lon_packed.shape[-1], 1, bathy.shape[0]
    domain = pygetm.domain.create(
        nx,
        ny,
        nz,
        lon=lon_packed[np.newaxis, :],
        lat=lat_packed[np.newaxis, :],
        mask=1,
        H=H_packed[np.newaxis, :],
        spherical=True,
        tiling=tiling,
    )

    slc_loc, slc_glob, shape_loc, _ = domain.tiling.subdomain2slices()
    ideep_loc = np.zeros(shape_loc, dtype=ideep.dtype)
    ideep_loc[slc_loc] = ideep_packed[np.newaxis, :][slc_glob]

    # 3D mask for local subdomain
    assert ideep_loc.shape[0] == 1
    domain.wet_loc = np.arange(1, nz + 1) <= ideep_loc[0, :, np.newaxis]

    wet_glob = np.arange(1, nz + 1) <= ideep[:, :, np.newaxis]  # y, x, z
    wet_glob_tmm_order = np.moveaxis(wet_glob, 2, 0)  # z, y, x
    nwet_glob = wet_glob.sum()
    order = np.empty_like(wet_glob, dtype=int)
    order_tmm = np.moveaxis(order, 2, 0)
    order_tmm[wet_glob_tmm_order] = np.arange(nwet_glob)
    order = order[wet_glob]
    assert order.shape == (nwet_glob,)
    assert (np.sort(order) == np.arange(nwet_glob)).all()
    domain.order = order

    domain.nwet = ideep_loc.sum()
    domain.counts = np.empty(domain.tiling.n, dtype=int)
    domain.offsets = np.zeros_like(domain.counts)
    domain.tiling.comm.Allgather(domain.nwet, domain.counts)
    nwets_cum = domain.counts.cumsum()
    domain.offsets[1:] = nwets_cum[:-1]
    domain.nwet_tot = nwets_cum[-1]
    offset = domain.offsets[domain.tiling.rank]
    domain.tmm_slice = slice(offset, offset + domain.nwet)
    domain.dz = dz[:, np.newaxis, mask_hz]
    global_indices = np.indices(ideep.shape)
    local_indices = [i[mask_hz][slc_glob[-1]] for i in global_indices]
    domain.input_grid_mappers.append(
        functools.partial(
            map_input_to_compressed_grid,
            indices=local_indices,
            global_shape=mask_hz.shape,
        )
    )
    domain._grid_file = path
    domain._delta_t = delta_t

    if domain.tiling.rank == 0:
        tiling = pygetm.parallel.Tiling(nrow=1, ncol=1, ncpus=1)
        tiling.rank = 0
        full_domain = pygetm.domain.create(
            lon.shape[-1],
            lon.shape[-2],
            nz,
            lon=lon,
            lat=lat,
            mask=np.where(mask_hz, 1, 0),
            H=H,
            spherical=True,
            tiling=tiling,
        )
        full_domain.initialize(pygetm.BAROCLINIC)
        _update_vertical_coordinates(full_domain.T, dz)

        # By default, transform compressed fields to original 3D domain on output
        tf = functools.partial(
            CompressedToFullGrid, grid=full_domain.T, mapping=mask_hz
        )
        domain.default_output_transforms.append(tf)

    if False:
        # Testing:
        # Initialize a local 3D array (nz, 1, nx) with horizontal indices,
        # then allgather that array and verify its minimum (0), maximum (nwet_tot-1),
        # and the difference between consecutive values (0 if from same column,
        # 1 if from different ones)
        v_all = np.empty(domain.nwet_tot)
        v = np.empty((nz, 1, domain.tiling.nx_sub), dtype=float)
        v[...] = np.arange(domain.tiling.xoffset, domain.tiling.xoffset + v.shape[-1])
        domain.tiling.comm.Allgatherv(
            [v.T[domain.wet_loc], MPI.DOUBLE],
            (v_all, domain.counts, domain.offsets, MPI.DOUBLE),
        )
        assert v_all.min() == 0 and v_all.max() == domain.nwet_tot - 1
        v_diff = np.diff(v_all)
        assert v_diff.min() == 0 and v_diff.max() == 1

    return domain


def _update_vertical_coordinates(grid: pygetm.domain.Grid, dz: np.ndarray):
    slc_loc, slc_glob, _, _ = grid.domain.tiling.subdomain2slices()
    grid.ho.values[slc_loc] = dz[slc_glob]
    grid.hn.values[slc_loc] = dz[slc_glob]
    grid.zf.all_values.fill(0.0)
    grid.zf.all_values[1:, ...] = -grid.hn.all_values.cumsum(axis=0)
    grid.zc.all_values[...] = 0.5 * (
        grid.zf.all_values[:-1, :, :] + grid.zf.all_values[1:, :, :]
    )
    grid.zc.all_values[:, grid._land] = 0.0
    grid.zf.all_values[:, grid._land] = 0.0
    grid.ho.all_values[grid.ho.all_values == 0.0] = grid.ho.fill_value
    grid.hn.all_values[grid.hn.all_values == 0.0] = grid.hn.fill_value
    grid.ho.attrs["_time_varying"] = False
    grid.hn.attrs["_time_varying"] = False
    grid.zc.attrs["_time_varying"] = False
    grid.zf.attrs["_time_varying"] = False


def climatology_times(calendar="standard"):
    return [cftime.datetime(2000, imonth + 1, 16, calendar=calendar) for imonth in range(12)]


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        matrix_files: Iterable[str],
        matrix_times: Iterable[Union[datetime.datetime, cftime.datetime]],
        fabm_config: str = "fabm.yaml",
    ):
        super().__init__(domain, fabm_config)
        self.tmm_logger = self.logger.getChild("TMM")
        self.tmm_logger.info(f"Initializing TMM component")
        _update_vertical_coordinates(self.domain.T, self.domain.dz)
        if self.domain.glob and self.domain.glob is not self.domain:
            _update_vertical_coordinates(self.domain.glob.T, self.domain.dz)

        #self._tmm_matrix_config(tmm_config)
        self.Aexp_src = TransportMatrix(
            matrix_files,
            "Aexp",
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
            order=domain.order,
        )
        self.Aimp_src = TransportMatrix(
            matrix_files,
            "Aimp",
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
            order=domain.order,
        )
        self.Aexp = self.Aexp_src.create_sparse_array()
        self.Aimp = self.Aimp_src.create_sparse_array()
        self._matrix_times = matrix_times
        if self.Aexp_src.ndim == 2:
            Aexp_tip = pygetm.input.TemporalInterpolation(
                self.Aexp_src, 0, self._matrix_times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aexp_tip.name, Aexp_tip, self.Aexp.data)
            )
        if self.Aimp_src.ndim == 2:
            Aimp_tip = pygetm.input.TemporalInterpolation(
                self.Aimp_src, 0, self._matrix_times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aimp_tip.name, Aimp_tip, self.Aimp.data)
            )

        self.load_environment()

    def start(
        self,
        time: Union[cftime.datetime, datetime.datetime],
        timestep: float,
        nstep_transport: int = 1,
        report: datetime.timedelta = datetime.timedelta(days=1),
    ):
        dt = nstep_transport * timestep
        nphys = dt / self.domain._delta_t
        self.tmm_logger.info(
            f"Using transport timestep of {dt} s, which is {nphys} *"
            f" the original online timestep of {self.domain._delta_t} s."
        )
        assert (nphys % 1.0) < 1e-8
        self.Aexp_src._scale_factor = dt
        self.Aimp_src._power = int(nphys)
        if self.Aexp_src.ndim == 1:
            self.Aexp.data[:] = self.Aexp_src
        if self.Aimp_src.ndim == 1:
            self.Aimp.data[:] = self.Aimp_src
        super().start(time, timestep, nstep_transport, report)

    def transport(self, timestep: float):
        packed_values = np.empty(self.domain.nwet_tot, self.Aexp.dtype)
        rcvbuf = (packed_values, self.domain.counts, self.domain.offsets, MPI.DOUBLE)
        assert self.Aexp_src._scale_factor == timestep
        assert self.Aexp_src._power == 1
        for tracer in self.tracers:
            tracer_tmm = tracer.values[:, 0, :].T

            # packed_values is the packed TMM-style array with tracer values
            self.domain.tiling.comm.Allgatherv(tracer_tmm[self.domain.wet_loc], rcvbuf)

            # do the explicit step
            vtmp = self.Aexp.dot(packed_values)
            packed_values[self.domain.tmm_slice] += vtmp

            # everything back to all processes
            self.domain.tiling.comm.Allgatherv(MPI.IN_PLACE, rcvbuf)

            # do the implicit step
            vtmp = self.Aimp.dot(packed_values)

            # Copy updated tracer values back to authoratitive array
            tracer_tmm[self.domain.wet_loc] = vtmp

    def load_environment(self):
        root = os.path.dirname(self.domain._grid_file)

        def _get_variable(path: str, varname: str, **kwargs) -> pygetm.core.Array:
            array = self.domain.T.array(**kwargs)
            src = get_mat_array(
                path,
                varname,
                self.domain._grid_file,
                times=self._matrix_times,
            )
            array.set(src, on_grid=pygetm.input.OnGrid.ALL, climatology=True)
            return array

        self.temp = _get_variable(
            os.path.join(root, "GCM/Theta_gcm.mat"),
            "Tgcm",
            z=CENTERS,
            name="temp",
            units="degrees_Celsius",
            long_name="temperature",
            fabm_standard_name="temperature",
        )
        self.salt = _get_variable(
            os.path.join(root, "GCM/Salt_gcm.mat"),
            "Sgcm",
            z=CENTERS,
            name="salt",
            units="PSU",
            long_name="salinity",
            fabm_standard_name="practical_salinity",
        )
        self.wind = _get_variable(
            os.path.join(root, "BiogeochemData/wind_speed.mat"),
            "windspeed",
            name="wind",
            units="m s-1",
            long_name="wind speed",
            fabm_standard_name="wind_speed",
        )
        self.fice = _get_variable(
            os.path.join(root, "BiogeochemData/ice_fraction.mat"),
            "Fice",
            name="ice",
            units="1",
            long_name="ice cover",
            fabm_standard_name="ice_area_fraction",
        )

    def _tmm_matrix_config(self, config: dict):
        assert config["matrix_type"] in matrix_types
        assert config["path"]

        if config["matrix_type"] == "constant":
            constant = config["constant"]
            assert constant["Ae_fname"]
            assert constant["Ai_fname"]
            self._matrix_paths_Ae = list(
                os.path.join(config["path"], constant["Ae_fname"])
            )
            self._matrix_paths_Ai = list(
                os.path.join(config["path"], constant["Ai_fname"])
            )
        if config["matrix_type"] == "periodic":
            periodic = config["periodic"]
            assert periodic["num_periods"] > 0
            assert periodic["Ae_template"]
            if "base_number" in periodic:
                offset = periodic["base_number"]
            else:
                offset = 0
            self._matrix_paths_Ae = [
                os.path.join(
                    config["path"],
                    periodic["Ae_template"] % (i + offset),
                )
                for i in range(periodic["num_periods"])
            ]
            self._matrix_paths_Ai = [
                os.path.join(
                    config["path"],
                    periodic["Ai_template"] % (i + offset),
                )
                for i in range(periodic["num_periods"])
            ]

        if config["matrix_type"] == "time_dependent":
            print("time varying TMM matrices not implemented yet - except for periodic")
            sys.exit()

