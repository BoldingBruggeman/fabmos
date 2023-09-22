import os
import functools
import datetime
import logging
from typing import Optional, Tuple, Iterable, Union, Mapping, Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
import cftime
import xarray as xr
import h5py

import pygetm
import pygetm.parallel
from pygetm.constants import CENTERS, INTERFACES
from .. import simulator, environment, Array

# Note: mpi4py components should be imported after pygetm.parallel,
# as the latter configures mpi4py.rc
from mpi4py import MPI
import mpi4py.util.dtlib


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

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.var, dtype=dtype)

    def __getitem__(self, slices) -> np.ndarray:
        return self.var[slices]


def get_mat_array(
    path: str, name: str, grid_file: str, times: Optional[npt.ArrayLike] = None
) -> xr.DataArray:
    """This routine should ultimately read .mat files in HDF5
    as well as proprietary MATLAB formats"""
    return _wrap_ongrid_array(MatArray(path, name), grid_file, times)


def _wrap_ongrid_array(
    data: MatArray, grid_file: str, times: Optional[npt.ArrayLike] = None
) -> xr.DataArray:
    x, y, dz = _load_mat(grid_file, "x", "y", "dz")
    dims = ["y", "x"]
    coords = {}
    coords["lon"] = xr.DataArray(x, dims=("x",), attrs=dict(units="degrees_east"))
    coords["lat"] = xr.DataArray(y, dims=("y",), attrs=dict(units="degrees_north"))
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

    region_slices = tuple([slice(ind.min(), ind.max() + 1) for ind in indices])
    all_region_slices = (slice(None),) * (value.ndim - 2) + region_slices
    indices = tuple([ind - ind.min() for ind in indices])

    if isinstance(value, np.ndarray):
        final_slices = (slice(None),) * (value.ndim - 2) + (np.newaxis,) + indices
        return value[all_region_slices][final_slices]

    # First select the region (x and y slice) that encompasses all points
    # in the local subdomain
    local_shape = value.shape[:-2] + tuple([s.stop - s.start for s in region_slices])
    s_region = pygetm.input.Slice(
        pygetm.input._as_lazyarray(value),
        shape=local_shape,
        passthrough=range(value.ndim - 2),
    )
    s_region._slices.append((all_region_slices, (slice(None),) * value.ndim))

    # In the extracted region, index the individual points
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
        global_array: Optional[Array] = None,
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
    MAX_CACHE_SIZE = 1024**3

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
        localize: bool = False,
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
        self.diag_indices = self._get_diagonal_indices(
            self.indptr, self.indices - istartrow
        )

        self._power = 1
        self._scale_factor = 1.0
        self._add_identity = False

        if localize:
            assert ((self.indices >= istartrow) & (self.indices < istoprow)).all()
            self.indices -= istartrow
            self.dense_shape = (counts[rank], counts[rank])
        else:
            self.dense_shape = (counts[rank], counts.sum())

        nbytes = float(indptr[-1]) * dtype.itemsize * len(file_list)
        self._cache = {}
        self._use_cache = nbytes < self.MAX_CACHE_SIZE and len(file_list) > 1

        shape = (self.indices.size,)
        if len(file_list) > 1:
            shape = (len(file_list),) + shape
        super().__init__(shape, dtype.newbyteorder("="), name)

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value: int):
        self._cache.clear()
        self._power = value

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float):
        self._cache.clear()
        self._scale_factor = value

    @property
    def add_identity(self) -> bool:
        return self._add_identity

    @add_identity.setter
    def add_identity(self, value: bool):
        self._cache.clear()
        self._add_identity = value

    def __getitem__(self, slices) -> np.ndarray:
        itime = 0 if len(self._file_list) == 1 else slices[0]
        assert isinstance(itime, (int, np.integer))
        if itime in self._cache:
            return self._cache[itime][slices[-1]]
        if self._rank == 0:
            d = self._get_matrix_data(self._file_list[itime]).newbyteorder("=")
        else:
            d = None
        values = np.empty((self.shape[-1],), self.dtype)
        self._comm.Scatterv([d, self._counts, self._offsets, MPI.DOUBLE], values)
        if self._scale_factor != 1.0:
            values *= self._scale_factor
        if self._add_identity:
            values[self.diag_indices] += 1.0
        if self._power != 1:
            # Matrix power for local block (current subdomain only)
            csr = self.create_sparse_array(values, scipy.sparse.csr_matrix)
            csr2 = csr**self._power
            csr2.sort_indices()
            assert (csr.indices == csr2.indices).all()
            assert (csr.indptr == csr2.indptr).all()
            values = csr2.data
        if self._use_cache:
            self._cache[itime] = values
        return values[slices[-1]]

    def __array__(self, dtype=None) -> np.ndarray:
        assert len(self._file_list) == 1
        return self[(slice(None),)]

    def _get_diagonal_indices(
        self, indptr: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        diag_indices = np.empty((indptr.size - 1,), dtype=indptr.dtype)
        for irow, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:])):
            idiag = (indices[start:stop] == irow).nonzero()[0]
            assert idiag.size == 1
            diag_indices[irow] = start + idiag
        return diag_indices

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

        old2newindex = np.arange(n)
        if order is not None:
            assert order.shape == (n,)
            old2newindex[order] = np.arange(n)
            A_ir[:] = old2newindex[A_ir]

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

    def create_sparse_array(
        self, data: Optional[np.ndarray] = None, type=scipy.sparse.csr_array
    ) -> scipy.sparse.csr_array:
        if data is None:
            data = np.empty((self.shape[-1],), self.dtype)
        return type(
            (data, self.indices, self.indptr),
            self.dense_shape,
        )


def _read_grid(fn: str):
    if not os.path.isfile(fn):
        raise Exception(f"Grid file {fn} does not exist")
    with h5py.File(fn) as ds:
        bathy = np.asarray(ds["bathy"], dtype=int)
        ideep = np.asarray(ds["ideep"], dtype=int)
        x = np.asarray(ds["x"])
        y = np.asarray(ds["y"])
        z = np.asarray(ds["z"])
        dz = np.asarray(ds["dz"])
        da = np.asarray(ds["da"])
        delta_t = np.asarray(ds["deltaT"][0, 0])
    return bathy, ideep, x, y, z, dz, da, delta_t


def _read_config(fn: str) -> Mapping[str, Any]:
    if not os.path.isfile(fn):
        raise Exception(f"Configuration file {fn} does not exist")
    config = {}
    try:
        # MATLAB >= 7.3 format (HDF5)
        with h5py.File(fn) as ds:
            for k in ds.keys():
                v = np.squeeze(ds[k][...])
                if v.dtype == "<u2":
                    v = bytes(v[:]).decode("utf16")
                config[k] = v
    except OSError:
        # MATLAB < 7.3 format
        for k, v in scipy.io.loadmat(fn).items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, "U"):
                v = str(v[0])
            config[k] = v
    for k in ("fixEmP", "rescaleForcing", "useAreaWeighting"):
        config[k] = (config[k] != 0).any()
    return config


def _load_mat(
    fn: str, *names: str, dtype: Optional[npt.DTypeLike] = None
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    if not os.path.isfile(fn):
        raise Exception(f"File {fn} does not exist")

    values = []
    try:
        # MATLAB >= 7.3 format (HDF5)
        with h5py.File(fn) as ds:
            for name in names:
                values.append(np.asarray(ds[name], dtype=dtype))
    except OSError:
        # MATLAB < 7.3 format
        name2values = scipy.io.loadmat(fn)
        for name in names:
            values.append(name2values[name])

    def _process(data):
        while data.ndim and data.shape[0] == 1:
            data = data[0, ...]
        return data
    values = [_process(v) for v in values]

    return values[0] if len(names) == 1 else tuple(values)


def create_domain(
    path: str, logger: Optional[logging.Logger] = None, comm: Optional[MPI.Comm] = None
) -> pygetm.domain.Domain:
    if os.path.isdir(path):
        path = os.path.join(path, "grid.mat")

    logger = logger or pygetm.parallel.get_logger()
    logger.info(f"Reading grid information from: {path}")
    bathy, ideep, lon, lat, z, dz, da, delta_t = _read_grid(path)

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
    tiling = pygetm.parallel.Tiling(nrow=1, comm=comm)

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
        logger=logger,
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
    domain.ideep_loc = ideep_loc
    domain.counts = np.empty(domain.tiling.n, dtype=int)
    domain.offsets = np.zeros_like(domain.counts)
    domain.tiling.comm.Allgather(domain.nwet, domain.counts)
    nwets_cum = domain.counts.cumsum()
    domain.offsets[1:] = nwets_cum[:-1]
    domain.nwet_tot = nwets_cum[-1]
    offset = domain.offsets[domain.tiling.rank]
    domain.tmm_slice = slice(offset, offset + domain.nwet)
    domain.dz = dz[:, np.newaxis, mask_hz]
    domain.da = da[0, mask_hz][np.newaxis, :]
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
        logger.info("Creating uncompressed global domain for output")
        tiling = pygetm.parallel.Tiling(nrow=1, ncol=1, ncpus=1, comm=tiling.comm)
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
            logger=domain.root_logger,
        )
        full_domain.initialize(pygetm.BAROCLINIC)
        _update_coordinates(full_domain.T, dz, da[0, ...])

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


def _update_coordinates(grid: pygetm.domain.Grid, dz: np.ndarray, da: np.ndarray):
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
    grid.domain.depth.all_values[...] = -grid.zc.all_values
    grid.ho.all_values[grid.ho.all_values == 0.0] = grid.ho.fill_value
    grid.hn.all_values[grid.hn.all_values == 0.0] = grid.hn.fill_value
    grid.ho.attrs["_time_varying"] = False
    grid.hn.attrs["_time_varying"] = False
    grid.zc.attrs["_time_varying"] = False
    grid.zf.attrs["_time_varying"] = False
    grid.domain.depth.attrs["_time_varying"] = False

    grid.area.values[slc_loc] = da[slc_glob]
    grid.D.values[slc_loc] = grid.H.values[slc_loc]


def climatology_times(calendar="standard"):
    return [
        cftime.datetime(2000, imonth + 1, 16, calendar=calendar) for imonth in range(12)
    ]


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        periodic_matrix: bool = True,
        periodic_forcing: bool = True,
        calendar: str = "standard",
        fabm_config: str = "fabm.yaml",
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_tmm")
        domain.mask3d = domain.T.array(z=CENTERS, dtype=np.intc, fill=0)
        domain.mask3d.values[:, 0, :] = np.where(domain.wet_loc.T, 1, 0)
        domain.T._land3d = domain.mask3d.all_values == 0
        domain.bottom_indices = domain.T.array(dtype=np.intc, fill=0)
        domain.bottom_indices.values[0, :] = domain.ideep_loc

        super().__init__(domain, fabm_config, fabm_libname=fabm_libname)
        self.tmm_logger = self.logger.getChild("TMM")
        _update_coordinates(self.domain.T, self.domain.dz, self.domain.da)
        if self.domain.glob and self.domain.glob is not self.domain:
            _update_coordinates(self.domain.glob.T, self.domain.dz, self.domain.da)

        root = os.path.dirname(self.domain._grid_file)
        config = _read_config(os.path.join(root, "config_data.mat"))
        Ir_pre = _load_mat(
            os.path.join(root, config["matrixPath"], "Data/profile_data.mat"),
            "Ir_pre",
            dtype=int,
        )
        assert (Ir_pre - 1 == domain.order).all()

        Aexp_files = []
        Aimp_files = []
        if periodic_matrix:
            matrix_times = climatology_times(calendar)
            exp_base = config["explicitMatrixFileBase"]
            imp_base = config["implicitMatrixFileBase"]
            for imonth in range(1, 13):
                Aexp_files.append(os.path.join(root, exp_base + f"_{imonth:02}.mat"))
                Aimp_files.append(os.path.join(root, imp_base + f"_{imonth:02}.mat"))
            Aexp_name = "Aexp"
            Aimp_name = "Aimp"
        else:
            matrix_times = None
            exp_base = config["explicitAnnualMeanMatrixFile"]
            imp_base = config["implicitAnnualMeanMatrixFile"]
            Aexp_files.append(os.path.join(root, exp_base + ".mat"))
            Aimp_files.append(os.path.join(root, imp_base + ".mat"))
            Aexp_name = "Aexpms"
            Aimp_name = "Aimpms"

        self.Aexp_src = TransportMatrix(
            Aexp_files,
            Aexp_name,
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
            order=domain.order,
        )
        self.Aimp_src = TransportMatrix(
            Aimp_files,
            Aimp_name,
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
            order=domain.order,
            localize=True,
        )
        self.Aexp = self.Aexp_src.create_sparse_array()
        self.Aimp = self.Aimp_src.create_sparse_array()
        if self.Aexp_src.ndim == 2:
            Aexp_tip = pygetm.input.TemporalInterpolation(
                self.Aexp_src, 0, matrix_times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aexp_tip.name, Aexp_tip, self.Aexp.data)
            )
        if self.Aimp_src.ndim == 2:
            Aimp_tip = pygetm.input.TemporalInterpolation(
                self.Aimp_src, 0, matrix_times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aimp_tip.name, Aimp_tip, self.Aimp.data)
            )

        self.load_environment(periodic_forcing, calendar)
        self._redistribute: Tuple[
            Array, Array, Array, Optional[Array], Union[datetime.timedelta, int], int
        ] = []

        self.tot_area = domain.T.area.global_sum(where=domain.T.mask == 1, to_all=True)

    def request_redistribution(
        self,
        source: str,
        target: str,
        weights: Optional[Array],
        update_interval: Union[datetime.timedelta, int],
        initial_value: float = 0.0,
    ):
        for v, source_array in self.fabm._variable2array.items():
            if v.name == source:
                break
        else:
            names = [v.name for v in self.fabm._variable2array]
            raise Exception(
                f"Variable {source} not found in FABM model."
                f" Available: {', '.join(names)}"
            )
        source_array.saved = True
        target_array = self.fabm.get_dependency(target)
        target_array.fill(initial_value)
        cum_source = source_array.grid.array(fill=0.0)
        iupdate = update_interval if isinstance(update_interval, int) else None
        self._redistribute.append(
            [source_array, target_array, cum_source, weights, update_interval, iupdate]
        )

    def start(
        self,
        time: Union[cftime.datetime, datetime.datetime],
        timestep: Union[float, datetime.timedelta],
        transport_timestep: Optional[Union[float, datetime.timedelta]] = None,
        report: datetime.timedelta = datetime.timedelta(days=1),
        report_totals: Union[int, datetime.timedelta] = datetime.timedelta(days=10),
        profile: Optional[str] = None,
    ):
        if isinstance(timestep, datetime.timedelta):
            timestep = timestep.total_seconds()
        if isinstance(transport_timestep, datetime.timedelta):
            transport_timestep = transport_timestep.total_seconds()

        dt = transport_timestep or timestep
        nphys = dt / self.domain._delta_t
        self.tmm_logger.info(
            f"Transport timestep of {dt} s is {nphys} *"
            f" the original online timestep of {self.domain._delta_t} s."
        )
        if nphys % 1.0 > 1e-8:
            raise Exception(
                f"The transport timestep of {dt} s must be an exact multiple"
                f" of the original online timestep of {self.domain._delta_t} s"
            )

        self.Aexp_src.scale_factor = dt
        self.Aexp_src.add_identity = True
        self.Aimp_src.power = int(round(nphys))
        if self.Aexp_src.ndim == 1:
            self.Aexp.data[:] = self.Aexp_src
        if self.Aimp_src.ndim == 1:
            self.Aimp.data[:] = self.Aimp_src

        super().start(
            time, timestep, transport_timestep, report, report_totals, profile
        )

        ntracer = len(self.tracers)
        tracer_dtype = self.domain.T.H.dtype
        self.global_tracers = np.empty((self.domain.nwet_tot, ntracer), tracer_dtype)
        self.local_tracers = self.global_tracers[self.domain.tmm_slice, :]
        for i, tracer in enumerate(self.tracers):
            self.local_tracers[:, i] = tracer.values[:, 0, :].T[self.domain.wet_loc]
        recvbuf = (
            self.global_tracers,
            self.domain.counts * ntracer,
            self.domain.offsets * ntracer,
            MPI.DOUBLE,
        )
        self.gather_tracer = functools.partial(
            self.domain.tiling.comm.Iallgatherv, MPI.IN_PLACE, recvbuf
        )
        self.gather_req = self.gather_tracer()

        for info in self._redistribute:
            interval = info[-2]
            if isinstance(interval, datetime.timedelta):
                info[-1] = max(1, int(round(interval.total_seconds() / timestep)))
            self.logger.info(
                f"Flux redistribution: {info[1].name} will be updated "
                f"every {info[-1]} steps."
            )

    def _preload_transport_matrices(self):
        if self.Aexp_src.ndim == 2 and self.Aexp_src._use_cache:
            self.tmm_logger.info("Preloading all explicit matrices")
            for itime in range(self.Aexp_src.shape[0]):
                self.Aexp_src[itime, :]
        if self.Aimp_src.ndim == 2 and self.Aexp_src._use_cache:
            self.tmm_logger.info("Preloading all implicit matrices")
            for itime in range(self.Aimp_src.shape[0]):
                self.Aimp_src[itime, self.domain.tmm_slice]

    def advance_fabm(self, timestep: float):
        for source, target, cum_source, w, _, freq in self._redistribute:
            cum_source.all_values += source.all_values
            if self.istep % freq == 0:
                self.logger.info(
                    f"Deriving {target.name} from horizontally integrated {source.name}"
                )
                np.divide(cum_source.all_values, freq, out=target.all_values)
                target.all_values *= target.grid.area.all_values
                unmasked = target.grid.mask == 1
                int_flux = target.global_sum(where=unmasked, to_all=True)
                if w is not None:
                    w_int = (w * w.grid.area).global_sum(where=unmasked, to_all=True)
                    scale_factor = int_flux / w_int
                    self.logger.info(
                        f"  global scale factor for {target.name} "
                        f"= {scale_factor} {source.units} ({w.units})-1"
                    )
                    values = w.values * scale_factor
                else:
                    values = int_flux / self.tot_area
                    self.logger.info(
                        f"  global mean {source.name} = {values} {source.units}"
                    )
                target.fill(values)
                cum_source.all_values.fill(0.0)

        return super().advance_fabm(timestep)

    def transport(self, timestep: float):
        # Compute the change in tracers due to biogeochemical [FABM] processes
        # This is done by collecting the updated compressed state
        # and subtracting the previous compressed state
        fabm_tracer_change = np.empty_like(self.local_tracers)
        for i, tracer in enumerate(self.tracers):
            fabm_tracer_change[:, i] = tracer.values[:, 0, :].T[self.domain.wet_loc]
        fabm_tracer_change -= self.local_tracers

        # Ensure the compressed global tracer state is synchronized across subdomains
        MPI.Request.Wait(self.gather_req)

        # Do the explicit step
        new_local_tracers = self.Aexp.dot(self.global_tracers)

        # Add the biogeochemical/FABM tracer increment
        new_local_tracers += fabm_tracer_change

        # Do the implicit step
        self.local_tracers[:, :] = self.Aimp.dot(new_local_tracers)

        # Start synchronizing the compressed tracer state across subdomains
        self.gather_req = self.gather_tracer()

        # Copy the updated compressed state back to the uncompressed state
        for i, tracer in enumerate(self.tracers):
            tracer.values[:, 0, :].T[self.domain.wet_loc] = self.local_tracers[:, i]

    def load_environment(self, periodic: bool, calendar: str):
        root = os.path.dirname(self.domain._grid_file)

        times = climatology_times(calendar)

        def _get_variable(
            path: str, varname: str, **kwargs
        ) -> Tuple[Array, xr.DataArray]:
            array = self.domain.T.array(**kwargs)
            src = get_mat_array(path, varname, self.domain._grid_file, times=times)
            _set(array, src)
            return array, src

        def _set(array, src):
            if not periodic:
                src = src.mean(dim="time")
                array.attrs["_time_varying"] = False
            array.set(src, on_grid=pygetm.input.OnGrid.ALL, climatology=periodic)

        self.temp, temp_src = _get_variable(
            os.path.join(root, "GCM/Theta_gcm.mat"),
            "Tgcm",
            z=CENTERS,
            name="temp",
            units="degrees_Celsius",
            long_name="temperature",
            fabm_standard_name="temperature",
        )
        self.salt, salt_src = _get_variable(
            os.path.join(root, "GCM/Salt_gcm.mat"),
            "Sgcm",
            z=CENTERS,
            name="salt",
            units="PSU",
            long_name="salinity",
            fabm_standard_name="practical_salinity",
        )
        if self.fabm.has_dependency("density"):
            self.rho = self.domain.T.array(
                z=CENTERS,
                name="rho",
                units="kg m-3",
                long_name="density",
                fabm_standard_name="density",
            )
            self.rho.set(
                environment.density(salt_src, temp_src),
                on_grid=pygetm.input.OnGrid.ALL,
                climatology=periodic,
            )

        self.wind, _ = _get_variable(
            os.path.join(root, "BiogeochemData/wind_speed.mat"),
            "windspeed",
            name="wind",
            units="m s-1",
            long_name="wind speed",
            fabm_standard_name="wind_speed",
        )
        self.fice, _ = _get_variable(
            os.path.join(root, "BiogeochemData/ice_fraction.mat"),
            "Fice",
            name="ice",
            units="1",
            long_name="ice cover",
            fabm_standard_name="ice_area_fraction",
        )

        fwf_path = os.path.join(root, "GCM/FreshWaterForcing_gcm.mat")
        tau = _load_mat(fwf_path, "saltRelaxTimegcm")
        if np.any(tau):
            self.logger.info(
                "Reconstructing net freshwater flux from salinity relaxation"
                " in original hydrodynamic simulation"
            )
            assert np.ndim(tau) == 0
            Srelax_src = MatArray(fwf_path, "Srelaxgcm")
            dz = _load_mat(self.domain._grid_file, "dz")
            salt_sf_src = pygetm.input.Slice(
                pygetm.input._as_lazyarray(salt_src),
                shape=Srelax_src.shape,
                passthrough={0: 0, 1: 2, 2: 3},
            )
            src_slice = (slice(None), 0, slice(None), slice(None))
            tgt_slice = (slice(None), slice(None), slice(None))
            salt_sf_src._slices.append((src_slice, tgt_slice))

            # The change in surface salinity due to relaxation is
            #   (S_relax - S) / tau (# m-3 s-1)
            # Multiply with top layer thickness dz to obtain a surface flux (# m-2 s-1)
            # This must equal the virtual salt flux of S * fwf
            # Thus, fwf = (S_relax / S - 1) * dz / tau
            emp_src = (Srelax_src / salt_sf_src - 1.0) * (dz[0, ...] / tau)
        else:
            self.logger.info(
                "Using freshwater flux from original hydrodynamic simulation"
            )
            emp_src = MatArray(fwf_path, "EmPgcm")
        emp_src = _wrap_ongrid_array(emp_src, self.domain._grid_file, times=times)
        self.emp = self.domain.T.array(
            name="emp", units="m s-1", long_name="evaporation minus precipitation"
        )
        _set(self.emp, emp_src)
