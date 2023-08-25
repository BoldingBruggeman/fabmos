import sys
import os
import functools
from typing import Optional, Tuple

import scipy.sparse

import xarray as xr
import numpy as np
import numpy.typing as npt
import cftime
from mpi4py import MPI
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
        self.file = h5py.File(path)
        if name not in self.file:
            raise KeyError(
                f"Variable {name} not found in {path}."
                f" Available: {', '.join(self.file.keys())}"
            )
        self.var = self.file[name]
        name = f"MatArray({path!r}, {name!r})"
        super().__init__(self.var.shape, self.var.dtype, name)

    def __getitem__(self, slices) -> np.ndarray:
        return self.var[slices]


def get_mat_array(path: str, name: str, grid_file: str) -> xr.DataArray:
    """This routine should ultimately read .mat files in HDF5
    as well as proprietary MATLAB formats"""
    data = MatArray(path, name)
    bathy, ideep, x, y, z, dz = _read_grid(grid_file)
    dims = ["y", "x"]
    coords = {}
    coords["lon"] = xr.DataArray(x[0, :], dims=("x",))
    coords["lat"] = xr.DataArray(y[0, :], dims=("y",))
    print(data.shape)
    assert data.shape[-1] == x.size
    assert data.shape[-2] == y.size
    if data.ndim > 2 and data.shape[-3] == dz.shape[0]:
        z_if = np.zeros((dz.shape[0] + 1,) + dz.shape[1:], dz.dtype)
        z_if[1:] = dz.cumsum(axis=0)
        coords["zc"] = xr.DataArray(
            0.5 * (z_if[:-1, ...] + z_if[1:, ...]), dims=("z", "y", "x")
        )
        dims.insert(0, "z")
    dims = [f"dim_{i}" for i in range(data.ndim - len(dims))] + dims
    ar = xr.DataArray(data, coords=coords, dims=dims)
    return ar


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
        logger=None,
    ):
        if logger:
            logger.info(f"Initializing {name} arrays")
        self._file_list = file_list
        self._group_name = name

        # Obtain global sparse array indices and data type
        # Currently these are determined by reading the first global sparse matrix,
        # but in the future, they could might be read directly from file if the arrays
        # are stored (or have been transformed to) CSR format.
        mat1 = self._master_matrix(file_list[0], name)
        indptr = mat1.indptr
        indices = mat1.indices
        dtype = mat1.dtype.newbyteorder("=")

        # Collect information for MPI scatter of sparse array data
        # For this, we need arrays with sparse data offsets and counts for every rank
        self._comm = comm
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
        self.indices = np.asarray(indices[indptr[istartrow] : indptr[istoprow]])
        self.indptr = np.asarray(indptr[istartrow : istoprow + 1])
        self.indptr -= self.indptr[0]
        self.dense_shape = (counts[rank], counts.sum())

        shape = (self.indices.size,)
        if len(file_list) > 1:
            shape = (len(file_list),) + shape
        super().__init__(shape, dtype, name)

    def __getitem__(self, slices) -> np.ndarray:
        itime = 0 if len(self._file_list) == 1 else slices[0]
        assert isinstance(itime, (int, np.integer))
        if rank == 0:
            d = self._master_matrix(self._file_list[itime]).data.newbyteorder("=")
        else:
            d = None
        values = np.empty((self.shape[-1],), self.dtype)
        self._comm.Scatterv([d, self._counts, self._offsets, MPI.DOUBLE], values)
        return values[slices[1:]] if len(self._file_list) > 1 else values[slices]

    def _master_matrix(self, fn: str, verbose=False) -> scipy.sparse.csr_array:
        with h5py.File(fn) as ds:
            group = ds[self._group_name]
            Aexp_data = group["data"]
            Aexp_ir = group["ir"]
            Aexp_jc = group["jc"]
            shape = (Aexp_jc.size - 1, Aexp_jc.size - 1)
            Aexp = scipy.sparse.csc_array((Aexp_data, Aexp_ir, Aexp_jc), shape)
        return Aexp.tocsr()

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
    ds = xr.open_dataset(
        fn,
        engine="h5netcdf",
        phony_dims="sort",
    )
    if rank == 0 and logger:
        logger.info(f"Reading grid information from: {fn}")

    bathy = np.asarray(ds["bathy"], dtype=int)
    ideep = np.asarray(ds["ideep"], dtype=int)
    x = np.asarray(ds["x"])
    y = np.asarray(ds["y"])
    z = np.asarray(ds["z"])
    dz = np.asarray(ds["dz"])

    return bathy, ideep, x, y, z, dz


def _profile_indices(
    ideep: np.ndarray,
    verbose=False,
) -> Tuple[int, np.ndarray, np.ndarray]:
    if rank == 0:
        ie = ideep.ravel().cumsum().reshape(ideep.shape)
        ie = np.ma.array(ie, mask=ideep == 0).compressed()
        nprof = np.asarray(len(ie), dtype=int)
        ib = np.empty(nprof, dtype=int)
        ib[0] = 1
        ib[1:] = ie[:-1] + 1
    else:
        ib = None
        ie = None
        nprof = np.empty(1, dtype=int)

    if rank == 0 and verbose:
        print("ib:  ", nprof, ib)
        print("ie:  ", nprof, ie)

    _comm.Bcast([nprof, MPI.INT], root=0)

    return int(nprof), ib, ie


def _mapping_arrays(
    nprof: int, ib: np.ndarray, ie: np.ndarray, verbose=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rank == 0:
        rows = np.empty(nprof + 1, dtype="i")
        rows[0] = 0
        rows[1:] = np.cumsum(ie - ib + 1)
        if verbose:
            print("rows:", len(rows), rows)

        indx = np.empty(size + 1, dtype=int)
        if use_root:
            k = int(np.ceil(nprof / size))
            indx[0] = 0
            indx[1:] = k * (np.arange(size) + 1)
        else:
            k = int(np.ceil(nprof / (size - 1)))
            indx[0:2] = 0
            indx[2:] = k * (np.arange(size - 1) + 1)
        # indx[0] = 0
        # indx[1:] = k * (np.arange(size) + 1)
        indx[size] = min(k * size, nprof)
        nlp = np.asarray(indx[1:] - indx[:-1], dtype=int)
        counts = np.asarray(
            [(rows[indx[i + 1]] - rows[indx[i]]) for i in range(size)], dtype=int
        )

        o = np.empty(size, dtype=int)
        o[0] = 0
        o[1:] = np.cumsum(counts[:-1])
        offsets = np.append(o, ie[-1])
    else:
        rows = None
        indx = np.empty(size + 1, dtype=int)
        nlp = np.empty(size, dtype=int)
        counts = np.empty(size, dtype=int)
        offsets = np.empty(size + 1, dtype=int)

    _comm.Bcast((nlp, MPI.INTEGER), root=0)
    _comm.Bcast((indx, MPI.INTEGER), root=0)
    _comm.Bcast((counts, MPI.INTEGER), root=0)
    _comm.Bcast((offsets, MPI.INTEGER), root=0)

    if rank == 0 and verbose:
        print("nlp:     ", np.sum(nlp), nlp)
        print("indx:    ", len(indx), indx)
        print("counts:  ", np.sum(counts), counts)
        print("offsets: ", len(offsets), offsets)

    return nlp, indx, counts, offsets


def create_domain(path: str, logger=None) -> pygetm.domain.Domain:
    bathy, ideep, lon, lat, z, dz = _read_grid(path, logger=logger)

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
    # (already in nx,1,nz order as TMM needs z to be fastest varying)
    domain.wet_loc = np.arange(1, nz + 1) <= ideep_loc.T[:, :, np.newaxis]

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


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        tmm_config: dict,
        fabm_config: str = "fabm.yaml",
    ):
        super().__init__(domain, fabm_config)
        self.tmm_logger = self.logger.getChild("TMM")
        self.tmm_logger.info(f"Initializing TMM component")
        _update_vertical_coordinates(self.domain.T, self.domain.dz)
        if self.domain.glob and self.domain.glob is not self.domain:
            _update_vertical_coordinates(self.domain.glob.T, self.domain.dz)

        self._tmm_matrix_config(tmm_config)
        Aexp_src = TransportMatrix(
            self._matrix_paths_Ae,
            "Aexp",
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
        )
        Aimp_src = TransportMatrix(
            self._matrix_paths_Ai,
            "Aimp",
            self.domain.offsets,
            self.domain.counts,
            self.domain.tiling.rank,
            self.domain.tiling.comm,
            logger=self.tmm_logger,
        )
        self.Aexp = Aexp_src.create_sparse_array()
        self.Aimp = Aimp_src.create_sparse_array()
        times = np.array(
            [cftime.datetime(2000, imonth + 1, 16) for imonth in range(12)]
        )
        if Aexp_src.ndim == 2:
            Aexp_tip = pygetm.input.TemporalInterpolation(
                Aexp_src, 0, times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aexp_tip.name, Aexp_tip, self.Aexp.data)
            )
        else:
            self.Aexp.data[:] = Aexp_src
        if Aimp_src.ndim == 2:
            Aimp_tip = pygetm.input.TemporalInterpolation(
                Aimp_src, 0, times, climatology=True
            )
            self.input_manager._all_fields.append(
                (Aimp_tip.name, Aimp_tip, self.Aimp.data)
            )
        else:
            self.Aimp.data[:] = Aimp_src

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

    def transport(self, timestep: float):
        packed_values = np.empty(self.domain.nwet_tot)
        for tracer in self.tracers:
            self.domain.tiling.comm.Allgatherv(
                [tracer.values.T[self.domain.wet_loc], MPI.DOUBLE],
                (packed_values, self.domain.counts, self.domain.offsets, MPI.DOUBLE),
            )
            # packed_values is the packed TMM-style array with tracer values

            # Copy updated tracer values back to authoratitive array
            tracer.values.T[self.domain.wet_loc] = packed_values[self.domain.tmm_slice]
