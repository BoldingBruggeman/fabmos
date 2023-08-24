import sys
import functools
from typing import Optional, Tuple

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


class TransportMatrices(pygetm.input.LazyArray):
    def __init__(self, template: str, ntime: int):
        # here you open one of the transport matrix files (e.g., template % 0)
        # to determine the number of data points n and the matrix data type dtype
        n = 2000
        dtype = float  # should be based on dtype used in .mat file
        name = template  # some human readable name for the data (all time points)
        self._n = n
        super().__init__((ntime, n), dtype, name)

    def __getitem__(self, slices) -> np.ndarray:
        assert isinstance(slices[0], (int, np.integer))
        itime = slices[0]
        # TODO for KB: return matrix for time index itime
        # The matrix can be in memory already or read from file on the spot
        values = np.full(self.shape[1:], itime, self.dtype)
        return values[slices[1:]]


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
    print("shapes: bathy, lon, lat: ", bathy.shape, lon.shape, lat.shape)

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

    nprof, ib, ie = _profile_indices(ideep, verbose=verbose)
    nlp, indx, counts, offsets = _mapping_arrays(nprof, ib, ie, verbose=verbose)
    sys.exit()

    # Jorn
    # Here is what I've done - using counts, and offsets variables:
    # Similar code for other exchanges
    if False:
        if rank == 0:
            a = np.empty(ncol)
            for i in range(size):
                a[offsets[i] : offsets[i + 1]] = i + 1
        else:
            a = None

        # Send portion of global array - a - to all
        b = np.empty(offsets[rank + 1] - offsets[rank])
        sendbuf = [a, counts, offsets[0:-1], MPI.DOUBLE]
        _comm.Scatterv(sendbuf, b)

        b[:] = b[:] + 1

        # root receives local arrays and store in global
        recvbuf = [a, counts, offsets[0:-1], MPI.DOUBLE]
        _comm.Gatherv(b, recvbuf)

    lon = np.arange(0.0, 10.0, 0.5)
    lat = np.arange(10.0, 40.0, 0.5)
    return pygetm.domain.create_spherical(lon=lon, lat=lat, mask=1, H=H, nz=15)


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
    def __init__(self, domain: pygetm.domain.Domain, fabm_config: str = "fabm.yaml"):
        super().__init__(domain, fabm_config)
        self.tmm_logger = self.logger.getChild("tmm")
        _update_vertical_coordinates(self.domain.T, self.domain.dz)
        if self.domain.glob and self.domain.glob is not self.domain:
            _update_vertical_coordinates(self.domain.glob.T, self.domain.dz)

        matrices = TransportMatrices("file_%i", 12)
        times = np.array(
            [cftime.datetime(2000, imonth + 1, 16) for imonth in range(12)]
        )
        tm_ip = pygetm.input.TemporalInterpolation(matrices, 0, times, climatology=True)

        self._current_tm = np.empty_like(tm_ip._current)  # placeholder; this should be a view of the sparse matrix data
        self.input_manager._all_fields.append((tm_ip.name, tm_ip, self._current_tm))

    def transport(self, timestep: float):
        packed_values = np.empty(self.domain.nwet_tot)
        for tracer in self.tracers:
            print(tracer.name)
            self.domain.tiling.comm.Allgatherv(
                [tracer.values.T[self.domain.wet_loc], MPI.DOUBLE],
                (packed_values, self.domain.counts, self.domain.offsets, MPI.DOUBLE),
            )
            # packed_values is the packed TMM-style array with tracer values

            # Copy updated tracer values back to authoratitive array
            tracer.values.T[self.domain.wet_loc] = packed_values[self.domain.tmm_slice]
