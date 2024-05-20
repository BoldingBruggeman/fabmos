from typing import Iterable, Optional, Tuple
import functools

import numpy as np
import numpy.typing as npt
import xarray as xr

import pygetm
from pygetm.domain import *
from pygetm.constants import CENTERS, INTERFACES

from mpi4py import MPI

from . import Array


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
        if self.values is None:
            raise Exception(f'self.values is None')
        if compressed_values is None:
            raise Exception(f'compressed_values is None {self._source.name}')
        #raise Exception(f'{self.values.shape}, {self._slice[0]!r}, {self._slice[1].shape!r} {self._source!r}')
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




def _update_coordinates(grid: pygetm.domain.Grid, area: np.ndarray, h: Optional[np.ndarray]=None):
    slc_loc, slc_glob, _, _ = grid.domain.tiling.subdomain2slices()
    if h is None:
        grid.ho.values[slc_loc] = grid.H.values[slc_loc]
        grid.hn.values[slc_loc] = grid.H.values[slc_loc]
    else:
        grid.ho.values[slc_loc] = h[slc_glob]
        grid.hn.values[slc_loc] = h[slc_glob]
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

    grid.area.values[slc_loc] = area[slc_glob]
    grid.D.values[slc_loc] = grid.H.values[slc_loc]


def compress(full_domain: Optional[Domain], comm: Optional[MPI.Comm] = None) -> Domain:
    full_domain.initialize(pygetm.BAROCLINIC)

    nx, mask, lon, lat, x, y, H, area = None, None, None, None, None, None, None, None
    if full_domain.glob is not None:
        mask = full_domain.glob.mask[1::2, 1::2] != 0
        lon = full_domain.glob.lon[1::2, 1::2][mask][np.newaxis, :]
        lat = full_domain.glob.lat[1::2, 1::2][mask][np.newaxis, :]
        if full_domain.glob.x is not None:
            x = full_domain.glob.x[1::2, 1::2][mask][np.newaxis, :]
            y = full_domain.glob.y[1::2, 1::2][mask][np.newaxis, :]
        H = full_domain.glob.H[1::2, 1::2][mask][np.newaxis, :]
        area = full_domain.glob.area[1::2, 1::2][mask][np.newaxis, :]
        nx = mask.sum()

    # Simple subdomain division along x dimension
    tiling = pygetm.parallel.Tiling(nrow=1, comm=comm)

    mask = tiling.comm.bcast(mask)
    lon = tiling.comm.bcast(lon)
    lat = tiling.comm.bcast(lat)
    x = tiling.comm.bcast(x)
    y = tiling.comm.bcast(y)
    H = tiling.comm.bcast(H)
    nx = tiling.comm.bcast(nx)
    area = tiling.comm.bcast(area)

    domain = pygetm.domain.create(
        nx,
        1,
        full_domain.nz,
        lon=lon,
        lat=lat,
        x=x,
        y=y,
        mask=1,
        H=H,
        spherical=full_domain.spherical,
        tiling=tiling,
        logger=full_domain.root_logger,
        halox=full_domain.halox,
        haloy=full_domain.haloy
    )

    slc_loc, slc_glob, _, _ = domain.tiling.subdomain2slices()
    global_indices = np.indices(mask.shape)
    local_indices = [i[mask][slc_glob[-1]] for i in global_indices]
    domain.input_grid_mappers.append(
        functools.partial(
            map_input_to_compressed_grid,
            indices=local_indices,
            global_shape=mask.shape,
        )
    )

    if domain.tiling.rank == 0:
        tf = functools.partial(
            CompressedToFullGrid, grid=full_domain.glob.T, mapping=mask
        )
        domain.default_output_transforms.append(tf)

    domain.uncompressed_area = area

    return domain
