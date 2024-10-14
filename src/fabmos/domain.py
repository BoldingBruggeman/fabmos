from typing import Iterable, Optional, Tuple, Union
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
    value: Union[np.ndarray, xr.DataArray],
    global_shape: Tuple[int],
    indices: Iterable[np.ndarray],
) -> Optional[xr.DataArray]:
    if value.shape[-2:] != global_shape:
        return

    # Determine i,j slices for extracting the rectangular region that spans all
    # grid points included in the local compressed subdomain.
    # Then subtract the left offset from the indices to make them correspond to
    # the extracted region.
    region_slices = tuple([slice(ind.min(), ind.max() + 1) for ind in indices])
    all_region_slices = (slice(None),) * (value.ndim - 2) + region_slices
    indices = tuple([ind - ind.min() for ind in indices])

    if isinstance(value, np.ndarray):
        # Values are a simple numpy array (no known dimensions for time, depth, etc.)
        # Just extract the grid points of the local subdomain (i.e., compress)
        # and preserve any preceding dimensions (e.g., on-grid depth)
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


def map_input_to_cluster_grid(
    value: Union[np.ndarray, xr.DataArray], cluster_mask: np.ndarray, bath: np.ndarray
) -> Optional[xr.DataArray]:
    if value.shape[-2:] != cluster_mask.shape[1:]:
        return

    if isinstance(value, np.ndarray):
        raise NotImplementedError

    assert bath.shape == cluster_mask.shape[1:]
    include = True
    if value.getm.z is not None:
        z = np.asarray(value.getm.z)
        if (z < 0.0).any():
            z *= -1
        include = z[:, np.newaxis, np.newaxis] >= bath[np.newaxis, :, :]

    np_values = np.asarray(value).reshape(value.shape[:-2] + (-1,))
    result = np.empty(value.shape[:-2] + (1, cluster_mask.shape[0]))
    for icluster, mask in enumerate(cluster_mask):
        if include is not True:
            mask = mask & include
        mask = mask.reshape(mask.shape[:-2] + (-1,))
        result[..., 0, icluster] = np_values.mean(axis=-1, where=mask)

    # Keep only coordinates without any horizontal coordinate dimension (x, y)
    alldims = frozenset(value.dims[-2:])
    coords = {}
    for n, c in value.coords.items():
        if not alldims.intersection(c.dims):
            coords[n] = c
    return xr.DataArray(result, dims=value.dims, coords=coords, attrs=value.attrs)


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
            raise Exception(f"self.values is None")
        if compressed_values is None:
            raise Exception(f"compressed_values is None {self._source.name}")
        # raise Exception(f'{self.values.shape}, {self._slice[0]!r}, {self._slice[1].shape!r} {self._source!r}')
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


class ClustersToFullGrid(pygetm.output.operators.UnivariateTransformWithData):
    def __init__(
        self,
        source: pygetm.output.operators.Base,
        grid: Optional[pygetm.domain.Grid],
        clusters: Iterable[np.ndarray],
        global_array: Optional[Array] = None,
    ):
        self._grid = grid
        self._clusters = clusters
        self._slices = [(c,) for c in clusters]
        shape = source.shape[:-2] + (grid.ny, grid.nx)
        self._z = None
        if source.ndim > 2:
            self._z = CENTERS if source.shape[0] == grid.nz else INTERFACES
            self._slices = [(slice(None),) + s for s in self._slices]
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
            raise Exception(f"self.values is None")
        if compressed_values is None:
            raise Exception(f"compressed_values is None {self._source.name}")
        # raise Exception(f'{self.values.shape}, {self._slice[0]!r}, {self._slice[1].shape!r} {self._source!r}')
        for i, s in enumerate(self._slices):
            self.values[s] = compressed_values[..., 0, i, np.newaxis]
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
            yield ClustersToFullGrid(c, self._grid, self._clusters, g)


def _update_coordinates(
    grid: pygetm.domain.Grid,
    area: np.ndarray,
    h: Optional[np.ndarray] = None,
    bottom_to_surface: bool = False,
):
    slc_loc, slc_glob, _, _ = grid.domain.tiling.subdomain2slices()
    grid.D.values[slc_loc] = grid.H.values[slc_loc]
    if h is None:
        grid.domain.vertical_coordinates.initialize(grid)
        grid.domain.vertical_coordinates.update(0.0)
        grid.ho.values[slc_loc] = grid.hn.values[slc_loc]
    else:
        grid.ho.values[slc_loc] = h[slc_glob]
        grid.hn.values[slc_loc] = h[slc_glob]
    grid.zf.all_values.fill(0.0)
    grid.zf.all_values[1:] = grid.hn.all_values.cumsum(axis=0)
    if bottom_to_surface:
        # First interface = -sum(hn), then increasing to 0
        grid.zf.all_values -= grid.zf.all_values[-1]
    else:
        # First interface = 0, then decreasing to -sum(hn)
        grid.zf.all_values *= -1.0
    grid.zc.all_values[...] = 0.5 * (grid.zf.all_values[:-1] + grid.zf.all_values[1:])
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
    grid.iarea.values[slc_loc] = 1.0 / grid.area.values[slc_loc]


def compress(full_domain: Optional[Domain], comm: Optional[MPI.Comm] = None) -> Domain:
    """Compress domain by filtering out all land points (with mask=0).

    The resulting domain will have ny=1 and nx=number of wet points.
    """
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
        halox=0,
        haloy=0,
        vertical_coordinates=full_domain.vertical_coordinates,
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

    domain.global_area = area

    return domain


def compress_clusters(
    full_domain: Optional[Domain],
    clusters: npt.ArrayLike,
    comm: Optional[MPI.Comm] = None,
    decompress_output: bool = False,
) -> Domain:
    """Compress domain by merging grid cells with the same cluster value.

    Grid cells that are masked in the full domain or that have a masked cluster
    value will be excluded.
    Coordinates per cluster (x, y, lon, lat) are taken from the grid cell in
    the cluster that is closed to the cluster mean. Bathymetric depths will
    equal the mean over the cluster. Surface areas will equal the sum over the
    cluster.
    The resulting domain will have ny=1 and nx=number of clusters.
    """
    clusters = np.ma.asarray(clusters)
    assert clusters.shape == (full_domain.ny, full_domain.nx)

    # For compressed domain (ny=1): simple subdomain division along x dimension
    tiling = pygetm.parallel.Tiling(nrow=1, comm=comm)

    to_compress = ("mask", "lon", "lat", "x", "y", "H", "area")
    global_fields = {}
    values = None
    for name in to_compress:
        if full_domain.glob is not None and hasattr(full_domain.glob, name):
            all_values = getattr(full_domain.glob, name)
            if all_values is not None:
                values = all_values[1::2, 1::2]
        global_values = tiling.comm.bcast(values)
        if global_values is not None:
            global_fields[name] = global_values
    if "lon" in global_fields:
        lon_rad = np.pi * global_fields["lon"] / 180.0
        global_fields["coslon"] = np.cos(lon_rad)
        global_fields["sinlon"] = np.sin(lon_rad)

    unmasked = global_fields["mask"] != 0
    assert clusters.shape == unmasked.shape
    unmasked &= ~np.ma.getmaskarray(clusters)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters[unmasked])

    compressed_fields = {}
    for name, values in global_fields.items():
        compressed_fields[name] = np.empty((unique_clusters.size,), dtype=values.dtype)

    logger = full_domain.logger
    logger.info(f"Found {unique_clusters.size} unique clusters:")
    for i, c in enumerate(unique_clusters):
        sel = clusters == c
        cluster_values = {}
        for name, values in global_fields.items():
            cluster_values[name] = values[sel]
            compressed_fields[name][i] = cluster_values[name].mean()
        area = cluster_values["area"].sum()
        compressed_fields["area"][i] = area

        compressed_fields["lon"] = (
            np.arctan2(compressed_fields["sinlon"], compressed_fields["coslon"])
            / np.pi
            * 180
        )
        mean_lon = compressed_fields["lon"][i]
        mean_lat = compressed_fields["lat"][i]
        dlon = np.abs(global_fields["lon"] - mean_lon)
        dlon[dlon > 180] = 360 - dlon[dlon > 180]
        dist = dlon**2 + (global_fields["lat"] - mean_lat) ** 2
        inear = np.ma.array(dist, mask=~sel).argmin()
        near_lon = global_fields["lon"].flat[inear]
        near_lat = global_fields["lat"].flat[inear]
        compressed_fields["lon"][i] = near_lon
        compressed_fields["lat"][i] = near_lat
        if "x" in compressed_fields:
            compressed_fields["x"][i] = global_fields["x"].flat[inear]
            compressed_fields["y"][i] = global_fields["y"].flat[inear]

        logger.info(f"{c}:")
        logger.info(f"  cell count: {sel.sum()}")
        logger.info(f"  mean coordinates: {mean_lon:.6f} °East, {mean_lat:.6f} °North")
        logger.info(f"  final coordinates: {near_lon:.6f} °East, {near_lat:.6f} °North")
        logger.info(f"  mean depth: {compressed_fields['H'][i]:.1f} m")
        logger.info(f"  total area: {1e-6 * area:.1f} km2")

    domain = pygetm.domain.create(
        unique_clusters.size,
        1,
        full_domain.nz,
        x=compressed_fields["x"],
        y=compressed_fields["y"],
        lon=compressed_fields["lon"],
        lat=compressed_fields["lat"],
        mask=1,
        H=compressed_fields["H"],
        spherical=full_domain.spherical,
        tiling=tiling,
        logger=full_domain.root_logger,
        halox=0,
        haloy=0,
    )

    domain.global_area = compressed_fields["area"][np.newaxis, :]

    if decompress_output:
        if domain.tiling.rank == 0:
            tf = functools.partial(
                ClustersToFullGrid,
                grid=full_domain.T,
                clusters=[clusters == c for c in unique_clusters],
            )
            domain.default_output_transforms.append(tf)
    else:
        dims = ("lat", "lon") if full_domain.spherical else ("y", "x")
        dims_ = (dims[0] + "_", dims[1] + "_")
        cluster_index = np.full(clusters.shape, -1, dtype=np.int16)
        for i, v in enumerate(unique_clusters):
            cluster_index[clusters == v] = i
        domain.T.extra_output_coordinates.append(
            pygetm.output.operators.WrappedArray(
                cluster_index, "cluster_index", dims_, fill_value=-1
            )
        )
        for name in dims:
            array = getattr(domain.T, name)
            attrs = {"units": array.units, "long_name": array.long_name}
            domain.T.extra_output_coordinates.append(
                pygetm.output.operators.WrappedArray(
                    global_fields[name], name + "_", dims_, attrs=attrs
                )
            )

    slc_loc, slc_glob, _, _ = domain.tiling.subdomain2slices()
    assert domain.nx == slc_glob[-1].stop - slc_glob[-1].start
    cluster_mask = np.empty((domain.nx,) + cluster_index.shape, dtype=bool)
    for i, cm in zip(range(slc_glob[-1].start, slc_glob[-1].stop), cluster_mask):
        cm[...] = cluster_index == i

    domain.input_grid_mappers.append(
        functools.partial(
            map_input_to_cluster_grid,
            cluster_mask=cluster_mask,
            bath=global_fields["H"],
        )
    )
    return domain


def split_clusters(clusters: npt.ArrayLike):
    from skimage.segmentation import flood_fill

    masked = np.ma.getmaskarray(clusters)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters[~masked])
    n = -1
    for c in unique_clusters:
        while True:
            indices = (clusters == c).nonzero()
            if indices[0].size == 0:
                break
            n += 1
            seed_point = (indices[0][0], indices[1][0])
            flood_fill(clusters, seed_point, -n, connectivity=1, in_place=True)
    return np.ma.array(-clusters, mask=masked)


def drop_grids(domain: Domain, *grids: Grid):
    for name in list(domain.fields):
        if domain.fields[name].grid in grids:
            del domain.fields[name]
