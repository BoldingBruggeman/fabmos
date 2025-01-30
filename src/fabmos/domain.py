from typing import Iterable, Optional, Tuple, Union
import functools

import numpy as np
import numpy.typing as npt
import xarray as xr

import pygetm
from pygetm.domain import *
from pygetm import CENTERS, INTERFACES

from . import Array, Grid


class map_input_to_compressed_grid:
    def __init__(
        self,
        grid: Grid,
        global_shape: Tuple[int],
        global_indices: Iterable[np.ndarray],
    ):
        self.global_shape = global_shape

        slc_loc, _, _, _ = grid.tiling.subdomain2slices()

        local_indices = [grid.array(dtype=np.intp), grid.array(dtype=np.intp)]
        local_indices[0].scatter(global_indices[0])
        local_indices[1].scatter(global_indices[1])
        self.indices = [i.values[0, slc_loc[-1]] for i in local_indices]

    def __call__(
        self, value: Union[np.ndarray, xr.DataArray]
    ) -> Optional[xr.DataArray]:
        if value.shape[-2:] != self.global_shape:
            return

        indices = self.indices

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
        local_shape = value.shape[:-2] + tuple(
            [s.stop - s.start for s in region_slices]
        )
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


class map_input_to_cluster_grid:
    def __init__(self, grid: Grid, cluster_index: np.ndarray, bath: np.ndarray):
        slc_loc, slc_glob, _, _ = grid.tiling.subdomain2slices()

        nx_loc = slc_loc[-1].stop - slc_loc[-1].start
        self.cluster_mask = np.full((nx_loc,) + cluster_index.shape, False)
        iglobs = range(slc_glob[-1].start, slc_glob[-1].stop)
        for iglob, cm in zip(iglobs, self.cluster_mask):
            cm[...] = cluster_index == iglob
        self.bath = bath

    def __call__(
        self, value: Union[np.ndarray, xr.DataArray]
    ) -> Optional[xr.DataArray]:
        if value.shape[-2:] != self.cluster_mask.shape[1:]:
            return

        if isinstance(value, np.ndarray):
            raise NotImplementedError

        assert self.bath.shape == self.cluster_mask.shape[1:]
        include = True
        if value.getm.z is not None:
            z = np.asarray(value.getm.z)
            if (z < 0.0).any():
                z *= -1

            # Include all points in source (to-be-interpolated) grid that fall
            # within the target depth range, i.e., that lie at or above the bottom.
            include = z[:, np.newaxis, np.newaxis] <= self.bath[np.newaxis, :, :]

            # Also include each source layer just below the bottom depth
            if z[1] > z[0]:
                include[1:] = include[:-1] | include[1:]
                include[0] = True
            else:
                include[:-1] = include[:-1] | include[1:]
                include[-1] = True

        np_values = np.asarray(value).reshape(value.shape[:-2] + (-1,))
        result = np.full(value.shape[:-2] + (1, self.cluster_mask.shape[0]), np.nan)
        for icluster, mask in enumerate(self.cluster_mask):
            if include is not True:
                mask = mask & include
            mask = mask.reshape(mask.shape[:-2] + (-1,))
            mask = np.broadcast_to(mask, np_values.shape)
            sum = np_values.sum(axis=-1, where=mask)
            count = mask.sum(axis=-1)
            np.divide(sum, count, where=count > 0, out=result[..., 0, icluster])

        # Keep only coordinates without any horizontal coordinate dimension (x, y)
        alldims = frozenset(value.dims[-2:])
        coords = {}
        for n, c in value.coords.items():
            if not alldims.intersection(c.dims):
                coords[n] = c
        return xr.DataArray(result, dims=value.dims, coords=coords, attrs=value.attrs)


class CompressedToFullDomain(pygetm.output.operators.UnivariateTransformWithData):
    @staticmethod
    def apply(source: pygetm.output.operators.Base, **kwargs):
        if isinstance(source, pygetm.output.operators.WrappedArray):
            return source
        return CompressedToFullDomain(source, **kwargs)

    def __init__(
        self,
        source: pygetm.output.operators.Base,
        target_domain: Domain,
        mapping: np.ndarray,
        global_array: Optional[np.ndarray] = None,
    ):
        shape = source.shape[:-2] + (target_domain.ny, target_domain.nx)
        dims = source.dims[:-2] + ("y", "x")
        expression = f"{self.__class__.__name__}({source.expression})"
        super().__init__(source, shape=shape, dims=dims, expression=expression)
        self.values.fill(self.fill_value)
        self._slice = (Ellipsis, mapping)
        self._global_values = global_array
        self._target_domain = target_domain

    def get(
        self, out: Optional[npt.ArrayLike] = None, slice_spec: Tuple[int, ...] = ()
    ) -> npt.ArrayLike:
        compressed_values = self._source.get()
        if self._global_values is not None:
            if out is None:
                return self._global_values
            else:
                out[slice_spec] = self._global_values
                return out[slice_spec]
        if self.values is None:
            raise Exception("self.values is None")
        if compressed_values is None:
            raise Exception(f"compressed_values is None {self._source.name}")
        # raise Exception(f'{self.values.shape}, {self._slice[0]!r}, {self._slice[1].shape!r} {self._source!r}')
        self.values[self._slice] = compressed_values[..., 0, :]
        return super().get(out, slice_spec)

    @property
    def coords(self):
        if self._target_domain.coordinate_type == pygetm.CoordinateType.LONLAT:
            x, y = self._target_domain.lon, self._target_domain.lat
        else:
            x, y = self._target_domain.x, self._target_domain.y
        globals_arrays = (x[1::2, 1::2], y[1::2, 1::2], None)
        for c, g in zip(self._source.coords, globals_arrays):
            yield CompressedToFullDomain.apply(
                c,
                target_domain=self._target_domain,
                mapping=self._slice[-1],
                global_array=g,
            )


class ClustersToFullGrid(pygetm.output.operators.UnivariateTransformWithData):
    def __init__(
        self,
        source: pygetm.output.operators.Base,
        grid: Optional[Grid],
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
    def grid(self) -> Grid:
        return self._grid

    @property
    def coords(self):
        global_x = self._grid.lon if self._grid.domain.spherical else self._grid.x
        global_y = self._grid.lat if self._grid.domain.spherical else self._grid.y
        global_z = self._grid.zc if self._z == CENTERS else self._grid.zf
        globals_arrays = (global_x, global_y, global_z)
        for c, g in zip(self._source.coords, globals_arrays):
            yield ClustersToFullGrid(c, self._grid, self._clusters, g)


def freeze_vertical_coordinates(
    grid: Grid,
    depth: Array,
    vertical_coordinates: Optional[pygetm.vertical_coordinates.Base] = None,
    bottom_to_surface: bool = False,
):
    """Update layer thicknesses, vertical coordinates, and depth-below-surface,
    using either prescribed layer thicknesses or a vertical coordinate algorithm.
    Water depth ``grid.D`` must be up to date before entering this routine
    (by default, this will equal bathymetric depth ``grid.H``)"""
    if vertical_coordinates is not None:
        # layer thicknesses from vertical coordinate algorithm and water depth D
        vertical_coordinates.update(0.0)
    elif grid.nz == 1:
        grid.hn.all_values[0, grid._water] = grid.D.all_values[grid._water]

    # Verify that D and hn are in sync
    where = grid.mask3d.all_values != 0 if hasattr(grid, "mask3d") else np._NoValue
    h_sum = grid.hn.all_values.sum(axis=0, where=where)
    assert np.isclose(h_sum, grid.D.all_values, rtol=1e-14).all(where=grid._water)

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
    grid.zc._fill_value = 0.0
    grid.zf._fill_value = 0.0
    depth.fill(-grid.zc.all_values)
    grid.hn.all_values[grid.hn.all_values == 0.0] = grid.hn.fill_value
    grid.hn.attrs["_time_varying"] = False
    grid.zc.attrs["_time_varying"] = False
    grid.zf.attrs["_time_varying"] = False
    depth.attrs["_time_varying"] = False
    grid.D.attrs["_time_varying"] = False


def compress(
    full_domain: Optional[Domain], extra_fields: Optional[Iterable[str]] = ()
) -> Domain:
    """Compress domain by filtering out all land points (with mask=0).

    The resulting domain will have ny=1 and nx=number of wet points.
    """
    nwet = None
    if full_domain.comm.rank == 0:
        mask_hz = full_domain._mask[1::2, 1::2] == 1
        nwet = mask_hz.sum()
    nwet = full_domain.comm.bcast(nwet)

    n_ori = full_domain.nx * full_domain.ny
    if nwet == n_ori:
        # Original domain already contained water points only
        return full_domain

    full_domain.logger.info(
        f"Compressing {full_domain.nx} x {full_domain.ny} grid"
        f" to {nwet} wet points ({100 * nwet / n_ori:.1f} %)"
    )

    # Squeeze out land points from arguments that wil be provided to Domain
    kwargs = {}
    if full_domain.comm.rank == 0:
        kwargs["mask"] = 1
        for n in ("lon", "lat", "x", "y", "f", "H"):
            source = getattr(full_domain, "_" + n)
            if source is not None:
                kwargs[n] = source[1::2, 1::2][mask_hz]

    domain = Domain(
        nwet, 1, **kwargs, logger=full_domain.root_logger, comm=full_domain.comm
    )

    domain.global_indices = (None, None)
    if domain.comm.rank == 0:
        # Override calculated metrics with original values
        # (before land points were squeezed out)
        for n in ("_dx", "_dy", "_area", "_rotation"):
            source = getattr(full_domain, n)
            if source is not None:
                getattr(domain, n)[1, 1::2] = source[1::2, 1::2][mask_hz]
            else:
                setattr(domain, n, None)
        for n in extra_fields:
            source = getattr(full_domain, n)
            setattr(domain, n, source[Ellipsis, np.newaxis, mask_hz])

        domain.global_indices = [
            i[np.newaxis, mask_hz] for i in np.indices(mask_hz.shape)
        ]

        # By default, transform compressed fields to original 3D domain on output
        kwargs = dict(target_domain=full_domain, mapping=mask_hz)
        tf = functools.partial(CompressedToFullDomain.apply, **kwargs)
        domain.default_output_transforms.append(tf)
    else:
        for n in extra_fields:
            setattr(domain, n, None)

    domain.input_grid_mappers.append(
        functools.partial(
            map_input_to_compressed_grid,
            global_indices=domain.global_indices,
            global_shape=(full_domain.ny, full_domain.nx),
        )
    )

    return domain


class ClusterConnections:
    def __init__(self, domain: Domain, cluster_index: np.ndarray, n: int):
        assert cluster_index.shape == (domain.ny, domain.nx)
        uconnection = cluster_index[:, :-1] != cluster_index[:, 1:]
        vconnection = cluster_index[:-1, :] != cluster_index[1:, :]
        valid = cluster_index != -1
        uconnection &= valid[:, 1:] & valid[:, :-1]
        vconnection &= valid[1:, :] & valid[:-1, :]
        usources = cluster_index[:, :-1][uconnection]
        utargets = cluster_index[:, 1:][uconnection]
        vsources = cluster_index[:-1, :][vconnection]
        vtargets = cluster_index[1:, :][vconnection]
        assert (usources != utargets).all()
        assert (vsources != vtargets).all()
        self.n = n
        self.nx, self.ny = domain.nx, domain.ny
        self.uslice = (Ellipsis, usources, utargets)
        self.vslice = (Ellipsis, vsources, vtargets)
        self.uslice_rev = (Ellipsis, utargets, usources)
        self.vslice_rev = (Ellipsis, vtargets, vsources)
        self.uconnection = (Ellipsis,) + np.nonzero(uconnection)
        self.vconnection = (Ellipsis,) + np.nonzero(vconnection)
        self.lon_u = domain.lon[1::2, 2:-1:2][uconnection]
        self.lat_u = domain.lat[1::2, 2:-1:2][uconnection]
        self.lon_v = domain.lon[2:-1:2, 1::2][vconnection]
        self.lat_v = domain.lat[2:-1:2, 1::2][vconnection]
        self.length_u = domain.dy[1::2, 2:-1:2][uconnection]
        self.length_v = domain.dx[2:-1:2, 1::2][vconnection]

        if domain.rotation is not None:
            self.rotator_u = pygetm.core.Rotator(
                domain.rotation[1::2, 2:-1:2][uconnection]
            )
            self.rotator_v = pygetm.core.Rotator(
                domain.rotation[2:-1:2, 1::2][vconnection]
            )

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u)
        v = np.asarray(v)
        if u.shape[-2:] == (self.ny, self.nx - 1):
            # u and v on original grid (2D); extract relevant interfaces (1D)
            assert v.shape[-2:] == (self.ny - 1, self.nx)
            assert u.shape[:-2] == v.shape[:-2]
            u = u[self.uconnection]
            v = v[self.vconnection]
        out = np.zeros(u.shape[:-1] + (self.n, self.n), dtype=u.dtype)
        np.add.at(out, self.uslice, np.maximum(u, 0.0))
        np.add.at(out, self.uslice_rev, -np.minimum(u, 0.0))
        np.add.at(out, self.vslice, np.maximum(v, 0.0))
        np.add.at(out, self.vslice_rev, -np.minimum(v, 0.0))
        return out


def compress_clusters(
    full_domain: Optional[Domain],
    clusters: npt.ArrayLike,
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

    logger = full_domain.logger

    to_compress = ("mask", "lon", "lat", "x", "y", "H", "area", "rotation")
    ncluster = None
    cluster_index = None
    global_fields = {}
    compressed_fields = {}
    if full_domain.comm.rank == 0:
        # Collect fields from full domain that will be averaged over clusters
        # For longitude, use cos and sin to handle periodic boundary condition
        for name in to_compress:
            source = getattr(full_domain, "_" + name)
            if source is not None:
                global_fields[name] = source[1::2, 1::2]
        if "lon" in global_fields:
            lon_rad = np.pi * global_fields["lon"] / 180.0
            global_fields["coslon"] = np.cos(lon_rad)
            global_fields["sinlon"] = np.sin(lon_rad)
        global_fields["j"], global_fields["i"] = np.indices(
            (full_domain.ny, full_domain.nx), dtype=float
        )

        # Mask combines original domain mask and cluster mask (if any)
        unmasked = global_fields["mask"] != 0
        assert clusters.shape == unmasked.shape
        unmasked &= ~np.ma.getmaskarray(clusters)

        # Ensure bathymetry is valid in all unmasked points
        assert np.isfinite(global_fields["H"][unmasked]).all()

        # Cast clusters to regular array and collect unique cluster identifiers
        clusters = np.asarray(clusters)
        unique_clusters = np.unique(clusters[unmasked])
        ncluster = unique_clusters.size
        logger.info(f"Found {ncluster} unique clusters:")

        cluster_index = np.full(clusters.shape, -1, dtype=np.int16)
        for i, v in enumerate(unique_clusters):
            cluster_index[(clusters == v) & unmasked] = i

        # Prepare arrays for compressed (= per-cluster) domain fields
        for name, values in global_fields.items():
            compressed_fields[name] = np.empty((ncluster,), dtype=values.dtype)

        for i, c in enumerate(unique_clusters):
            selected = cluster_index == i

            # Average over cluster (or in the case of surface area: sum)
            cluster_values = {}
            for name, values in global_fields.items():
                cluster_values[name] = values[selected]
                compressed_fields[name][i] = cluster_values[name].mean()
            area = cluster_values["area"].sum()
            compressed_fields["area"][i] = area
            if "lon" in global_fields:
                compressed_fields["lon"] = (
                    np.arctan2(compressed_fields["sinlon"], compressed_fields["coslon"])
                    / np.pi
                    * 180
                )

            logger.info(f"  {c}:")
            logger.info(f"    cell count: {selected.sum()}")
            if "lon" in global_fields and "lat" in global_fields:
                logger.info(
                    "    mean coordinates:"
                    f" {compressed_fields['lon'][i]:.6f} °East,"
                    f" {compressed_fields['lat'][i]:.6f} °North"
                )
            logger.info(f"    mean depth: {compressed_fields['H'][i]:.1f} m")
            logger.info(f"    total area: {1e-6 * area:.1f} km2")

    domain = pygetm.domain.Domain(
        nx=full_domain.comm.bcast(ncluster),
        ny=1,
        x=compressed_fields.get("x"),
        y=compressed_fields.get("y"),
        lon=compressed_fields.get("lon"),
        lat=compressed_fields.get("lat"),
        mask=1,
        H=compressed_fields.get("H"),
        coordinate_type=pygetm.CoordinateType.IJ,
        logger=full_domain.root_logger,
        comm=full_domain.comm,
    )

    if domain.comm.rank == 0:
        domain.full_cluster_index = cluster_index
        domain._area[1, 1::2] = compressed_fields["area"]
        domain._dx[1, 1::2] = np.sqrt(domain._area[1, 1::2])
        domain._dy[1, 1::2] = domain._dx[1, 1::2]
        domain.icluster = compressed_fields["i"]
        domain.jcluster = compressed_fields["j"]
        for name in ("x", "y", "lon", "lat"):
            source = getattr(full_domain, name)
            if source is not None:
                setattr(domain, "_{name}_full", source[1::2, 1::2])

        if decompress_output:
            tf = functools.partial(
                ClustersToFullGrid,
                grid=full_domain.T,
                clusters=[cluster_index == i for i in range(len(unique_clusters))],
            )
            domain.default_output_transforms.append(tf)
        else:
            domain.extra_output_coordinates = []
            dims = ("y", "x")
            dims_ = (dims[0] + "_", dims[1] + "_")
            domain.extra_output_coordinates.append(
                pygetm.output.operators.WrappedArray(
                    cluster_index, "cluster_index", dims_, fill_value=-1
                )
            )
            for name in ("x", "y", "lon", "lat"):
                if name in global_fields:
                    info = Grid._array_args[name]
                    attrs = {
                        k: v for k, v in info["attrs"].items() if not k.startswith("_")
                    }
                    attrs.update(units=info["units"], long_name=info["long_name"])
                    domain.extra_output_coordinates.append(
                        pygetm.output.operators.WrappedArray(
                            global_fields[name], name + "_", dims_, attrs=attrs
                        )
                    )
            domain.extra_output_coordinates.append(
                pygetm.output.operators.WrappedArray(
                    compressed_fields["i"], "icluster", ("x",)
                )
            )
            domain.extra_output_coordinates.append(
                pygetm.output.operators.WrappedArray(
                    compressed_fields["j"], "jcluster", ("x",)
                )
            )
            domain.extra_output_coordinates.append(
                pygetm.output.operators.WrappedArray(unique_clusters, "cluster", ("x",))
            )
        domain.full_mask = np.where(unmasked, 1, 0)
        domain.cluster_ids = unique_clusters
        for name in ("x", "y", "lon", "lat"):
            if name in global_fields:
                setattr(domain, f"full_{name}", global_fields[name])

        domain.connections = ClusterConnections(full_domain, cluster_index, ncluster)
        interfaces = domain.connections(
            full_domain._dy[1::2, 2:-1:2], full_domain._dx[2:-1:2, 1::2]
        )
        interfaces += interfaces.T
        domain.interfaces = interfaces

    domain.input_grid_mappers.append(
        functools.partial(
            map_input_to_cluster_grid,
            cluster_index=domain.comm.bcast(cluster_index),
            bath=domain.comm.bcast(global_fields.get("H")),
        )
    )
    return domain
