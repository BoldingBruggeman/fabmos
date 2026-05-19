import os
from typing import Optional, Iterable, Callable, Mapping, Union
import logging

import netCDF4
import xarray as xr
import numpy as np
import numpy.typing as npt

import fabmos

# fabmos.input.debug_nc_reads()


def average_uv(
    weights: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    where = np.isfinite(u) & np.isfinite(v)
    w_all = np.broadcast_to(weights, where.shape)
    wsum = w_all.sum(axis=-1, where=where, keepdims=True)
    relw = w_all / np.where(wsum == 0, 1, wsum)
    mean_length = (np.hypot(u, v) * relw).sum(axis=-1, where=where)
    u_sum = (u * relw).sum(axis=-1, where=where)
    v_sum = (v * relw).sum(axis=-1, where=where)
    mean_angle = np.arctan2(u_sum, v_sum)
    return np.sin(mean_angle) * mean_length, np.cos(mean_angle) * mean_length


def default_averager(weights: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray, ...]:
    means = []
    for a in args:
        where = np.isfinite(a)
        w_all = np.broadcast_to(weights, where.shape)
        wsum = w_all.sum(axis=-1, where=where, keepdims=True)
        relw = w_all / np.where(wsum == 0, 1, wsum)
        means.append((a * relw).sum(axis=-1, where=where))
    return tuple(means)


def average(
    domain: fabmos.domain.Domain,
    ds: xr.Dataset,
    outfile: Union[str, os.PathLike],
    *,
    variables: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
    averagers: Mapping[Iterable[str], Callable] = {},
    **kwargs,
) -> xr.Dataset:
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

    name2da: dict[str, xr.DataArray] = {}
    if variables is None:
        for name, da in ds.items():
            if da.getm.longitude is not None and da.getm.latitude is not None:
                name2da[name] = da
        logger.info(
            f"Detected {len(name2da)} variables to average: {', '.join(name2da)}"
        )
    else:
        for name in variables:
            name2da[name] = ds[name]

    kwargs.update(logger=logger)
    datasets: list[xr.Dataset] = []
    for names, averager in averagers.items():
        logger.info(f"Averaging {', '.join(names)} with {averager}")
        das = [name2da.pop(name) for name in names]
        ds_av = average_variables(domain, das, averager=averager, **kwargs)
        datasets.append(ds_av)
    for name, da in name2da.items():
        logger.info(f"Averaging {name}")
        ds_av = average_variables(domain, [da], **kwargs)
        datasets.append(ds_av)

    return xr.merge(datasets, compat="no_conflicts")


def average_variables(
    domain: fabmos.domain.Domain,
    variables: Iterable[xr.DataArray],
    *,
    chunksize: int = 24,
    chunkdim: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    averager: Optional[Callable] = None,
    on_grid: bool = False,
    periodic_lon: bool = False,
):
    assert variables, "No variables to average"
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

    cluster_index = domain.full_cluster_index
    area = domain.full_area
    if not on_grid:
        unmasked = domain.full_mask == 1
        lon = domain.full_lon[unmasked]
        lat = domain.full_lat[unmasked]
        cluster_index = cluster_index[unmasked]
        area = area[unmasked]

    # Create lazy DataArray for each source variable interpolated to the unmasked
    # points of the full domain (1D arrays)
    da_ips = []
    variables = list(variables)
    for da in variables:
        cd = da.getm.time.dims[0] if chunkdim is None else chunkdim
        if not on_grid:
            da = fabmos.input.limit_region(
                da,
                lon.min(),
                lon.max(),
                lat.min(),
                lat.max(),
                periodic_lon=periodic_lon,
            )
            mask = np.asarray(~np.isfinite(da.isel({cd: 0})))
            if not mask.any() or da.ndim > 3:
                mask = None
            else:
                logger.info(f"Detected {mask.sum()} masked points in {da.name}")
            da = fabmos.input.horizontal_interpolation(da, lon, lat, mask=mask)
        da = da.chunk({cd: chunksize})
        da_ips.append(da)

    ncluster = domain.nx

    def _average(
        ncluster, cluster_index, weights, *values, averager: Optional[Callable] = None
    ):
        if averager is None:
            averager = default_averager
        n = cluster_index.ndim
        means = tuple(
            np.empty(v.shape[:-n] + (1, ncluster), dtype=v.dtype) for v in values
        )
        for icluster in range(ncluster):
            sel = cluster_index == icluster
            current_values = [np.asarray(v)[..., sel] for v in values]
            current_weights = weights[..., sel]
            cluster_means = averager(current_weights, *current_values)
            for m, cv in zip(means, cluster_means):
                m[..., 0, icluster] = cv
        return means if len(means) > 1 else means[0]

    logger.info(f"Averaging into {ncluster} clusters...")
    n = cluster_index.ndim
    da_avs = xr.apply_ufunc(
        _average,
        ncluster,
        cluster_index,
        area,
        *da_ips,
        input_core_dims=[[], da.dims[-n:], da.dims[-n:]]
        + [da.dims[-n:]] * (len(variables)),
        output_core_dims=[["y", "x"]] * len(variables),
        kwargs=dict(averager=averager),
        dask="parallelized",
        output_dtypes=[da.dtype] * len(variables),
        dask_gufunc_kwargs=dict(output_sizes={"y": 1, "x": ncluster}),
    )
    if len(variables) == 1:
        da_avs = (da_avs,)
    logger.info("Chunking and casting...")
    name2da_av = {}
    for da_ori, da_av in zip(variables, da_avs):
        da_av = da_av.chunk({"y": 1, "x": ncluster})
        da_av.attrs.update(da_ori.attrs)
        name2da_av[da_ori.name] = da_av.astype("float32")

    logger.info("Creating dataset...")
    return xr.Dataset(name2da_av)


def get_connectivity(
    domain: fabmos.domain.ClusteredDomain,
    u: xr.DataArray,
    v: xr.DataArray,
    outfile: Union[str, os.PathLike],
    periodic_lon: bool = False,
    depth_integrate: bool = True,
    logger: Optional[logging.Logger] = None,
    on_grid: bool = False,
    dz: Optional[np.ndarray] = None,
):
    """Compute time-varying connectivity matrix from u and v velocity fields
    and save this to a NetCDF file. Interpolate and rotate velocities
    to the native grid first if needed (i.e., if on_grid=False).

    Args:
        domain: target domain that defines the grid and clusters for which to
            compute connectivity
        u: eastward velocity (if on_grid=False) or x-velocity on the native
            (pre-clustering) U grid (if on_grid=True)
        v: northward velocity (if on_grid=False) or y-velocity on the native
            (pre-clustering) V grid (if on_grid=True)
        outfile: Path to output NetCDF file
        periodic_lon: Whether to treat longitude as periodic when interpolating
            (only used if on_grid=False)
        depth_integrate: Whether to integrate connectivity vertically (True)
            or keep it as a function of depth (False)
        logger: Optional logger for progress messages
        on_grid: Whether u and v are already on the native grid (True)
            or need to be interpolated and rotated (False)
        dz: Optional array of layer thicknesses (only needed if depth_integrate=True)
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

    depth = u.getm.z.values
    assert depth.ndim == 1, "Expected 1D depth coordinate"
    assert (depth[1:] > depth[:-1]).all(), "Depth must be monotonically increasing"

    if dz is None:
        depth_if = np.zeros_like(depth, shape=(depth.size + 1,))
        depth_if[-1] = depth[-1]
        depth_if[1:-1] = 0.5 * (depth[:-1] + depth[1:])
        dz = depth_if[1:] - depth_if[:-1]
    assert dz.shape == depth.shape, f"Expected dz shape {depth.shape}, got {dz.shape}"
    assert (dz >= 0).all(), "Layer thicknesses must be non-negative"
    depthdim = u.getm.z.dims[0]

    def get_clipped_dz(H):
        depth_if = np.zeros(shape=(dz.size + 1,) + H.shape)
        depth_if[1:] = dz.cumsum()[:, np.newaxis]
        np.minimum(depth_if, H, out=depth_if)
        return np.diff(depth_if, axis=0)

    face_U = domain.connections.length_u * get_clipped_dz(domain.connections.H_u)
    face_V = domain.connections.length_v * get_clipped_dz(domain.connections.H_v)

    if on_grid:
        # Verify that u and v have the correct shape for the native grid
        ny, nx = domain.full_mask.shape
        assert u.shape[-2:] == (
            ny,
            nx,
        ), f"Expected u shape (..., {ny}, {nx}), got {u.shape}"
        assert v.shape[-2:] == (
            ny,
            nx,
        ), f"Expected v shape (..., {ny}, {nx}), got {v.shape}"
    else:
        # Interpolate velocities to the interfaces between clusters
        def interpolate(values: xr.DataArray, lon: np.ndarray, lat: np.ndarray):
            values = fabmos.input.limit_region(
                values,
                lon.min(),
                lon.max(),
                lat.min(),
                lat.max(),
                periodic_lon=periodic_lon,
            )
            return fabmos.input.horizontal_interpolation(values, lon, lat)

        u_U = interpolate(u, domain.connections.lon_u, domain.connections.lat_u)
        u_V = interpolate(u, domain.connections.lon_v, domain.connections.lat_v)
        v_U = interpolate(v, domain.connections.lon_u, domain.connections.lat_u)
        v_V = interpolate(v, domain.connections.lon_v, domain.connections.lat_v)

    logger.info(f"Creating {outfile}...")
    with netCDF4.Dataset(outfile, "w") as nc:
        nc.createDimension("source", domain.nx)
        nc.createDimension("target", domain.nx)
        extra_dims = []
        for idim, dimname in enumerate(u.dims[:-2]):
            if depth_integrate and dimname == depthdim:
                continue
            extra_dims.append(dimname)
            nc.createDimension(dimname, u.shape[idim])
            c = u.coords[dimname]
            if dimname == "time":
                ncvar = nc.createVariable(dimname, c.encoding["dtype"], c.dims)
                ncvar.units = c.encoding["units"]
                ncvar.calendar = c.encoding["calendar"]
                ncvar[:], _, _ = xr.coding.times.encode_cf_datetime(
                    c, c.encoding["units"], c.encoding["calendar"], c.encoding["dtype"]
                )
            else:
                ncvar = nc.createVariable(dimname, c.dtype, c.dims)
                for key, val in c.attrs.items():
                    setattr(ncvar, key, val)
                ncvar[:] = c
        ncvar = nc.createVariable(
            "exchange", float, tuple(extra_dims) + ("source", "target")
        )
        ncvar.long_name = "horizontal exchange"
        ncvar.units = "m3 s-1" if depth_integrate else "m2 s-1"

        ncinflow = nc.createVariable(
            "net_inflow",
            float,
            [d for d in extra_dims if d != depthdim] + ["target"],
        )
        ncinflow.long_name = (
            "net inflow into cluster = sum of exchange over source clusters"
        )
        ncinflow.units = "m3 s-1"
        if not depth_integrate:
            nch = nc.createVariable("h", depth.dtype, (depthdim,))
            nch.units = "m"
            nch.long_name = "layer thickness"
            nch[:] = dz

            ncdiv = nc.createVariable(
                "divergence", float, u.dims[:-2] + ("target",), fill_value=-2e20
            )
            ncdiv.long_name = "local divergence = relative loss of volume"
            ncdiv.units = "s-1"

            ncw = nc.createVariable(
                "w", float, u.dims[:-2] + ("target",), fill_value=-2e20
            )
            ncw.long_name = (
                "inferred net vertical velocity at top of layer (>0 for downward)"
            )
            ncw.units = "m s-1"
        ntime = u.shape[0]
        for itime in range(ntime):
            logger.info(f"  Saving connectivity for time {itime} of {ntime}...")
            if on_grid:
                # Velocities on the original (pre-clustering) U and V grid.
                # Extract values at interfaces connecting clusters
                u_cur = u[itime].values[domain.connections.uconnection]
                v_cur = v[itime].values[domain.connections.vconnection]
            else:
                # Velocities were interpolated to grid. Assume they were
                # originally eastward and northward and rotate them back
                # to grid (i.e., along- and across-connection)
                u_cur, _ = domain.connections.rotator_u(u_U[itime], v_U[itime])
                _, v_cur = domain.connections.rotator_v(u_V[itime], v_V[itime])

            # Per-layer volume flow between clusters = velocity integrated over xy-z interfaces (m3 s-1)
            u_cur = np.nan_to_num(u_cur) * face_U
            v_cur = np.nan_to_num(v_cur) * face_V
            flow_across = domain.connections(u_cur, v_cur)

            if depth_integrate:
                flow_across = flow_across.sum(axis=0)
            inflow = flow_across.sum(axis=-2)
            outflow = flow_across.sum(axis=-1)
            ncinflow[itime, ...] = (inflow - outflow).sum(axis=0)
            if not depth_integrate:
                net_outflow = outflow - inflow  # recall: volume flow (m3 s-1)
                net_outflow_from_bottom = net_outflow[::-1].cumsum(axis=0)  # m3 s-1
                Vdiv = domain.hypsograph.volume_from_elevation(
                    -depth + 0.5 * dz
                ) - domain.hypsograph.volume_from_elevation(-depth - 0.5 * dz)
                ncdiv[itime, ...] = np.ma.divide(net_outflow, Vdiv)
                ncw[itime, ...] = np.ma.divide(
                    net_outflow_from_bottom[::-1],
                    domain.hypsograph.area_from_elevation(-depth + 0.5 * dz),
                )
                flow_across /= dz[:, np.newaxis, np.newaxis]
            ncvar[itime, ...] = flow_across


def split(clusters: npt.ArrayLike) -> np.ma.MaskedArray:
    from skimage.segmentation import flood_fill

    unmasked = ~np.ma.getmaskarray(clusters)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters[unmasked])
    fill_value = len(unique_clusters)
    cluster_indices = np.full(clusters.shape, fill_value, dtype=int)
    for icluster, cluster_id in enumerate(unique_clusters):
        cluster_indices[(clusters == cluster_id) & unmasked] = icluster
    n = 0
    for icluster in range(len(unique_clusters)):
        while True:
            indices = (cluster_indices == icluster).nonzero()
            if indices[0].size == 0:
                break
            n += 1
            seed_point = (indices[0][0], indices[1][0])
            flood_fill(cluster_indices, seed_point, -n, connectivity=1, in_place=True)
    return np.ma.masked_equal(-cluster_indices, -fill_value, copy=False)


def get_representative_points(clusters: npt.ArrayLike, mincount: int = 1):
    clusters = np.asanyarray(clusters)
    split_clusters = split(clusters).filled(0)
    jj, ii = np.indices(split_clusters.shape)
    for split_id in range(1, split_clusters.max() + 1):
        sel = split_clusters == split_id
        if sel.sum() < mincount:
            continue
        ii_sel = ii[sel]
        jj_sel = jj[sel]
        cluster_id = clusters[jj_sel[0], ii_sel[0]]
        dist = (ii_sel - ii_sel[:, np.newaxis]) ** 2
        dist += (jj_sel - jj_sel[:, np.newaxis]) ** 2
        dist.sort(axis=1)
        best_dist = dist.min(axis=0)
        has_best_dist = dist == best_dist
        still_best_dist = np.logical_and.accumulate(has_best_dist, axis=1)
        longest_best = still_best_dist.sum(axis=1)
        ind = np.argmax(longest_best)
        yield cluster_id, ii_sel[ind], jj_sel[ind]
