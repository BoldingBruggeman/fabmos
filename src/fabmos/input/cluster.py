from typing import Optional, Iterable, Callable, Mapping
import logging
import timeit

import netCDF4
import xarray as xr
import numpy as np
import numpy.typing as npt

import fabmos


# fabmos.input.debug_nc_reads()


def average_uv(u10: npt.ArrayLike, v10: npt.ArrayLike):
    mean_length = np.hypot(u10, v10).mean(axis=-1)
    mean_angle = np.arctan2(u10.sum(axis=-1), v10.sum(axis=-1))
    return np.sin(mean_angle) * mean_length, np.cos(mean_angle) * mean_length


def average(
    domain: fabmos.domain.Domain,
    ds: xr.Dataset,
    outfile: str,
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
            if da.getm.longitude is not None and da.getm.longitude is not None:
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
    logger: Optional[logging.Logger] = None,
    averager: Optional[Callable] = None,
    on_grid: bool = False,
    periodic_lon: bool = False,
):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

    cluster_index = domain.full_cluster_index
    if not on_grid:
        unmasked = domain.full_mask == 1
        lon = domain.full_lon[unmasked]
        lat = domain.full_lat[unmasked]
        cluster_index = cluster_index[unmasked]

    # Create lazy DataArray for each source variable interpolated to the unmasked
    # points of the full domain (1D arrays)
    da_ips = []
    for da in variables:
        if not on_grid:
            da = fabmos.input.limit_region(
                da,
                lon.min(),
                lon.max(),
                lat.min(),
                lat.max(),
                periodic_lon=periodic_lon,
            )
            mask = ~np.isfinite(da[0, ...].values)
            if not mask.any():
                mask = None
            else:
                logger.info(f"Detected {mask.sum()} masked points in {da.name}")
            da = fabmos.input.horizontal_interpolation(da, lon, lat, mask=mask)
        da = da.chunk({da.dims[0]: chunksize})
        da_ips.append(da)

    ncluster = domain.nx

    def _average(ncluster, cluster_index, *values, averager: Optional[Callable] = None):
        if averager is None:
            averager = lambda *args: tuple(a.mean(axis=-1) for a in args)
        means = tuple(
            np.empty(v.shape[:-1] + (1, ncluster), dtype=v.dtype) for v in values
        )
        for icluster in range(ncluster):
            sel = cluster_index == icluster
            current_values = [np.asarray(v)[..., sel] for v in values]
            cluster_means = averager(*current_values)
            for m, cv in zip(means, cluster_means):
                m[..., 0, icluster] = cv
        return means if len(means) > 1 else means[0]

    logger.info(f"Averaging into {ncluster} clusters...")
    da_avs = xr.apply_ufunc(
        _average,
        ncluster,
        cluster_index,
        *da_ips,
        input_core_dims=[[], da.dims[-1:]] + [da.dims[-1:]] * (len(variables)),
        output_core_dims=[["y", "x"]] * len(variables),
        kwargs=dict(averager=averager),
        dask="parallelized",
        output_dtypes=[da.dtype] * len(variables),
        dask_gufunc_kwargs=dict(output_sizes={"y": 1, "x": ncluster}),
    )
    if len(variables) == 1:
        da_avs = (da_avs,)
    logger.info(f"Chunking and casting...")
    name2da_av = {}
    for da_ori, da_av in zip(variables, da_avs):
        da_av = da_av.chunk({"y": 1, "x": ncluster})
        da_av.attrs.update(da_ori.attrs)
        name2da_av[da_ori.name] = da_av.astype("float32")

    logger.info(f"Creating dataset...")
    return xr.Dataset(name2da_av)


def get_connectivity(
    domain,
    u: xr.DataArray,
    v: xr.DataArray,
    outfile: str,
    periodic_lon: bool = False,
    depth_integrate: bool = True,
    logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

    depth = u.getm.z.values

    depth_if = np.zeros_like(depth, shape=(depth.size + 1,))
    depth_if[-1] = depth[-1]
    depth_if[1:-1] = 0.5 * (depth[:-1] + depth[1:])
    dz = depth_if[1:] - depth_if[:-1]

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
        extra_dims = 1 if depth_integrate else 2
        for idim in range(extra_dims):
            dimname = u_U.dims[idim]
            nc.createDimension(dimname, u_U.shape[idim])
            c = u_U.coords[dimname]
            if dimname == "time":
                ncvar = nc.createVariable(dimname, c.encoding["dtype"], c.dims)
                ncvar.units = c.encoding["units"]
                ncvar.calendar = c.encoding["calendar"]
                ncvar[:], _, _ = xr.coding.times.encode_cf_datetime(
                    c, c.encoding["units"], c.encoding["calendar"], c.encoding["dtype"]
                )
            else:
                ncvar = nc.createVariable(dimname, c.dtype, c.dims)
                for k, v in u_U.attrs():
                    setattr(ncvar, k, v)
                ncvar[:] = c
        ncvar = nc.createVariable(
            "exchange", float, (u_U.dims[:extra_dims] + ("source", "target"))
        )
        ncvar.long_name = "horizontal exchange"
        ncvar.units = "m3 s-1" if depth_integrate else "m2 s-1"
        ntime = u_U.shape[0]
        for itime in range(ntime):
            logger.info(f"  Saving connectivity for time {itime} of {ntime}...")
            u, _ = domain.connections.rotator_u(u_U[itime, ...], v_U[itime, ...])
            _, v = domain.connections.rotator_v(u_V[itime, ...], v_V[itime, ...])
            u = u.fillna(0.0) * domain.connections.length_u
            v = v.fillna(0.0) * domain.connections.length_v
            flow_across = domain.connections(u, v)  # m2 s-1
            if depth_integrate:
                flow_across = (flow_across * dz[:, np.newaxis, np.newaxis]).sum(axis=0)
            i = np.arange(domain.nx)
            flow_across[..., i, i] = -flow_across.sum(axis=-2)
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
