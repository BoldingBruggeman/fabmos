from typing import Optional, Iterable, Callable
import logging
import timeit

import netCDF4
import xarray as xr
import numpy as np

import fabmos


# fabmos.input.debug_nc_reads()


def average_uv(u10, v10, **name2values):
    mean_length = np.hypot(u10, v10).mean(axis=-1)
    mean_angle = np.arctan2(u10.sum(axis=-1), v10.sum(axis=-1))
    return dict(
        u10=np.sin(mean_angle) * mean_length, v10=np.cos(mean_angle) * mean_length
    )


def average(
    domain: fabmos.domain.Domain,
    infile: str,
    outfile: str,
    *,
    variables: Optional[Iterable[str]] = None,
    chunksize=24,
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

    if variables is None:
        variables = []
        with xr.open_dataset(infile) as ds:
            for name, da in ds.items():
                if da.getm.longitude is not None and da.getm.longitude is not None:
                    variables.append(name)
        logger.info(
            f"Detected {len(variables)} variables to average: {', '.join(variables)}"
        )

    # Create lazy DataArray for each source variable interpolated to the unmasked
    # points of the full domain (1D arrays)
    name2da = {}
    for name in variables:
        da = fabmos.input.from_nc(infile, name)
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
                logger.info(f"Detected {mask.sum()} masked points in {name}")
            da = fabmos.input.horizontal_interpolation(da, lon, lat, mask=mask)
        name2da[name] = da

    ncluster = domain.nx
    clusters = []
    for icluster in range(ncluster):
        clusters.append((slice(None),) + np.nonzero(cluster_index == icluster))

    with netCDF4.Dataset(infile) as nc, netCDF4.Dataset(outfile, "w") as ncout:
        nc.set_auto_mask(False)
        ntime = da.shape[0]
        timedim = da.getm.time.dims[0]

        # Dimensions
        ncout.createDimension("x", ncluster)
        ncout.createDimension("y", 1)
        ncout.createDimension(timedim, ntime)

        # Time coordinate
        nctime = nc.variables[timedim]
        nctime_out = ncout.createVariable(timedim, nctime.dtype, (timedim,))
        nctime_out.units = nctime.units
        nctime_out.calendar = nctime.calendar
        nctime_out[:] = nctime[:]

        # Output variables (creation & metadata)
        ncvars_out = {}
        for name in name2da:
            ncvar = nc.variables[name]
            ncvar_out = ncout.createVariable(name, ncvar.dtype, (timedim, "y", "x"))
            for k in ncvar.ncattrs():
                if k != "_FillValue":
                    setattr(ncvar_out, k, getattr(ncvar, k))
            ncvars_out[name] = ncvar_out

        # Process time chunks
        start = timeit.default_timer()
        remaining = "unknown time"
        for itime in range(0, ntime, chunksize):
            # Estimate time remaining
            time_passed = timeit.default_timer() - start
            if itime > 0:
                remaining = f"{time_passed / itime * (ntime - itime):.1f} s"
            logger.info(
                f"Processing time={itime}:{itime + chunksize} ({remaining} remaining)"
            )

            # Read all values for current time slice
            name2values = {
                name: da[itime : itime + chunksize].values
                for name, da in name2da.items()
            }

            # Calculate means per cluster
            for icluster, ind in enumerate(clusters):
                name2cvalues = {n: v[ind] for n, v in name2values.items()}
                name2mean = {n: v.mean(axis=-1) for n, v in name2cvalues.items()}
                if averager is not None:
                    name2mean.update(averager(**name2cvalues))
                for name, values in name2mean.items():
                    ncvars_out[name][itime : itime + chunksize, 0, icluster] = values

        logger.info(f"Total time taken: {timeit.default_timer() - start:.1f} s")


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
                ncvar[:] = netCDF4.date2num(c, ncvar.units, ncvar.calendar)
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
