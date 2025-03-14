import numpy as np
import numpy.typing as npt

import fabmos

from mpi4py import MPI


def map_to_grid(
    grid: fabmos.Grid,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    flow: npt.ArrayLike,
    ksurface: int = 0,
    dmax: float = 10.0,
) -> fabmos.Array:
    """Convert a list of river locations and flow rates to a 2D field for
    surface elevation increase (m s-1) due to water input from rivers.
    Rivers with large flow rates will be distributed over several cells
    nearest to the river mouth.

    Args:
        grid: the model grid to map the river input to
        lon: river longitudes (°)
        lat: river latitudes (°)
        flow: river flow rates (m3 s-1)
        ksurface: vertical index of the surface layer
        dmax: maximum radius (°) around river mouth for distributing flow
            from large rivers

    Returns:
        array with surface level increase (m s-1) due to river flow
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    flow = np.asarray(flow)

    unmasked = grid.mask.values == 1
    area = grid.area.values[unmasked]
    vol = area * grid.hn.values[ksurface, unmasked]

    # Cartesian distance from rivers in lon-lat space
    dlon = grid.lon.values[unmasked][np.newaxis, :] - lon[:, np.newaxis]
    dlon1 = grid.lon.values[unmasked][np.newaxis, :] - 360 - lon[:, np.newaxis]
    dlon2 = grid.lon.values[unmasked][np.newaxis, :] + 360 - lon[:, np.newaxis]
    dlon = np.min(np.abs([dlon, dlon1, dlon2]), axis=0)
    dlat = grid.lat.values[unmasked][np.newaxis, :] - lat[:, np.newaxis]
    d = np.sqrt(dlon**2 + dlat**2)

    # Minimum Cartesian distance
    dmin = grid.tiling.allreduce(d.min(axis=1), MPI.MIN)

    # Expand search radius based on the ratio of river flow to receiving cell volume
    nearest = (d == dmin[:, np.newaxis]) & (d < dmax)
    scaledist = np.maximum(
        1.0,
        2
        * np.where(
            nearest, 365 * 86400.0 * flow[:, np.newaxis] / vol[np.newaxis, :], 0.0
        ).max(axis=1),
    )
    scaledist = grid.tiling.allreduce(scaledist, MPI.MAX)

    # Select receiving cells and divide river flow by their combined area to
    # calculate level increase (m/s)
    active = d <= np.minimum(dmin * scaledist, dmax)[:, np.newaxis]
    area_per_cell_per_river = np.where(active, area[np.newaxis, :], 0.0)
    area_per_river = grid.tiling.allreduce(area_per_cell_per_river.sum(axis=1))
    active_rivers = area_per_river > 0
    h_increase_per_river = np.zeros_like(flow)
    np.divide(flow, area_per_river, where=active_rivers, out=h_increase_per_river)
    h_increase_per_cell_per_river = np.where(
        active, h_increase_per_river[:, np.newaxis], 0.0
    )
    h_increase_per_cell = h_increase_per_cell_per_river.sum(axis=0)
    # flow_tot = flow[active_rivers].sum()
    out = grid.array(
        long_name="water level increase due to rivers",
        units="m s-1",
        attrs=dict(_time_varying=False),
    )
    out.values[unmasked] = h_increase_per_cell
    return out
