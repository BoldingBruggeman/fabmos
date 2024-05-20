import cftime
from typing import Union

import numpy as np
import xarray as xr

import awex
from awex import AlbedoMethod
import pygsw
import fabmos
from pygetm.constants import FILL_VALUE, CENTERS


class ShortWaveRadiation:
    def __init__(self, grid: fabmos.domain.Grid):
        self.logger = grid.domain.root_logger.getChild("ShortWaveRadiation")
        self.lon = grid.lon
        self.lat = grid.lat
        self.tcc = grid.array(
            name="tcc",
            long_name="total cloud cover",
            units="1",
            fill_value=FILL_VALUE,
            attrs=dict(standard_name="cloud_area_fraction"),
        )
        self.zen = grid.array(
            name="zen",
            long_name="solar zenith angle",
            units="degree",
            fill_value=FILL_VALUE,
            attrs=dict(
                standard_name="solar_zenith_angle",
            ),
        )
        self.swr = grid.array(
            name="swr",
            long_name="surface net downwelling shortwave radiation",
            units="W m-2",
            fill_value=FILL_VALUE,
            fabm_standard_name="surface_downwelling_shortwave_flux",
            attrs=dict(
                standard_name="net_downward_shortwave_flux_at_sea_water_surface",
            ),
        )
        self.albedo = grid.array(
            name="albedo",
            long_name="albedo",
            units="1",
            fill_value=FILL_VALUE,
            attrs=dict(
                standard_name="surface_albedo",
                _mask_output=True,
            ),
        )
        self.albedo_method = AlbedoMethod.PAYNE
        self._ready = False

    def update(self, time: cftime.datetime):
        if not self._ready:
#KB            assert self.tcc.require_set(self.logger)
            self._ready = True

        hh = time.hour + time.minute / 60.0 + time.second / 3600.0
        yday = time.timetuple().tm_yday  # 1 for all of 1 January
        awex.solar_zenith_angle(
            yday, hh, self.lon.all_values, self.lat.all_values, self.zen.all_values
        )
        awex.shortwave_radiation(
            yday,
            self.zen.all_values,
            self.lat.all_values,
            self.tcc.all_values,
            self.swr.all_values,
        )
        awex.albedo_water(
            self.albedo_method, self.zen.all_values, yday, self.albedo.all_values
        )
        self.swr.all_values *= 1.0 - self.albedo.all_values


def _broadcast_coordinate(
    c: xr.DataArray, t: xr.DataArray
) -> Union[xr.DataArray, np.ndarray]:
    if c.ndim == t.ndim:
        return c
    s = [np.newaxis] * t.ndim
    for i, d in enumerate(t.dims):
        if d in c.dims:
            s[i] = slice(None)
    return np.asarray(c)[tuple(s)]


def density(salt: xr.DataArray, temp: xr.DataArray) -> xr.DataArray:
    lon = _broadcast_coordinate(salt.getm.longitude, salt)
    lat = _broadcast_coordinate(salt.getm.latitude, salt)
    pres = _broadcast_coordinate(salt.getm.z, salt)
    rho = LazyDensity(salt.variable, temp.variable, lon, lat, pres)
    rho_xr = xr.DataArray(rho, coords=salt.coords, dims=salt.dims)
    return rho_xr


class LazyDensity(fabmos.input.Operator):
    def __init__(
        self,
        salt: Union[np.ndarray, xr.Variable],
        temp: Union[np.ndarray, xr.Variable],
        lon: Union[np.ndarray, xr.Variable],
        lat: Union[np.ndarray, xr.Variable],
        pres: Union[np.ndarray, xr.Variable],
    ):
        # Accept longitude, latitude, pressure in anticipation of a need to
        # convert between in-situ/potential/conservative temperature, and
        # practical/absolute salinity
        if np.all(pres < 0):
            pres = -pres
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        pres = np.asarray(pres, dtype=float)
        super().__init__(salt, temp, lon, lat, pres, passthrough=True)

    def apply(
        self,
        salt: np.ndarray,
        temp: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        pres: np.ndarray,
        dtype=None,
    ) -> np.ndarray:
        salt = np.asarray(salt, dtype=float)
        temp = np.asarray(temp, dtype=float)
        rho = np.empty_like(temp)
        salt, temp, pres = np.broadcast_arrays(salt, temp, pres)
        pygsw.rho(
            salt.ravel(),
            temp.ravel(),
            pres.ravel(),
            rho.ravel(),
        )
        return rho
