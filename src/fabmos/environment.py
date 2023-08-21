import cftime

import pyairsea
from pyairsea import AlbedoMethod
import fabmos
from pygetm.constants import FILL_VALUE


class ShortWaveRadiation:
    def __init__(self, grid: fabmos.domain.Grid):
        self.logger = grid.domain.root_logger.getChild('ShortWaveRadiation')
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
            assert self.tcc.require_set(self.logger)
            self._ready = True

        hh = time.hour + time.minute / 60.0 + time.second / 3600.0
        yday = time.timetuple().tm_yday  # 1 for all of 1 January
        pyairsea.solar_zenith_angle(
            yday, hh, self.lon.all_values, self.lat.all_values, self.zen.all_values
        )
        pyairsea.shortwave_radiation(
            yday,
            self.zen.all_values,
            self.lat.all_values,
            self.tcc.all_values,
            self.swr.all_values,
        )
        pyairsea.albedo_water(
            self.albedo_method, self.zen.all_values, yday, self.albedo.all_values
        )
        self.swr.all_values *= 1.0 - self.albedo.all_values
