from typing import Optional, Union, Tuple
import os.path
import logging

import numpy as np
import xarray as xr

import pygetm.domain
import pygetm.airsea
from pygetm import _pygetm

from pygetm.constants import INTERFACES, FILL_VALUE, GRAVITY, RHO0, CENTERS
from .. import simulator, Array
from fabmos.domain import freeze_vertical_coordinates, compress
import fabmos


class TwoBand(pygetm.radiation.TwoBand):

    def __call__(self, swr: Array, kc2_add: Optional[Array] = None):
        """Compute heating due to shortwave radiation throughout the water column

        Args:
            swr: net downwelling shortwave radiation just below the water surface
                (i.e., what is left after reflection).
            kc2_add: additional depth-varying attenuation (m-1) of second waveband,
                typically associated with chlorophyll or suspended matter
        """
        if self._first:
            assert (
                self.A.require_set(self.logger)
                * self.kc1.require_set(self.logger)
                * self.kc2.require_set(self.logger)
            )
            self._first = False

        assert swr.grid is self.grid and not swr.z
        assert kc2_add is None or (kc2_add.grid is self.grid and kc2_add.z == CENTERS)

        # Non-visible band - downward
        self._kc.all_values = self.kc1.all_values
        self._swr.all_values = self.A.all_values * swr.all_values
        _pygetm.exponential_profile_1band_interfaces(
            self.grid.mask, self.grid.hn, self._kc, self._swr, up=False, out=self.rad
        )

        # Visible band - downward
        self._kc.all_values = self.kc2.all_values
        if kc2_add is not None:
            self._kc.all_values += kc2_add.all_values
        self._swr.all_values = swr.all_values - self._swr.all_values
        _pygetm.exponential_profile_1band_interfaces(
            self.grid.mask, self.grid.hn, self._kc, self._swr, up=False, out=self._rad
        )

        # Total downward
        self.rad.all_values += self._rad.all_values
        self.swr_abs.all_values = self.rad.all_values[1:] - self.rad.all_values[:-1]

        if self.par0.saved:
            # Visible part of shortwave radiation just below sea surface
            # (i.e., reflection/albedo already accounted for)
            self.par0.all_values = self._swr.all_values
        if self.par.saved:
            # Visible part of shortwave radiation at layer centers,
            # often used by biogeochemistry
            _pygetm.exponential_profile_1band_centers(
                self.grid.mask, self.grid.hn, self._kc, top=self._swr, out=self.par
            )


class Momentum:
    def __init__(self, cnpar: float = 1.0, avmmol: float = 1.3e-6):
        self.cnpar = cnpar
        self.avmmol = avmmol

    def initialize(self, grid: pygetm.core.Grid, logger: logging.Logger):
        self.grid = grid
        self.logger = logger
        self.ustar_b = grid.array(
            fill=0.0,
            name="ustar_b",
            units="m s-1",
            long_name="bottom shear velocity",
            fill_value=FILL_VALUE,
        )
        self.rr = grid.array(
            fill=0.0,
            name="rr",
            units="bottom drag * bottom velocity",
        )
        self.taub = grid.array(
            fill=0.0,
            name="taub",
            units="Pa",
            long_name="bottom shear stress",
            fabm_standard_name="bottom_stress",
        )
        self.u = grid.array(
            z=CENTERS,
            fill=0.0,
            name="u",
            units="m s-1",
            long_name="velocity in x-direction",
            attrs=dict(standard_name="sea_water_x_velocity"),
        )
        self.v = grid.array(
            z=CENTERS,
            fill=0.0,
            name="v",
            units="m s-1",
            long_name="velocity in y-direction",
            attrs=dict(standard_name="sea_water_y_velocity"),
        )
        self.w = grid.array(
            z=INTERFACES,
            fill=0.0,
            name="ww",
            units="m s-1",
            long_name="velocity in z-direction",
            attrs=dict(standard_name="sea_water_z_velocity"),
        )
        self.SS = grid.array(
            z=INTERFACES,
            fill=0.0,
            name="SS",
            units="s-2",
            long_name="shear frequency squared",
        )

        self._vertical_diffusion = pygetm.operators.VerticalDiffusion(
            grid, cnpar=self.cnpar
        )
        self.ea2 = grid.array(fill=0.0, z=CENTERS)
        self.ea4 = grid.array(fill=0.0, z=CENTERS)
        self.u_bot = self.u.isel(z=0)
        self.v_bot = self.v.isel(z=0)
        self.h_bot = self.grid.hn.isel(z=0)
        self.ufirst = True
        self.uo = grid.array(z=CENTERS, fill=0.0)
        self.vo = grid.array(z=CENTERS, fill=0.0)

    def advance(
        self,
        timestep: float,
        tausx: Array,
        tausy: Array,
        dpdx: Array,
        dpdy: Array,
        viscosity: Array,
    ):
        RHO0I = 1.0 / RHO0
        self.uo.values[...] = self.u.values
        self.vo.values[...] = self.v.values

        def advance_component(vel: Array, taus: Array, dp: Array, a_cor: np.ndarray):
            np.subtract(a_cor, GRAVITY * dp.values, out=self.ea4.values)
            self.ea4.values *= self.grid.hn.values
            self.ea4.values[-1] += RHO0I * taus.values
            self.ea4.values *= timestep
            self._vertical_diffusion(
                viscosity,
                timestep,
                vel,
                molecular=self.avmmol,
                ea2=self.ea2,
                ea4=self.ea4,
                use_ho=True,
            )

        self.ea2.values[0] = -timestep * self.rr.values
        if self.ufirst:
            advance_component(self.u, tausx, dpdx, self.grid.cor.values * self.v.values)
            advance_component(
                self.v, tausy, dpdy, -self.grid.cor.values * self.u.values
            )
        else:
            advance_component(
                self.v, tausy, dpdy, -self.grid.cor.values * self.u.values
            )
            advance_component(self.u, tausx, dpdx, self.grid.cor.values * self.v.values)
        self.ufirst = not self.ufirst

        # Calculate shear frequency
        # du = self.u.values[1:] - self.u.values[:-1]
        # dv = self.v.values[1:] - self.v.values[:-1]
        # self.SS.values[1:-1] = du**2 + dv**2
        SSU = 0.5 * (
            (self.u.values[1:] - self.u.values[:-1])
            * (self.u.values[1:] - self.uo.values[:-1])
            + (self.u.values[1:] - self.u.values[:-1])
            * (self.uo.values[1:] - self.u.values[:-1])
        )
        SSV = 0.5 * (
            (self.v.values[1:] - self.v.values[:-1])
            * (self.v.values[1:] - self.vo.values[:-1])
            + (self.v.values[1:] - self.v.values[:-1])
            * (self.vo.values[1:] - self.v.values[:-1])
        )
        self.SS.values[1:-1] = SSU + SSV
        self.SS.values[1:-1] /= (0.5 * (self.grid.hn[:-1] + self.grid.hn[1:])) ** 2

        # Calculate bottom friction
        _pygetm.bottom_friction(
            self.u_bot, self.v_bot, self.h_bot, self.avmmol, self.rr, update_z0b=True
        )

        self.ustar_b.values[...] = np.sqrt(
            np.sqrt(self.u.values[0] ** 2 + self.v.values[0] ** 2) * self.rr.values
        )
        self.taub.values[...] = self.ustar_b.values**2 * RHO0


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        vertical_coordinates: fabmos.vertical_coordinates.Base,
        *,
        fabm: Union[str, pygetm.fabm.FABM] = "fabm.yaml",
        gotm: Union[str, None] = None,
        log_level: Optional[int] = None,
        airsea: Optional[pygetm.airsea.Fluxes] = None,
        ice: Optional[pygetm.ice.Base] = None,
        charnock: bool = False,
        charnock_val: float = 1400.0,
        z0s_min: float = 0.02,
        wind_relative_to_ssuv: bool = True,
        connectivity: Optional[xr.DataArray] = None,
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_gotm")

        super().__init__(
            compress(domain),
            nz=vertical_coordinates.nz,
            fabm=fabm,
            log_level=log_level,
            use_virtual_flux=False,
            fabm_libname=fabm_libname,
            add_swr=False,
            process_vertical_movement=False,
        )

        vertical_coordinates.initialize(
            self.T, logger=self.logger.getChild("vertical_coordinates")
        )

        freeze_vertical_coordinates(
            self.T,
            self.depth,
            vertical_coordinates=vertical_coordinates,
            bottom_to_surface=True,
        )
        self.T.ho = self.T.hn

        self.airsea = airsea or pygetm.airsea.FluxesFromMeteo()
        self.airsea.initialize(self.T, logger=self.logger.getChild("airsea"))

        self.ice = ice or pygetm.ice.Ice()
        self.ice.initialize(self.T, logger=self.logger.getChild("ice"))

        self.momentum = Momentum()
        self.momentum.initialize(self.T, logger=self.logger.getChild("momentum"))

        self.turbulence = pygetm.vertical_mixing.GOTM(gotm)
        self.turbulence.initialize(
            self.T, logger=self.logger.getChild("vertical_mixing")
        )
        self.NN = self.T.array(
            z=INTERFACES,
            name="NN",
            units="s-2",
            long_name="buoyancy frequency squared",
            fill_value=FILL_VALUE,
            attrs=dict(standard_name="square_of_brunt_vaisala_frequency_in_sea_water"),
        )
        self.NN.fill(0.0)
        self.ustar_s = self.T.array(
            fill=0.0,
            name="ustar_s",
            units="m s-1",
            long_name="surface shear velocity",
            fill_value=FILL_VALUE,
            attrs=dict(_mask_output=True),
        )
        self.z0s = self.T.array(
            name="z0s",
            units="m",
            long_name="hydrodynamic surface roughness",
            fill_value=FILL_VALUE,
        )
        self.z0s.fill(z0s_min)
        self.charnock = charnock
        self.charnock_val = charnock_val
        self.z0s_min = z0s_min
        self.density = pygetm.density.Density()

        self.dpdx = self.T.array(
            name="dpdx",
            units="-",
            long_name="surface pressure gradient in x-direction",
            fill_value=FILL_VALUE,
        )
        self.dpdy = self.T.array(
            name="dpdy",
            units="-",
            long_name="surface pressure gradient in y-direction",
            fill_value=FILL_VALUE,
        )
        self.dpdx.fill(0.0)
        self.dpdy.fill(0.0)

        self.radiation = TwoBand()
        self.radiation.initialize(self.T, logger=self.logger.getChild("radiation"))

        self.temp = self.tracers.add(
            name="temp",
            units="degrees_Celsius",
            long_name="conservative temperature",
            fabm_standard_name="temperature",
            fill_value=FILL_VALUE,
            source=self.radiation.swr_abs,
            surface_flux=self.airsea.shf,
            source_scale=1.0 / (RHO0 * self.density.CP),
            rivers_follow_target_cell=True,
            precipitation_follows_target_cell=True,
            molecular_diffusivity=1.4e-7,
            attrs=dict(standard_name="sea_water_conservative_temperature"),
        )
        self.salt = self.tracers.add(
            name="salt",
            units="g kg-1",
            long_name="absolute salinity",
            fabm_standard_name="practical_salinity",
            fill_value=FILL_VALUE,
            molecular_diffusivity=1.1e-9,
            attrs=dict(standard_name="sea_water_absolute_salinity"),
        )
        self.temp.fill(5.0)
        self.salt.fill(35.0)
        self.rho = self.T.array(
            z=CENTERS,
            name="rho",
            units="kg m-3",
            long_name="density",
            fabm_standard_name="density",
            fill_value=FILL_VALUE,
            attrs=dict(standard_name="sea_water_density", _mask_output=True),
        )

        # Pressure (in dbar = m) is needed for density calculations and FABM
        self.pres = self.depth
        self.pres.fabm_standard_name = "pressure"
        self.pres.saved = True

        self.sst = self.T.array(
            name="sst",
            units="degrees_Celsius",
            long_name="sea surface temperature",
            fill_value=FILL_VALUE,
            attrs=dict(standard_name="sea_surface_temperature", _mask_output=True),
        )

        self.buoy = self.T.array(
            z=CENTERS,
            name="buoy",
            units="m s-2",
            long_name="buoyancy",
            attrs=dict(_mask_output=True),
        )

        self.nuh_ct = None
        if self.fabm and self.fabm.has_dependency("vertical_tracer_diffusivity"):
            self.nuh_ct = self.T.array(
                name="nuh_ct",
                units="m2 s-1",
                long_name="turbulent diffusivity of heat",
                z=CENTERS,
                fill_value=FILL_VALUE,
                attrs=dict(
                    standard_name="ocean_vertical_heat_diffusivity",
                    _mask_output=True,
                ),
            )
            self.nuh_ct.fabm_standard_name = "vertical_tracer_diffusivity"

        self.tracer_totals += [
            pygetm.tracer.TracerTotal(
                self.salt, units="g", per_mass=True, long_name="salt"
            ),
            pygetm.tracer.TracerTotal(
                self.temp,
                units="J",
                per_mass=True,
                scale_factor=self.density.CP,
                offset=self.density.CP * 273.15,
                long_name="heat",
            ),
        ]
        self.temp_sf = self.temp.isel(z=-1)
        self.salt_sf = self.salt.isel(z=-1)
        if wind_relative_to_ssuv:
            self.ssu = self.momentum.u.isel(z=-1)
            self.ssv = self.momentum.v.isel(z=-1)
        else:
            self.ssu = self.ssv = self.T.array()
            self.ssu.fill(0.0)

        self.relaxation = []

        self.vertical_advection = pygetm.operators.VerticalAdvection(self.T)
        self._w = self.T.array(z=INTERFACES, fill=0.0)

        if connectivity is not None:
            self.connectivity_matrix = np.empty((domain.nx, domain.nx))
            if connectivity.getm.time is not None:
                connectivity = pygetm.input.temporal_interpolation(
                    connectivity, comm=self.tiling.comm, logger=self.logger
                )
                self.input_manager._all_fields.append(
                    (connectivity.name, connectivity.data, self.connectivity_matrix)
                )
            else:
                self.connectivity_matrix[...] = connectivity
        else:
            self.connectivity_matrix = None

    def add_relaxation(self, array: Union[str, Array]) -> Tuple[Array, Array]:
        if isinstance(array, str):
            array = self[array]
        target = array.grid.array(z=array.z)
        rate = array.grid.array(z=array.z)
        self.relaxation.append((array, target, rate))
        return target, rate

    def _update_forcing_and_diagnostics(self, macro_active: bool):
        # Update density and buoyancy to keep them in sync with T and S.
        self.density.get_density(self.salt, self.temp, p=self.pres, out=self.rho)

        # From conservative temperature to in-situ sea surface temperature,
        # needed to compute heat/momentum fluxes at the surface
        self.density.get_potential_temperature(self.salt_sf, self.temp_sf, out=self.sst)

        # Calculate squared buoyancy frequency NN
        self.density.get_buoyancy_frequency(
            self.salt, self.temp, p=self.pres, out=self.NN
        )

        # Update air-sea fluxes of heat and momentum (all on T grid)
        # Note: sst is the in-situ surface temperature, whereas temp_sf is the
        # conservative surface temperature (salt_sf is absolute salinity)
        self.airsea(
            self.time,
            self.sst,
            self.ssu,
            self.ssv,
            calculate_heat_flux=True,
        )
        self.ice(macro_active, self.temp_sf, self.salt_sf, self.airsea)

        # Update surface shear velocity (used by GOTM). This requires updated
        # surface stresses and there can only be done after the airsea update.
        _pygetm.surface_shear_velocity(self.airsea.taux, self.airsea.tauy, self.ustar_s)

        if self.charnock:
            # See Craig, P. D., & Banner, M. L. (1994).
            # https://doi.org/10.1175/1520-0485(1994)024<2546:MWETIT>2.0.CO;2
            self.z0s.values[...] = np.maximum(
                self.charnock_val / GRAVITY * self.ustar_s.values**2, self.z0s_min
            )

        # Update radiation in the interior.
        # This must come after the airsea update, which is responsible for
        # calculating downwelling shortwave radiation at the water surface (swr)
        self.radiation(self.airsea.swr, self.fabm.kc if self.fabm else None)

        if self.nuh_ct is not None:
            self.turbulence.nuh.interp(self.nuh_ct)

        super()._update_forcing_and_diagnostics(macro_active)

    def transport(self, timestep: float):
        self.momentum.advance(
            timestep,
            self.airsea.taux,
            self.airsea.tauy,
            self.dpdx,
            self.dpdy,
            self.turbulence.num,
        )

        self.turbulence.advance(
            timestep,
            self.ustar_s,
            self.momentum.ustar_b,
            self.z0s,
            self.T.z0b,
            self.NN,
            self.momentum.SS,
        )

        for tracer in self.tracers:
            if (
                tracer.vertical_velocity is not None
                and tracer.vertical_velocity.all_values.any()
            ):
                tracer.vertical_velocity.interp(self._w)
                self.vertical_advection(self.momentum.w, self._w, timestep, tracer)

        self.tracers._diffuse(timestep, self.turbulence.nuh)

        self._horizontal_connectivity(timestep)

    def _horizontal_connectivity(self, timestep):
        if self.connectivity_matrix is not None:
            scale = timestep / (self.T.D.values * self.T.area.values)
            for tracer in self.tracers:
                gains = tracer.values[:, 0, :, np.newaxis] * self.connectivity_matrix
                tracer.values[:, 0, :] += gains.sum(axis=-2) * scale

        for array, target, rate in self.relaxation:
            array += (target.values - array.values) * (rate.values * timestep)
