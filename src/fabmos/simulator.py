import datetime
from typing import List, Union, Optional, Tuple, Sequence
import os.path

import cftime

import numpy as np
import pygetm
from . import environment, __version__


class Simulator(pygetm.simulation.BaseSimulation):
    @pygetm.simulation.log_exceptions
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        nz: int,
        mask3d: Optional[np.ndarray] = None,
        fabm: Optional[Union[str, pygetm.fabm.FABM]] = "fabm.yaml",
        fabm_libname: str = os.path.join(os.path.dirname(pygetm.fabm.__file__), "fabm"),
        log_level: Optional[int] = None,
        use_virtual_flux: bool = False,
        squeeze: bool = True,
        add_swr: bool = True,
        process_vertical_movement: bool = True,
        tiling: Optional[pygetm.parallel.Tiling] = None,
        velocity_grids: int = 0,
        halox: int = 0,
        haloy: int = 0,
    ):
        super().__init__(domain, log_level=log_level, tiling=tiling)

        self.logger.info(f"fabmos {__version__}")
        self.T = domain.create_grids(
            nz,
            halox=halox,
            haloy=haloy,
            fields=self._fields,
            tiling=self.tiling,
            input_manager=self.input_manager,
            velocity_grids=velocity_grids,
        )

        self.unmasked2d = self.T.mask != 0
        if domain.comm.bcast(mask3d is not None):
            self.logger.info("3D mask is explicitly defined")
            self.T.mask3d = self.T.array(z=pygetm.CENTERS, dtype=np.intc, fill_value=0)
            self.T.mask3d.scatter(mask3d)
            self.unmasked3d = self.T.mask3d != 0
            self.T._land3d = ~self.unmasked3d.all_values
            self.T._land3d_if = np.empty_like(
                self.T._land3d, shape=(nz + 1, *self.T._land3d.shape[1:])
            )
            self.T._land3d_if[0] = self.T._land3d[0]
            self.T._land3d_if[1:-1] = self.T._land3d[:-1] | self.T._land3d[1:]
            self.T._land3d_if[-1] = self.T._land3d[-1]
            self.T.bottom_indices = self.T.array(dtype=np.intc, fill=0)
            self.T.bottom_indices.all_values[:, :] = nz - self.T._land3d.sum(axis=0)
        else:
            self.logger.info("3D mask is not defined; inferring from 2D mask")
            self.unmasked3d = self.T.array(z=pygetm.CENTERS, dtype=bool)
            self.unmasked3d.all_values[...] = self.unmasked2d.all_values

        if self.unmasked3d.values.all():
            self.logger.info("No masked points")
            self.unmasked2d = None
            self.unmasked3d = None

        if fabm is not None and not isinstance(fabm, pygetm.fabm.FABM):
            fabm = pygetm.fabm.FABM(
                fabm,
                libname=fabm_libname,
                time_varying=pygetm.TimeVarying.MICRO,
                squeeze=squeeze,
            )
        self.fabm = fabm

        self.depth = self.T.array(
            z=pygetm.CENTERS,
            name="pres",
            units="dbar",
            long_name="pressure",
            fabm_standard_name="depth",
            fill_value=pygetm.FILL_VALUE,
        )
        self.depth.fabm_standard_name = "pressure"
        self.tracers = pygetm.tracer.TracerCollection(
            self.T, logger=domain.root_logger.getChild("tracers")
        )
        self.tracer_totals: List[pygetm.tracer.TracerTotal] = []
        self.surface_radiation = None
        if self.fabm:
            self.fabm.initialize(
                self.T,
                self.tracers,
                self.tracer_totals,
                self.logger.getChild("FABM"),
            )
            if add_swr and self.fabm.has_dependency(
                "surface_downwelling_shortwave_flux"
            ):
                self.surface_radiation = environment.ShortWaveRadiation(
                    self.T, self.logger.getChild("ShortWaveRadiation")
                )

        if use_virtual_flux:
            self.pe = self.T.array(
                name="pe",
                units="m s-1",
                long_name="net freshwater flux due to precipitation, condensation,"
                " evaporation",
            )
        self.use_virtual_flux = use_virtual_flux
        self.process_vertical_movement = process_vertical_movement

    @pygetm.simulation.log_exceptions
    def start(
        self,
        time: Union[cftime.datetime, datetime.datetime],
        timestep: Union[float, datetime.timedelta],
        transport_timestep: Optional[float] = None,
        report: datetime.timedelta = datetime.timedelta(days=1),
        report_totals: Union[int, datetime.timedelta] = datetime.timedelta(days=10),
        profile: Optional[str] = None,
    ):
        if isinstance(timestep, datetime.timedelta):
            timestep = timestep.total_seconds()
        if transport_timestep is None:
            transport_timestep = timestep
        nstep_transport = transport_timestep / timestep
        self.logger.info(
            f"Using transport timestep of {transport_timestep} s, which is"
            f" {nstep_transport} * the biogeochemical timestep of {timestep} s"
        )
        nstep_transport = int(round(nstep_transport))
        if (transport_timestep / (nstep_transport * timestep) % 1.0) > 1e-8:
            raise Exception(
                f"The transport timestep of {transport_timestep} s must be an"
                f" exact multiple of the biogeochemical timestep of {timestep} s"
            )
        self.nstep_transport = nstep_transport
        super().start(
            time,
            timestep,
            nstep_transport,
            report=report,
            report_totals=report_totals,
            profile=profile,
        )

    def _start(self):
        self.tracers_with_virtual_flux: List[pygetm.tracer.Tracer] = []
        if self.use_virtual_flux:
            for tracer in self.tracers:
                if not tracer.precipitation_follows_target_cell:
                    self.tracers_with_virtual_flux.append(tracer)
        if self.tracers_with_virtual_flux:
            self.logger.info(
                f"Virtual flux due to net freshwater flux will be applied to"
                f" {', '.join([t.name for t in self.tracers_with_virtual_flux])}"
            )
        else:
            self.logger.info("Virtual tracer flux due to net freshwater flux not used")

        if self.fabm:
            self.fabm.start(self.timestep, self.time)

    def _advance_state(self, macro_active: bool):
        self.logger.debug(f"fabm advancing to {self.time} (dt={self.timestep} s)")
        self.advance_fabm(self.timestep)

        if macro_active:
            timestep_transport = self.nstep_transport * self.timestep
            self.logger.debug(
                f"transport advancing to {self.time} (dt={timestep_transport} s)"
            )
            self.transport(timestep_transport)

    def _update_forcing_and_diagnostics(self, macro_active: bool):
        if self.surface_radiation:
            self.surface_radiation.update(self.time)
        if self.fabm:
            self.fabm.update_sources(self.timestep * self.istep, self.time)
            if self.process_vertical_movement:
                self.fabm.add_vertical_movement_to_sources()

        if self.report_totals != 0 and self.istep % self.report_totals == 0:
            self.report_domain_integrals()

    def advance_fabm(self, timestep: float):
        if self.tracers_with_virtual_flux:
            # Add virtual flux due to precipitation - evaporation
            scale = 1.0 - timestep * self.pe.values / self.T.hn.values[0, ...]
            for tracer in self.tracers_with_virtual_flux:
                tracer.values[0, :, :] *= scale

        if self.fabm:
            self.fabm.advance(timestep)

    def transport(self, timestep: float):
        pass

    @property
    def totals(
        self,
    ) -> Optional[Sequence[Tuple[pygetm.tracer.TracerTotal, float, float]]]:
        """Global totals of tracers.

        Returns:
            A list with (tracer_total, total, mean) tuples on the root subdomains.
            On non-root subdomains it returns None
        """
        total_volume = (self.T.hn * self.T.area).global_sum(where=self.unmasked3d)
        tracer_totals = [] if total_volume is not None else None
        if self.fabm:
            self.fabm.update_totals()
        for tt in self.tracer_totals:
            grid = tt.array.grid
            total = tt.array * grid.area
            if tt.scale_factor != 1.0:
                total.all_values *= tt.scale_factor
            if tt.offset != 0.0:
                total.all_values += tt.offset * grid.area.all_values
            if total.ndim == 3:
                total.all_values *= grid.hn.all_values
                total = total.global_sum(where=self.unmasked3d)
            else:
                total = total.global_sum(where=self.unmasked2d)
            if total is not None:
                mean = (total / total_volume - tt.offset) / tt.scale_factor
                tracer_totals.append((tt, total, mean))
        return tracer_totals

    def report_domain_integrals(self):
        """Write totals of selected variables over the global domain
        (those in :attr:`tracer_totals`) to the log.
        """
        tracer_totals = self.totals
        if tracer_totals:
            self.logger.info("Integrals over global domain:")
            for tt, total, mean in tracer_totals:
                ar = tt.array
                long_name = tt.long_name if tt.long_name is not None else ar.long_name
                units = tt.units if tt.units is not None else f"{ar.units} m3"
                self.logger.info(
                    f"  {long_name}: {total:.15e} {units}"
                    f" (mean {ar.long_name}: {mean} {ar.units})"
                )
