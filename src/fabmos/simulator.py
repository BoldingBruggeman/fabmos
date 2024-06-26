import datetime
from typing import List, Union, Optional, Tuple, Sequence
import timeit
import pstats
import functools

import cftime

import pygetm
from . import environment, Array, __version__


def log_exceptions(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger = getattr(self, "logger", None)
            domain = getattr(self, "domain", None)
            if logger is None or domain is None or domain.tiling.n == 1:
                raise
            logger.exception(str(e), stack_info=True, stacklevel=3)
            domain.tiling.comm.Abort(1)

    return wrapper


class Simulator:
    @log_exceptions
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        fabm_config: str = "fabm.yaml",
        fabm_libname: str = "fabm",
        log_level: Optional[int] = None,
        use_virtual_flux: bool = False,
    ):
        self.logger = domain.root_logger
        if log_level is not None:
            self.logger.setLevel(log_level)
        self.logger.info(f"fabmos {__version__}")

        self.fabm = pygetm.fabm.FABM(
            fabm_config,
            libname=fabm_libname,
            time_varying=pygetm.TimeVarying.MICRO,
            squeeze=True,
        )

        self.domain = domain

        self.output_manager = pygetm.output.OutputManager(
            self.domain.fields,
            rank=domain.tiling.rank,
            logger=self.logger.getChild("output_manager"),
        )

        self.input_manager = self.domain.input_manager
        self.input_manager.set_logger(self.logger.getChild("input_manager"))

        self.domain.initialize(pygetm.BAROCLINIC)
        self.domain.depth.fabm_standard_name = "pressure"

        self.radiation = environment.ShortWaveRadiation(self.domain.T)

        self.tracers = pygetm.tracer.TracerCollection(self.domain.T)
        self.tracer_totals: List[pygetm.tracer.TracerTotal] = []
        self.fabm.initialize(
            self.domain.T,
            self.tracers,
            self.tracer_totals,
            self.logger.getChild("FABM"),
        )

        if use_virtual_flux:
            self.pe = self.domain.T.array(
                name="pe",
                units="m s-1",
                long_name="net freshwater flux due to precipitation, condensation,"
                " evaporation",
            )
        self.use_virtual_flux = use_virtual_flux

        self.unmasked2d = self.domain.T.mask != 0
        if hasattr(self.domain, "mask3d"):
            self.logger.info("3D mask is explicitly defined")
            self.unmasked3d = self.domain.mask3d != 0
        else:
            self.logger.info("3D mask is not defined; inferring from 2D mask")
            self.unmasked3d = self.domain.T.array(z=pygetm.CENTERS, dtype=bool)
            self.unmasked3d.all_values[...] = self.unmasked2d[...]
        if self.unmasked3d.values.all():
            self.logger.info("No masked points")
            self.unmasked2d = None
            self.unmasked3d = None

    def __getitem__(self, key: str) -> Array:
        return self.output_manager.fields[key]

    @log_exceptions
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

        self.time = pygetm.simulation.to_cftime(time)
        self.logger.info(f"Starting simulation at {self.time}")
        self.timestep = timestep
        self.timedelta = datetime.timedelta(seconds=timestep)
        self.nstep_transport = nstep_transport
        self.istep = 0
        self.report = int(report.total_seconds() / timestep)
        if isinstance(report_totals, datetime.timedelta):
            report_totals = int(round(report_totals.total_seconds() / self.timestep))
        self.report_totals = report_totals

        self.fabm.start(self.time)
        self.update_diagnostics(macro=True)
        self.output_manager.start(self.istep, self.time)
        self._start_time = timeit.default_timer()

        # Start profiling if requested
        self._profile = None
        if profile:
            import cProfile

            pr = cProfile.Profile()
            self._profile = (profile, pr)
            pr.enable()

    @log_exceptions
    def advance(self):
        self.time += self.timedelta
        self.istep += 1
        apply_transport = self.istep % self.nstep_transport == 0
        if self.report != 0 and self.istep % self.report == 0:
            self.logger.info(self.time)

        self.output_manager.prepare_save(
            self.timestep * self.istep, self.istep, self.time, macro=apply_transport
        )

        self.logger.debug(f"fabm advancing to {self.time} (dt={self.timestep} s)")
        self.advance_fabm(self.timestep)

        if apply_transport:
            timestep_transport = self.nstep_transport * self.timestep
            self.logger.debug(
                f"transport advancing to {self.time} (dt={timestep_transport} s)"
            )
            self.transport(timestep_transport)

        self.update_diagnostics(apply_transport)

        self.output_manager.save(self.timestep * self.istep, self.istep, self.time)

    def update_diagnostics(self, macro: bool):
        self.input_manager.update(self.time, macro=macro)
        self.radiation.update(self.time)
        self.fabm.update_sources(self.timestep * self.istep, self.time)
        self.fabm.add_vertical_movement_to_sources()

        if self.report_totals != 0 and self.istep % self.report_totals == 0:
            self.report_domain_integrals()

    def advance_fabm(self, timestep: float):
        if self.tracers_with_virtual_flux:
            # Add virtual flux due to precipitation - evaporation
            scale = 1.0 - timestep * self.pe.values / self.domain.T.hn.values[0, ...]
            for tracer in self.tracers_with_virtual_flux:
                tracer.values[0, :, :] *= scale

        self.fabm.advance(timestep)

    def transport(self, timestep: float):
        pass

    @log_exceptions
    def finish(self):
        if self._profile:
            name, pr = self._profile
            pr.disable()
            profile_path = f"{name}-{self.domain.tiling.rank:03}.prof"
            self.logger.info(f"Writing profiling report to {profile_path}")
            with open(profile_path, "w") as f:
                ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.TIME)
                ps.print_stats()

        nsecs = timeit.default_timer() - self._start_time
        self.logger.info(f"Time spent in main loop: {nsecs:.3f} s")
        self.output_manager.close(self.timestep * self.istep, self.time)

    @property
    def totals(
        self,
    ) -> Optional[Sequence[Tuple[pygetm.tracer.TracerTotal, float, float]]]:
        """Global totals of tracers.

        Returns:
            A list with (tracer_total, total, mean) tuples on the root subdomains.
            On non-root subdomains it returns None
        """
        total_volume = (self.domain.T.hn * self.domain.T.area).global_sum(
            where=self.unmasked3d
        )
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
