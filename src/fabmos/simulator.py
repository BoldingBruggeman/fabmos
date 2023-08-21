import datetime
from typing import List, Union

import cftime

import pygetm
from . import environment

class Simulator:
    def __init__(self, domain: pygetm.domain.Domain, fabm_config: str = "fabm.yaml"):
        self.logger = domain.root_logger

        self.fabm = pygetm.fabm.FABM(fabm_config)

        self.domain = domain

        self.output_manager = pygetm.output.OutputManager(
            self.domain.fields,
            rank=domain.tiling.rank,
            logger=self.logger.getChild("output_manager"),
        )

        self.input_manager = self.domain.input_manager
        self.input_manager.set_logger(self.logger.getChild("input_manager"))

        self.domain.initialize(pygetm.BAROCLINIC)

        self.radiation = environment.ShortWaveRadiation(self.domain.T)

        self.tracers = pygetm.tracer.TracerCollection(self.domain.T)
        self.tracer_totals: List[pygetm.tracer.TracerTotal] = []
        self.fabm.initialize(
            self.domain.T,
            self.tracers,
            self.tracer_totals,
            self.logger.getChild("FABM"),
        )

    def __getitem__(self, key: str) -> pygetm.core.Array:
        return self.output_manager.fields[key]

    def start(
        self,
        time: Union[cftime.datetime, datetime.datetime],
        timestep: float,
        nstep_transport: int = 1,
        report: datetime.timedelta = datetime.timedelta(days=1),
    ):
        self.time = pygetm.simulation.to_cftime(time)
        self.logger.info(f"Starting simulation at {self.time}")
        self.timestep = timestep
        self.timedelta = datetime.timedelta(seconds=timestep)
        self.nstep_transport = nstep_transport
        self.istep = 0
        self.report = int(report.total_seconds() / timestep)

        self.fabm.start(self.time)
        self.update_diagnostics()
        self.output_manager.start(self.istep, self.time)

    def advance(self):
        self.time += self.timedelta
        self.istep += 1
        apply_transport = self.istep % self.nstep_transport == 0
        if self.report != 0 and self.istep % self.report == 0:
            self.logger.info(self.time)

        self.output_manager.prepare_save(
            self.timestep * self.istep, self.istep, self.time, macro=True
        )

        self.logger.debug(f"fabm advancing to {self.time} (dt={self.timestep} s)")
        self.fabm.advance(self.timestep)

        if apply_transport:
            self.transport(self.nstep_transport * self.timestep)

        self.update_diagnostics()

        self.output_manager.save(self.timestep * self.istep, self.istep, self.time)

    def update_diagnostics(self):
        self.input_manager.update(self.time, macro=True)
        self.radiation.update(self.time)
        self.fabm.update_sources(self.time)

    def transport(self, timestep: float):
        self.logger.debug(f"transport advancing to {self.time} (dt={timestep} s)")

    def finish(self):
        self.output_manager.close(self.timestep * self.istep, self.time)
