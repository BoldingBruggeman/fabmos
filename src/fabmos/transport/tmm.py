import pygetm
from .. import simulator

def create_domain(path: str) -> pygetm.domain.Domain:
    # open files, read 2D lon, lat, mask (pygetm convention, 0=land, 1=water), H
    return pygetm.domain.create_spherical(lon=lon, lat=lat, mask=mask, H=H)

class Simulator(simulator.Simulator):
    def __init__(self, domain: pygetm.domain.Domain, fabm_config: str = "fabm.yaml"):
        super().__init__(domain, fabm_config)
        self.tmm_logger = self.logger.getChild("tmm")

    def transport(self, timestep: float):
        for tracer in self.tracers:
            #update tracer.all_values in place
