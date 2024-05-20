
from typing import Optional
import os

import pygetm
from .. import simulator
from ..domain import compress, _update_coordinates


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        fabm_config: str = "fabm.yaml",
        log_level: Optional[int] = None,
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_hz_only")

        domain = compress(domain)

        super().__init__(
            domain,
            fabm_config,
            fabm_libname=fabm_libname,
            log_level=log_level,
            use_virtual_flux=False,
        )

        _update_coordinates(domain.T, domain.uncompressed_area)
