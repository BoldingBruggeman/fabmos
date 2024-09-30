from typing import Optional
import os

import pygetm
from .. import simulator
import fabmos
from ..domain import compress, _update_coordinates, drop_grids


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        fabm_config: str = "fabm.yaml",
        vertical_coordinates: Optional[fabmos.vertical_coordinates.Base] = None,
        log_level: Optional[int] = None,
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_hz_only")

        domain = compress(domain)

        # Drop unused domain variables. Some of these will be NaN,
        # which causes check_finite to fail.
        drop_grids(
            domain,
            domain.U,
            domain.V,
            domain.X,
            domain.UU,
            domain.UV,
            domain.VU,
            domain.VV,
        )
        for name in ("dx", "dy", "idx", "idy"):
            del domain.fields[name]

        super().__init__(
            domain,
            fabm_config,
            fabm_libname=fabm_libname,
            log_level=log_level,
            use_virtual_flux=False,
        )

        if vertical_coordinates is not None:
            domain.vertical_coordinates = vertical_coordinates
        _update_coordinates(domain.T, domain.global_area)
