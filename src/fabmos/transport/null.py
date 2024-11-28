from typing import Optional, Union
import os

import pygetm
from .. import simulator
import fabmos
from ..domain import compress, _update_coordinates


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        *,
        vertical_coordinates: Optional[fabmos.vertical_coordinates.Base] = None,
        fabm: Union[str, pygetm.fabm.FABM] = "fabm.yaml",
        log_level: Optional[int] = None,
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_hz_only")

        super().__init__(
            compress(domain),
            nz=1 if vertical_coordinates is None else vertical_coordinates.nz,
            fabm=fabm,
            fabm_libname=fabm_libname,
            log_level=log_level,
            use_virtual_flux=False,
        )

        if vertical_coordinates:
            vertical_coordinates.initialize(
                self.T, logger=self.logger.getChild("vertical_coordinates")
            )
        _update_coordinates(
            self.T, self.depth, vertical_coordinates=vertical_coordinates
        )
