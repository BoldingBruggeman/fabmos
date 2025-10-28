import datetime
from typing import Optional, Union
import os

import numpy as np
import xarray as xr

import pygetm
import pygetm.output.netcdf
from pygetm.constants import INTERFACES, CENTERS, FILL_VALUE

from .. import simulator


def record_transports(
    sim: pygetm.Simulation, ncfile: Union[os.PathLike[str], str], **kwargs
) -> pygetm.output.netcdf.NetCDFFile:
    """Set up NetCDF output file to record pygetm transports for use with fabmos."""
    kwargs.setdefault("add_coordinates", False)
    kwargs.setdefault("default_dtype", np.float32)
    out = sim.output_manager.add_netcdf_file(ncfile, save_initial=True, **kwargs)
    out.request("pk", "qk", "ww", "nuh", time_average=True)
    out.request("hnu", "hnv", time_average=True)
    out.request("hnt", time_average=False)
    return out


class Simulator(simulator.Simulator):
    def __init__(
        self,
        domain: pygetm.domain.Domain,
        transport_nc: Union[os.PathLike[str], str],
        *,
        fabm: Union[str, pygetm.fabm.FABM] = "fabm.yaml",
        log_level: Optional[int] = None,
    ):
        with xr.open_dataset(transport_nc) as ds:
            nz = ds.sizes["z"]

        super().__init__(
            domain,
            nz=nz,
            fabm=fabm,
            log_level=log_level,
            use_virtual_flux=False,
            add_swr=False,
            process_vertical_movement=False,
            velocity_grids=1,
            halox=2,
            haloy=2,
        )

        self.uh = self.T.ugrid.array(
            z=CENTERS,
            fill=0.0,
            name="uh",
            units="m2 s-1",
            long_name="transport in x-direction",
            fill_value=FILL_VALUE,
            attrs=dict(_require_halos=True),
        )
        self.vh = self.T.vgrid.array(
            z=CENTERS,
            fill=0.0,
            name="vh",
            units="m2 s-1",
            long_name="transport in y-direction",
            fill_value=FILL_VALUE,
            attrs=dict(_require_halos=True),
        )
        self.w = self.T.array(
            z=INTERFACES,
            name="w",
            units="m s-1",
            long_name="velocity in z-direction",
            fill_value=FILL_VALUE,
        )
        self.nuh = self.T.array(
            z=INTERFACES,
            name="nuh",
            units="m2 s-1",
            long_name="vertical tracer diffusivity",
            fill_value=FILL_VALUE,
        )

        # The old thicknesses on the tracer grid are used for advection
        self.T.ho = self.T.hn

        # Thicknesses on velocity grids need halos for calculating velocities
        # that drive advection
        self.T.ugrid.hn.attrs["_require_halos"] = True
        self.T.vgrid.hn.attrs["_require_halos"] = True

        def from_nc(name: str) -> xr.DataArray:
            return pygetm.input.from_nc(transport_nc, name, mask_and_scale=False)

        kwargs = dict(on_grid=pygetm.input.OnGrid.ALL)
        self.T.hn.set(from_nc("hnt"), **kwargs)

        self.transport_updaters = []
        kwargs.update(updater_collection=self.transport_updaters)
        self.T.ugrid.hn.set(from_nc("hnu"), **kwargs)
        self.T.vgrid.hn.set(from_nc("hnv"), **kwargs)
        self.uh.set(from_nc("pk"), **kwargs)
        self.vh.set(from_nc("qk"), **kwargs)
        self.w.set(from_nc("ww"), **kwargs)
        self.nuh.set(from_nc("nuh"), **kwargs)

        self.u = self.T.ugrid.array(z=CENTERS)
        self.v = self.T.vgrid.array(z=CENTERS)

        self.T.open_boundaries.u.fill(0.0)
        self.T.open_boundaries.v.fill(0.0)

    def _start(self):
        super()._start()
        self.tracers.start()
        self.T.open_boundaries.start()

    def _update_forcing_and_diagnostics(self, macro_active: bool):
        # Called at the end of a timestep, just after the input manager has updated
        # all fields.
        pygetm._pygetm.thickness2vertical_coordinates(
            self.T.mask, self.T.H, self.T.hn, self.T.zc, self.T.zf
        )
        if self.depth.saved:
            pygetm._pygetm.thickness2center_depth(self.T.mask, self.T.hn, self.depth)

        super()._update_forcing_and_diagnostics(macro_active)

    def transport(self, timestep: float):
        # Update open boundary conditions for tracers just before transport operators.
        # This ensures FABM sources at the open boundaries are ignored.
        # (for that, it should happen after fabm.advance)
        for tracer in self.tracers:
            tracer.open_boundaries.update()

        # Start tracer halo exchange (to prepare for advection)
        for tracer in self.tracers:
            tracer.update_halos_start(self.tracers._advection.halo1)

        # Update velocities and diffusivity to time stage in between old and new tracer.
        transport_time = self.time - datetime.timedelta(seconds=0.5 * timestep)
        self.logger.debug(
            f"transporting using velocities and diffusivity from {transport_time}"
        )
        self.input_manager.update(transport_time, fields=self.transport_updaters)

        # Calculate horizontal velocities from transports
        np.divide(self.uh.all_values, self.T.ugrid.hn.all_values, out=self.u.all_values)
        np.divide(self.vh.all_values, self.T.vgrid.hn.all_values, out=self.v.all_values)

        # Advection and diffusion
        self.tracers.advance(timestep, self.u, self.v, self.w, self.nuh)
