import cftime
import os

import numpy as np
import fabmos.transport.tmm
import fabmos

tm_config_dir = "."  # directory with a TM configuration from https://sites.google.com/view/samarkhatiwala-research-tmm
calendar = "360_day"  # any valid calendar recognized by cftime, see https://cfconventions.org/cf-conventions/cf-conventions.html#calendar

script_dir = os.path.dirname(__file__)
fabm_yaml = os.path.join(
    script_dir, "../../extern/fabm/extern/pisces/testcases/fabm.yaml"
)

domain = fabmos.transport.tmm.create_domain(tm_config_dir)

sim = fabmos.transport.tmm.Simulator(domain, calendar=calendar, fabm=fabm_yaml)

# Crude estimate of turbulent diffusivity from temperature-only MLD
# PISCES uses this to determine turbocline depth
temp = fabmos.transport.tmm.get_mat_array(
    "GCM/Theta_gcm.mat",
    "Tgcm",
    domain._grid_file,
    times=fabmos.transport.tmm.climatology_times(calendar),
)
Kzval = np.where(temp.values < temp.values[:, :1, :, :] - 0.5, 1e-5, 1e-3)
Kz = fabmos.transport.tmm._wrap_ongrid_array(
    Kzval, domain._grid_file, times=fabmos.transport.tmm.climatology_times(calendar)
)
sim.fabm.get_dependency("vertical_tracer_diffusivity").set(
    Kz, on_grid=fabmos.input.OnGrid.ALL, climatology=True
)

sim.fabm.get_dependency("surface_air_pressure").set(101325.0)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=7, interval_units=fabmos.TimeUnit.DAYS, save_initial=False
)
out.request(*sim.fabm.state_variables, time_average=True)

start = cftime.datetime(2000, 1, 1, calendar=calendar)
stop = cftime.datetime(2001, 1, 1, calendar=calendar)
sim.start(start, timestep=7200, transport_timestep=12 * 3600)
while sim.time < stop:
    sim.advance()
sim.finish()
