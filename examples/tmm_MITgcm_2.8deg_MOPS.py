import cftime
import os

import fabmos.transport.tmm
import fabmos

root = "."
calendar = "360_day"

script_dir = os.path.dirname(__file__)
fabm_yaml = os.path.join(script_dir, "../extern/fabm-mops/testcases/fabm.yaml")

domain = fabmos.transport.tmm.create_domain(os.path.join(root, "grid.mat"))

sim = fabmos.transport.tmm.Simulator(domain, calendar=calendar, fabm_config=fabm_yaml)

sim.fabm.get_dependency("mole_fraction_of_carbon_dioxide_in_air").set(280.0)
sim.fabm.get_dependency("surface_air_pressure").set(101325.0)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=1, interval_units=fabmos.TimeUnit.MONTHS
)
out.request(*sim.fabm.default_outputs, time_average=True)

start = cftime.datetime(2000, 1, 1, calendar=calendar)
stop = cftime.datetime(2001, 1, 1, calendar=calendar)
sim.start(start, timestep=12 * 3600)
while sim.time < stop:
    sim.advance()
sim.finish()
