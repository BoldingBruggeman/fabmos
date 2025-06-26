import cftime
import os

import fabmos.transport.tmm
import fabmos

tm_config_dir = "."
calendar = "360_day"

# FABM: a single passive tracer
FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(
            model="tracer",
        ),
    )
)

domain = fabmos.transport.tmm.create_domain(tm_config_dir)

sim = fabmos.transport.tmm.Simulator(domain, calendar=calendar, fabm=FABM_CONFIG)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=1, interval_units=fabmos.TimeUnit.MONTHS
)
out.request("temp", "salt", "ice", "wind", *sim.fabm.default_outputs, time_average=True)

start = cftime.datetime(2000, 1, 1, calendar=calendar)
stop = cftime.datetime(2001, 1, 1, calendar=calendar)
sim.start(start, 12 * 3600)
while sim.time < stop:
    sim.advance()
sim.finish()
