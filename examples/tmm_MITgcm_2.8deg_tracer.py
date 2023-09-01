import cftime
import os
import glob

import fabmos.transport.tmm
import fabmos

root = "."
calendar = "360_day"

# FABM: a single passive tracer
FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(
            model="bb/passive",
        ),
    )
)

domain = fabmos.transport.tmm.create_domain(os.path.join(root, "grid.mat"))

matrix_files = sorted(
    glob.glob(os.path.join(root, "Matrix5/TMs/matrix_nocorrection_??.mat"))
)
matrix_times = fabmos.transport.tmm.climatology_times(calendar=calendar)
sim = fabmos.transport.tmm.Simulator(
    domain,
    matrix_files=matrix_files,
    matrix_times=matrix_times,
    fabm_config=FABM_CONFIG,
)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=1, interval_units=fabmos.TimeUnit.MONTHS
)
out.request("temp", "salt", "ice", "wind", *sim.fabm.default_outputs, time_average=True)

sim.start(cftime.datetime(2000, 1, 1, calendar=calendar), 12 * 3600.0, nstep_transport=1)
while sim.time < cftime.datetime(2001, 1, 1, calendar=calendar):
    sim.advance()
sim.finish()
