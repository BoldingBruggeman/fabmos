import datetime

import fabmos
import fabmos.transport.tmm

FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(
            model="bb/passive",
        ),
    )
)

domain = fabmos.transport.tmm.create_domain("<PATH>")

sim = fabmos.transport.tmm.Simulator(domain, FABM_CONFIG)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=1)
)
out.request(*sim.fabm.default_outputs, time_average=True)

# 1 hour time step for BGC, 12 hour for transport
sim.start(datetime.datetime(2000, 1, 1), 3600.0, nstep_transport=12)
while sim.time < datetime.datetime(2001, 1, 1):
    sim.advance()
sim.finish()
