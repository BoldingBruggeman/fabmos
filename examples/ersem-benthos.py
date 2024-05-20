import datetime
import os

import fabmos
import fabmos.transport.null

import pygetm.legacy

FABM_CONFIG = "fabm-ersem-benthos.yaml"


domain = pygetm.legacy.domain_from_topo(
    os.path.join("../../getm-setups/NorthSea/Topo/NS6nm.v01.nc"),
    nlev=1, halox=0, haloy=0,
)
sim = fabmos.transport.null.Simulator(domain, FABM_CONFIG)

sim.fabm.get_dependency("bL2/Om_Cal").set(1.0)
sim.fabm.get_dependency("ph_reported_on_total_scale").set(8.0)
sim.fabm.get_dependency("density").set(1025.0)
sim.fabm.get_dependency("temperature").set(15.0)
sim.fabm.get_dependency("bottom_stress").set(0.0)

out = sim.output_manager.add_netcdf_file(
    "ersem_benthos.nc", interval=datetime.timedelta(days=1), save_initial=False
)
out.request(*sim.fabm.default_outputs, time_average=True)

sim.start(datetime.datetime(2000, 1, 1), 3600.0)
while sim.time < datetime.datetime(2000, 3, 1):
    sim.advance()
sim.finish()
