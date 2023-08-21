import datetime
import os

import fabmos

FABM_CONFIG = dict(
    instances=dict(
        seagrass=dict(
            model="csiro/seagrass",
            coupling=dict(
                D_A_C="zero",
                D_A_N="zero",
                D_A_P="zero",
                D_B_C="zero_hz",
                D_B_N="zero_hz",
                D_B_P="zero_hz",
                DIC="zero",
                O2="zero",
                N_source="zero_hz",
                P_source="zero_hz",
            ),
        ),
    )
)

MINLON = -7.33
MAXLON = -1.30
MINLAT = 49.60
MAXLAT = 59.63

MINLON = -6.00
MAXLON = -3.11
MINLAT = 49.77
MAXLAT = 51.36

domain = fabmos.domain.create_spherical_at_resolution(
    MINLON, MAXLON, MINLAT, MAXLAT, 250.0, nz=1
)

if not os.path.isfile("SW_England.nc"):
    import pygetm.input.emodnet

    pygetm.input.emodnet.get(MINLON, MAXLON, MINLAT, MAXLAT).to_netcdf("SW_England.nc")

domain.set_bathymetry(-fabmos.input.from_nc("SW_England.nc", "bath"))
domain.mask_shallow(0.01)

sim = fabmos.Simulator(domain, FABM_CONFIG)

#sim.fabm.get_dependency("temperature").set(15.0)
sim.fabm.get_dependency("seagrass/E_par", sim.radiation.swr)
sim.fabm.get_dependency("temperature", sim.radiation.swr)
sim.fabm.get_dependency("seagrass/N").set(5.0)
sim.fabm.get_dependency("seagrass/P").set(5.0)

sim.radiation.tcc.set(0.1)

out = sim.output_manager.add_netcdf_file(
    "seagrass.nc", interval=datetime.timedelta(days=1)
)
out.request('swr', *sim.fabm.default_outputs, time_average=True)

sim.start(datetime.datetime(2000, 1, 1), 3600.0, nstep_transport=10)
while sim.time < datetime.datetime(2001, 1, 1):
    sim.advance()
sim.finish()
