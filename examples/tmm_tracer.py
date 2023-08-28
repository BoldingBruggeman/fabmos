import datetime
import os

import fabmos
import fabmos.transport.tmm
import pygetm

FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(
            model="bb/passive",
        ),
    )
)

TMM_matrix_config = {
    "matrix_type": "periodic",
    #    "path": ".",
    "path": "/data/kb/OceanICU/MITgcm_2.8deg/Matrix5/TMs",
    "constant": {"Ae_fname": "matrix_nocorrection_01.mat", "Ai_fname": "matrix_nocorrection_01.mat"},
    #"periodic": {"Ae_template": "Ae_%02d", "Ai_template": "Ai_%02d", "num_periods": 12},
    "periodic": {"Ae_template": "matrix_nocorrection_%02d.mat", "Ai_template": "matrix_nocorrection_%02d.mat", "num_periods": 12, "base_number": 1},
    "time_dependent": {},
}

grid_file = "./MITgcm_2.8deg/grid.mat"
domain = fabmos.transport.tmm.create_domain(grid_file)

temp_src = fabmos.transport.tmm.get_mat_array(
    os.path.join(os.path.dirname(grid_file), "GCM/Theta_gcm.mat"),
    "Tgcm",
    grid_file,
    times=fabmos.transport.tmm.climatology_times(),
)
temp = domain.T.array(z=pygetm.CENTERS, name="temp")
temp.set(temp_src, on_grid=pygetm.input.OnGrid.ALL, climatology=True)

# fabmos.transport.tmm.tmm_matrix_config(TMM_matrix_config)
sim = fabmos.transport.tmm.Simulator(domain, TMM_matrix_config, FABM_CONFIG)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=1)
)
out.request("temp", *sim.fabm.default_outputs, time_average=True)

# 1 hour time step for BGC, 12 hour for transport
sim.start(datetime.datetime(2000, 1, 1), 3600.0, nstep_transport=12)
while sim.time < datetime.datetime(2000, 1, 2):
    sim.advance()
sim.finish()
