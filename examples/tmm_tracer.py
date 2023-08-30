import sys
import datetime
import os
import argparse

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
    "array_format": "csc",
    # "array_format": "csr",
    "constant": {
        "Ae_fname": "matrix_nocorrection_01.mat",
        "Ai_fname": "matrix_nocorrection_01.mat",
    },
    # "periodic": {"Ae_template": "Ae_%02d", "Ai_template": "Ai_%02d", "num_periods": 12},
    "periodic": {
        "Ae_template": "matrix_nocorrection_%02d.mat",
        "Ai_template": "matrix_nocorrection_%02d.mat",
        "num_periods": 12,
        "base_number": 1,
    },
    "time_dependent": {},
}

parser = argparse.ArgumentParser(
    description=f"Python implementation of the Transport Matrix Method (TMM) - Khatiwala et. al (2005)",
    prog=f'mpiexec -np <N> {sys.argv[0]}',
    epilog="Implemented by BB in the OceanICU Horizon Europe project (Grant No.101083922).",
)
parser.add_argument(
    "path",
    type=str,
    nargs="?",
    default="./",
    help="the path to the original TMM data",
)
args = parser.parse_args()

# grid_file = os.path.join(args.path,"grid.mat")
domain = fabmos.transport.tmm.create_domain(os.path.join(args.path, "grid.mat"))

sim = fabmos.transport.tmm.Simulator(domain, TMM_matrix_config, FABM_CONFIG)

sim['tracer_c'].fill(0)
#sim['tracer_c'].values[:, (domain.T.lon.values > 320) & (domain.T.lon.values < 330) & (domain.T.lat.values > 50) & (domain.T.lat.values < 55)] = 10.
sim['tracer_c'].values[:, (domain.T.lon.values > 300) & (domain.T.lon.values < 330) & (domain.T.lat.values > 30) & (domain.T.lat.values < 55)] = 1.

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=1)
)
out.request("temp", "salt", "ice", "wind", *sim.fabm.default_outputs, time_average=True)

# 1 hour time step for BGC, 12 hour for transport
sim.start(datetime.datetime(2000, 1, 1), 12 * 3600.0, nstep_transport=1)
#while sim.time < datetime.datetime(2001, 1, 1):
while sim.time < datetime.datetime(2000, 2, 1):
    sim.advance()
sim.finish()
