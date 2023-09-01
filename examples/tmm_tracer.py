import sys
import cftime
import os
import argparse
import glob
import datetime

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

parser = argparse.ArgumentParser(
    description=f"Python implementation of the Transport Matrix Method (TMM) - Khatiwala et. al (2005)",
    prog=f"mpiexec -np <N> {sys.argv[0]}",
    epilog="Implemented by BB in the OceanICU Horizon Europe project (Grant No.101083922).",
)
parser.add_argument(
    "path",
    type=str,
    nargs="?",
    default="./",
    help="the path to the original TMM data",
)
parser.add_argument(
    "--start_time",
    type=str,
    default="2000-01-01 00:00:00",
    help="integration start time - 2000-01-01 00:00:00",
)
parser.add_argument(
    "--stop_time",
    type=str,
    default="2001-01-01 00:00:00",
    help="integration stop time - 2001-01-01 00:00:00",
)
parser.add_argument(
    "--area",
    type=float,
    nargs=4,
    default=[
        300.0,
        330.0,
        30.0,
        55.0,
    ],
    help=f"area with concentration set - lon1, lon2, lat1, lat2",
)
parser.add_argument(
    "--calendar",
    type=str,
    default="360_day",
    help=f"calendar to use - default - 360_day",
)
args = parser.parse_args()

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

# grid_file = os.path.join(args.path,"grid.mat")
domain = fabmos.transport.tmm.create_domain(os.path.join(args.path, "grid.mat"))

# calendar = "360_day"
sim = fabmos.transport.tmm.Simulator(
    domain,
    calendar=args.calendar,
    fabm_config=FABM_CONFIG,
)
sim["tracer_c"].fill(0)
sim["tracer_c"].values[
    :,
    (domain.T.lon.values > args.area[0])
    & (domain.T.lon.values < args.area[1])
    & (domain.T.lat.values > args.area[2])
    & (domain.T.lat.values < args.area[3]),
] = 1.0

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=30)
)
out.request("temp", "salt", "ice", "wind", *sim.fabm.default_outputs, time_average=True)

# 1 hour time step for BGC, 12 hour for transport
fmt = "%Y-%m-%d %H:%M:%S"
sim.start(cftime.datetime.strptime(args.start_time, fmt, calendar=args.calendar), 12 * 3600.0)

while sim.time < cftime.datetime.strptime(args.stop_time, fmt, calendar=args.calendar):
    sim.advance()
sim.finish()
