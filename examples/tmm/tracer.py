import sys
import cftime
import argparse
import datetime

import fabmos
import fabmos.transport.tmm

FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(model="bb/passive", parameters=dict(conserved=True)),
    )
)

parser = argparse.ArgumentParser(
    description="Python implementation of the Transport Matrix Method (TMM) - Khatiwala et. al (2005)",
    prog=f"mpiexec -n <N> {sys.argv[0]}",
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
    help="area with concentration set - lon1, lon2, lat1, lat2",
)
parser.add_argument(
    "--calendar",
    type=str,
    default="360_day",
    help="calendar to use - default - 360_day",
)
args = parser.parse_args()

domain = fabmos.transport.tmm.create_domain(args.path)

sim = fabmos.transport.tmm.Simulator(domain, calendar=args.calendar, fabm=FABM_CONFIG)
sim["tracer_c"].fill(
    (sim.T.lon.values >= args.area[0])
    & (sim.T.lon.values <= args.area[1])
    & (sim.T.lat.values >= args.area[2])
    & (sim.T.lat.values <= args.area[3])
)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=30), save_initial=False
)
out.request("temp", "salt", "ice", "wind", *sim.fabm.default_outputs, time_average=True)

fmt = "%Y-%m-%d %H:%M:%S"
sim.start(
    cftime.datetime.strptime(args.start_time, fmt, calendar=args.calendar), 12 * 3600.0
)

while sim.time < cftime.datetime.strptime(args.stop_time, fmt, calendar=args.calendar):
    sim.advance()
sim.finish()
