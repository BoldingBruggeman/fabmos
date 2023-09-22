import cftime
import os
import datetime

import fabmos
import fabmos.transport.tmm
import fabmos.input.riverlist

import pandas as pd

tm_config_dir = "."  # directory with a TM configuration from http://kelvin.earth.ox.ac.uk/spk/Research/TMM/TransportMatrixConfigs/
calendar = "360_day"  # any valid calendar recognized by cftime, see https://cfconventions.org/cf-conventions/cf-conventions.html#calendar

script_dir = os.path.dirname(__file__)
fabm_yaml = os.path.join(
    script_dir, "../../extern/fabm-mops/testcases/fabm_with_runoff.yaml"
)

domain = fabmos.transport.tmm.create_domain(tm_config_dir)

sim = fabmos.transport.tmm.Simulator(domain, calendar=calendar, fabm_config=fabm_yaml)

sim.fabm.get_dependency("mole_fraction_of_carbon_dioxide_in_air").set(280.0)
sim.fabm.get_dependency("surface_air_pressure").set(101325.0)

# Load rivers from https://doi.org/10.1029/96JD00932
fn = os.path.join(script_dir, "rivrstat.txt")
rlat, rlon, rflow = pd.read_fwf(fn, skiprows=7, header=None, usecols=(3, 4, 6)).T.values

# Exclude rivers flowing into Arctic as in https://doi.org/10.5194/bg-10-8401-2013
keep = rlat <= 60
rlat, rlon, rflow = rlat[keep], rlon[keep], rflow[keep]

# Convert river list to gridded [2D] river inputs (level increase in m s-1)
runoff = fabmos.input.riverlist.map_to_grid(domain, rlon, rlat, rflow)

# Reintroduce buried phosphorus, weighted by river runoff
# (https://doi.org/10.5194/bg-10-8401-2013)
sim.request_redistribution(
    "det/burial", "runoff/source", runoff, datetime.timedelta(days=360)
)

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=1, interval_units=fabmos.TimeUnit.MONTHS, save_initial=False
)
out.request("runoff_source", *sim.fabm.default_outputs, time_average=True)

start = cftime.datetime(2000, 1, 1, calendar=calendar)
stop = cftime.datetime(2001, 1, 1, calendar=calendar)
sim.start(start, timestep=12 * 3600)
while sim.time < stop:
    sim.advance()
sim.finish()
