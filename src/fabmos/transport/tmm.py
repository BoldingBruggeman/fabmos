import sys
import xarray as xr
import numpy as np
from mpi4py import MPI
import h5py

from typing import (
    MutableMapping,
    Optional,
    Tuple,
    Iterable,
    Union,
    Callable,
    Any,
    Mapping,
)

import pygetm
from .. import simulator

_comm = MPI.COMM_WORLD.Dup()
size = _comm.Get_size()
rank = _comm.Get_rank()

verbose = True
use_root = True

def _read_grid(fn):
    ds = xr.open_dataset(
        fn,
        engine="h5netcdf",
        phony_dims="sort",
    )

    bathy = np.asarray(ds["bathy"], dtype=int)
    ideep = np.asarray(ds["ideep"], dtype=int)
    x = np.asarray(ds["x"])
    y = np.asarray(ds["y"])
    z = np.asarray(ds["z"])

    return bathy, ideep, x, y, z


def _profile_indices(
    ideep: np.ndarray,
    verbose=False,
) -> Tuple[int, np.ndarray, np.ndarray]:
    if rank == 0:
        ie = ideep.ravel().cumsum().reshape(ideep.shape)
        ie = np.ma.array(ie, mask=ideep == 0).compressed()
        nprof = np.asarray(len(ie), dtype=int)
        ib = np.empty(nprof, dtype=int)
        ib[0] = 1
        ib[1:] = ie[:-1] + 1
    else:
        ib = None
        ie = None
        nprof = np.empty(1, dtype=int)

    if rank == 0 and verbose:
        print("ib:  ", nprof, ib)
        print("ie:  ", nprof, ie)

    _comm.Bcast([nprof, MPI.INT], root=0)

    return int(nprof), ib, ie


def _mapping_arrays(
    nprof: int, ib: np.ndarray, ie: np.ndarray, verbose=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rank == 0:
        rows = np.empty(nprof + 1, dtype="i")
        rows[0] = 0
        rows[1:] = np.cumsum(ie - ib + 1)
        if verbose:
            print("rows:", len(rows), rows)

        indx = np.empty(size + 1, dtype=int)
        if use_root:
            k = int(np.ceil(nprof / size))
            indx[0] = 0
            indx[1:] = k * (np.arange(size) + 1)
        else:
            k = int(np.ceil(nprof / (size - 1)))
            indx[0:2] = 0
            indx[2:] = k * (np.arange(size - 1) + 1)
        # indx[0] = 0
        # indx[1:] = k * (np.arange(size) + 1)
        indx[size] = min(k * size, nprof)
        nlp = np.asarray(indx[1:] - indx[:-1], dtype=int)
        counts = np.asarray(
            [(rows[indx[i + 1]] - rows[indx[i]]) for i in range(size)], dtype=int
        )

        o = np.empty(size, dtype=int)
        o[0] = 0
        o[1:] = np.cumsum(counts[:-1])
        offsets = np.append(o, ie[-1])
    else:
        rows = None
        indx = np.empty(size + 1, dtype=int)
        nlp = np.empty(size, dtype=int)
        counts = np.empty(size, dtype=int)
        offsets = np.empty(size + 1, dtype=int)

    _comm.Bcast((nlp, MPI.INTEGER), root=0)
    _comm.Bcast((indx, MPI.INTEGER), root=0)
    _comm.Bcast((counts, MPI.INTEGER), root=0)
    _comm.Bcast((offsets, MPI.INTEGER), root=0)

    if rank == 0 and verbose:
        print("nlp:     ", np.sum(nlp), nlp)
        print("indx:    ", len(indx), indx)
        print("counts:  ", np.sum(counts), counts)
        print("offsets: ", len(offsets), offsets)

    return nlp, indx, counts, offsets


def create_domain(path: str) -> pygetm.domain.Domain:
    # open files, read 2D lon, lat, mask (pygetm convention, 0=land, 1=water), H
    # squeeze().T

    if rank == 0:
        bathy, ideep, x, y, z = _read_grid(path)
        print("shapes: bathy, x, y: ",bathy.shape, x.shape, y.shape)

        # find depth variable - for now
        H = np.ones(bathy.shape)
    else:
        bathy = ideep = x = y = z = None
    nprof, ib, ie = _profile_indices(ideep, verbose=verbose)
    nlp, indx, counts, offsets = _mapping_arrays(nprof, ib, ie, verbose=verbose)
    sys.exit()

    # Jorn
    # Here is what I've done - using counts, and offsets variables:
    # Similar code for other exchanges
    if False:
        if rank == 0:
            a = np.empty(ncol)
            for i in range(size):
                a[offsets[i] : offsets[i + 1]] = i + 1
        else:
            a = None

        # Send portion of global array - a - to all
        b = np.empty(offsets[rank + 1] - offsets[rank])
        sendbuf = [a, counts, offsets[0:-1], MPI.DOUBLE]
        _comm.Scatterv(sendbuf, b)

        b[:] = b[:] + 1

        # root receives local arrays and store in global
        recvbuf = [a, counts, offsets[0:-1], MPI.DOUBLE]
        _comm.Gatherv(b, recvbuf)



    lon = np.arange(0.0, 10.0, 0.5)
    lat = np.arange(10.0, 40.0, 0.5)
    return pygetm.domain.create_spherical(lon=lon, lat=lat, mask=1, H=H, nz=15)


class Simulator(simulator.Simulator):
    def __init__(self, domain: pygetm.domain.Domain, fabm_config: str = "fabm.yaml"):
        super().__init__(domain, fabm_config)
        self.tmm_logger = self.logger.getChild("tmm")

    def transport(self, timestep: float):
        for tracer in self.tracers:
            print(tracer.name)
            # update tracer.all_values in place
