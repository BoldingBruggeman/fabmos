import os
import functools
import datetime
import logging
from typing import Optional, Tuple, Union, Mapping, Any, List

import numpy as np
import numpy.typing as npt
import scipy.sparse
import cftime
import xarray as xr
import h5py

import pygetm
import pygetm.parallel
from pygetm.constants import CENTERS
from .. import simulator, environment, Array
from ..domain import compress, freeze_vertical_coordinates

# Note: mpi4py components should be imported after pygetm.parallel,
# as the latter configures mpi4py.rc
from mpi4py import MPI
import mpi4py.util.dtlib

pygetm.input.TemporalInterpolation.MAX_CACHE_SIZE = 1024

# moles of gas in atmosphere = atmospheric mass (kg) divided
# by molecular weight of dry air (kg/mol)
ATMOSPHERIC_GAS_CONTENT = 5.15e18 / 28.93e-3


class MatArray(pygetm.input.LazyArray):
    def __init__(self, path: str, name: str):
        transpose = False
        try:
            # First try MATLAB 7.3 format (=HDF5)
            vardict = self.file = h5py.File(path)
        except Exception:
            # Now try older MATLAB formats
            vardict = scipy.io.loadmat(path, variable_names=(name,))
            transpose = True

        if name not in vardict:
            raise KeyError(
                f"Variable {name} not found in {path}."
                f" Available: {', '.join(vardict.keys())}"
            )

        self.var = vardict[name]
        if transpose:
            self.var = self.var.T

        name = f"MatArray({path!r}, {name!r})"
        super().__init__(self.var.shape, self.var.dtype, name)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.var, dtype=dtype)

    def __getitem__(self, slices) -> np.ndarray:
        return self.var[slices]


def get_mat_array(
    path: str,
    name: str,
    grid_file: str,
    times: Optional[npt.ArrayLike] = None,
    logger: logging.Logger = None,
) -> xr.DataArray:
    """This routine should ultimately read .mat files in HDF5
    as well as proprietary MATLAB formats"""
    return _wrap_ongrid_array(MatArray(path, name), grid_file, times)


def _wrap_ongrid_array(
    data: npt.ArrayLike, grid_file: str, times: Optional[npt.ArrayLike] = None
) -> xr.DataArray:
    x, y, dz = _load_mat(grid_file, "x", "y", "dz")
    dims = ["y", "x"]
    coords = {}
    coords["lon"] = xr.DataArray(x, dims=("x",), attrs=dict(units="degrees_east"))
    coords["lat"] = xr.DataArray(y, dims=("y",), attrs=dict(units="degrees_north"))
    assert data.shape[-1] == x.size
    assert data.shape[-2] == y.size
    if data.ndim > 2 and data.shape[-3] == dz.shape[0]:
        z_if = np.zeros((dz.shape[0] + 1,) + dz.shape[1:], dz.dtype)
        z_if[1:] = dz.cumsum(axis=0)
        coords["zc"] = xr.DataArray(
            0.5 * (z_if[:-1, ...] + z_if[1:, ...]),
            dims=("z", "y", "x"),
            attrs=dict(standard_name="depth"),
        )
        dims.insert(0, "z")
    if times is not None:
        assert len(times) == data.shape[0]
        dims.insert(0, "time")
        coords["time"] = times
    dims = [f"dim_{i}" for i in range(data.ndim - len(dims))] + dims
    return xr.DataArray(data, coords=coords, dims=dims)


class TransportMatrix(pygetm.input.LazyArray):
    def __init__(
        self,
        file_list: list,
        name: str,
        offsets: np.ndarray,
        counts: np.ndarray,
        rank: int,
        comm: MPI.Comm,
        order: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
        localize: bool = False,
    ):
        if logger:
            logger.info(f"Initializing {name} arrays")
        self.logger = logger
        self._file_list = file_list
        self._group_name = name

        # Obtain global sparse array indices and data type
        # Currently these are determined by reading the first global sparse matrix,
        # but in the future, they could might be read directly from file if the arrays
        # are stored (or have been transformed to) CSR format.
        self.global_indices = None
        self.global_indptr = None
        self.global_dtype = None
        if rank == 0:
            self._get_matrix_metadata(file_list[0], order=order)
        dtype = comm.bcast(self.global_dtype)
        indptr = comm.bcast(self.global_indptr)

        # Collect information for MPI scatter of sparse array indices and data
        # For this, we need arrays with sparse data offsets and counts for every rank
        self._comm = comm
        self._rank = rank
        self._counts = []
        self._offsets = []
        ioffset = 0
        for o, c in zip(offsets, counts):
            self._offsets.append(ioffset)
            self._counts.append(indptr[o + c] - indptr[o])
            ioffset += self._counts[-1]

        # Get local sparse array indices
        istartrow = offsets[rank]
        istoprow = istartrow + counts[rank]
        self.indptr = indptr[istartrow : istoprow + 1] - indptr[istartrow]
        self.indices = np.empty((self.indptr[-1],), indptr.dtype)
        mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.indices.dtype)
        comm.Scatterv(
            [self.global_indices, self._counts, self._offsets, mpi_dtype], self.indices
        )
        self.diag_indices = self._get_diagonal_indices(
            self.indptr, self.indices - istartrow
        )

        self.power = 1
        self.scale_factor = 1.0
        self.add_identity = False

        if localize:
            assert ((self.indices >= istartrow) & (self.indices < istoprow)).all()
            self.indices -= istartrow
            self.dense_shape = (counts[rank], counts[rank])
        else:
            self.dense_shape = (counts[rank], counts.sum())

        shape = (self.indices.size,)
        if len(file_list) > 1:
            shape = (len(file_list),) + shape
        super().__init__(shape, dtype.newbyteorder("="), name)

    def __getitem__(self, slices) -> np.ndarray:
        itime = 0 if len(self._file_list) == 1 else slices[0]
        assert isinstance(itime, (int, np.integer))
        if self._rank == 0:
            d = self._get_matrix_data(self._file_list[itime]).view(self.dtype)
        else:
            d = None
        values = np.empty((self.shape[-1],), self.dtype)
        self._comm.Scatterv([d, self._counts, self._offsets, MPI.DOUBLE], values)
        if self.scale_factor != 1.0:
            values *= self.scale_factor
        if self.add_identity:
            values[self.diag_indices] += 1.0
        if self.power != 1:
            # Matrix power for local block (current subdomain only)
            csr = self.create_sparse_array(values, scipy.sparse.csr_matrix)
            csr2 = csr**self.power
            csr2.sort_indices()
            assert (csr.indices == csr2.indices).all()
            assert (csr.indptr == csr2.indptr).all()
            values = csr2.data
        return values[slices[-1]]

    def __array__(self, dtype=None) -> np.ndarray:
        assert len(self._file_list) == 1
        return self[(slice(None),)]

    def _get_diagonal_indices(
        self, indptr: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        diag_indices = np.empty((indptr.size - 1,), dtype=indptr.dtype)
        for irow, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:])):
            idiag = (indices[start:stop] == irow).nonzero()[0]
            assert idiag.size == 1
            diag_indices[irow] = start + idiag
        return diag_indices

    def _get_matrix_metadata(self, fn: str, order: Optional[np.ndarray] = None):
        try:
            # MATLAB >= 7.3 format (HDF5)
            with h5py.File(fn) as ds:
                group = ds[self._group_name]
                A_ir = np.asarray(group["ir"], dtype=np.intp)
                A_jc = np.asarray(group["jc"], dtype=np.intp)
                dtype = group["data"].dtype
        except OSError:
            # MATLAB < 7.3 format
            vardict = scipy.io.loadmat(fn, variable_names=(self._group_name,))
            A = vardict[self._group_name]
            A_ir = A.indices
            A_jc = A.indptr
            dtype = A.dtype
        n = A_jc.size - 1

        old2newindex = np.arange(n)
        if order is not None:
            assert order.shape == (n,)
            old2newindex[order] = np.arange(n)
            A_ir[:] = old2newindex[A_ir]

        # Determine CSC to CSR mapping
        items_per_row = np.zeros((n,), dtype=int)
        np.add.at(items_per_row, A_ir, 1)

        self.index_map = np.empty_like(A_ir)
        indices = np.empty_like(A_ir)
        indptr = np.empty_like(A_jc)
        indptr[0] = 0
        indptr[1:] = items_per_row.cumsum()
        current_items_per_row = np.zeros_like(items_per_row)
        for icol, (colstart, colstop) in enumerate(zip(A_jc[:-1], A_jc[1:])):
            irows = A_ir[colstart:colstop]
            newi = indptr[irows] + current_items_per_row[irows]
            indices[newi] = old2newindex[icol]
            self.index_map[newi] = np.arange(colstart, colstop)
            current_items_per_row[irows] += 1
        self.global_indices = indices
        self.global_indptr = indptr
        self.global_dtype = dtype
        if False:
            assert (current_items_per_row == (indptr[1:] - indptr[:-1])).all()
            A_data = np.random.random(indices.shape)
            csr = scipy.sparse.csr_array((A_data[self.index_map], indices, indptr))
            csr2 = scipy.sparse.csc_array((A_data, A_ir, A_jc), (n, n)).tocsr()
            assert csr.nnz == csr2.nnz
            assert csr.shape == csr2.shape
            assert (csr.indices == csr2.indices).all()
            assert (csr.indptr == csr2.indptr).all()
            assert (csr.data == csr2.data).all()

    def _get_matrix_data(self, fn: str) -> np.ndarray:
        try:
            # MATLAB >= 7.3 format (HDF5)
            with h5py.File(fn) as ds:
                data = np.asarray(ds[self._group_name]["data"])
        except OSError:
            # MATLAB < 7.3 format
            vardict = scipy.io.loadmat(fn, variable_names=(self._group_name,))
            data = vardict[self._group_name].data
        return data[self.index_map]

    def create_sparse_array(
        self, data: Optional[np.ndarray] = None, type=scipy.sparse.csr_array
    ) -> scipy.sparse.csr_array:
        if data is None:
            data = np.empty((self.shape[-1],), self.dtype)
        return type(
            (data, self.indices, self.indptr),
            self.dense_shape,
        )


def _read_grid(fn: str):
    if not os.path.isfile(fn):
        raise Exception(f"Grid file {fn} does not exist")
    with h5py.File(fn) as ds:
        bathy = np.asarray(ds["bathy"], dtype=int)
        ideep = np.asarray(ds["ideep"], dtype=int)
        x = np.asarray(ds["x"])
        y = np.asarray(ds["y"])
        z = np.asarray(ds["z"])
        dz = np.asarray(ds["dz"])
        da = np.asarray(ds["da"])
        delta_t = np.asarray(ds["deltaT"][0, 0])
    return bathy, ideep, x, y, z, dz, da, delta_t


def _read_config(fn: str) -> Mapping[str, Any]:
    if not os.path.isfile(fn):
        raise Exception(f"Configuration file {fn} does not exist")
    config = {}
    try:
        # MATLAB >= 7.3 format (HDF5)
        with h5py.File(fn) as ds:
            for k in ds.keys():
                v = np.squeeze(ds[k][...])
                if v.dtype == "<u2":
                    v = bytes(v[:]).decode("utf16")
                config[k] = v
    except OSError:
        # MATLAB < 7.3 format
        for k, v in scipy.io.loadmat(fn).items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, "U"):
                v = str(v[0])
            config[k] = v
    for k in ("fixEmP", "rescaleForcing", "useAreaWeighting"):
        config[k] = (config[k] != 0).any()
    return config


def _load_mat(
    fn: str,
    *names: str,
    dtype: Optional[npt.DTypeLike] = None,
    slc: Tuple = (Ellipsis,),
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    if not os.path.isfile(fn):
        raise Exception(f"File {fn} does not exist")

    values = []
    try:
        # MATLAB >= 7.3 format (HDF5)
        with h5py.File(fn) as ds:
            for name in names:
                values.append(np.asarray(ds[name][slc], dtype=dtype))
    except OSError:
        # MATLAB < 7.3 format
        name2values = scipy.io.loadmat(fn, variable_names=names)
        for name in names:
            values.append(name2values[name].T[slc])

    def _process(data):
        while data.ndim and data.shape[0] == 1:
            data = data[0, ...]
        return data

    values = [_process(v) for v in values]

    return values[0] if len(names) == 1 else tuple(values)


def create_domain(
    path: str, logger: Optional[logging.Logger] = None, comm: Optional[MPI.Comm] = None
) -> pygetm.domain.Domain:
    if os.path.isdir(path):
        path = os.path.join(path, "grid.mat")

    logger = logger or pygetm.parallel.get_logger()
    logger.info(f"Reading grid information from: {path}")
    bathy, ideep, lon, lat, z, dz, da, delta_t = _read_grid(path)

    # If x,y coordinates are effectively 1D, transpose latitude to ensure
    # its singleton dimension comes last (x)
    if lon.shape[0] == 1 and lat.shape[0] == 1:
        lat = lat.T

    domain = pygetm.domain.create_spherical(
        lon, lat, mask=np.where(ideep, 1, 0), H=dz.sum(axis=0), logger=logger
    )
    domain._grid_file = path
    domain._delta_t = delta_t
    domain._order = None
    domain._nz = dz.shape[0]
    if domain.comm.rank == 0:
        domain._h = dz
        domain._area[1::2, 1::2] = da[0, :, :]

        wet = np.arange(1, domain._nz + 1)[:, np.newaxis, np.newaxis] <= ideep
        domain._mask3d = np.where(wet, 1, 0)

        wet_glob = np.arange(1, domain._nz + 1) <= ideep[:, :, np.newaxis]  # y, x, z
        wet_glob_tmm_order = np.moveaxis(wet_glob, 2, 0)  # z, y, x
        nwet_glob = wet_glob.sum()
        order = np.empty_like(wet_glob, dtype=int)
        order_tmm = np.moveaxis(order, 2, 0)
        order_tmm[wet_glob_tmm_order] = np.arange(nwet_glob)
        order = order[wet_glob]
        assert order.shape == (nwet_glob,)
        assert (np.sort(order) == np.arange(nwet_glob)).all()
        domain._order = order
    return domain


def climatology_times(calendar="standard"):
    return [
        cftime.datetime(2000, imonth + 1, 16, calendar=calendar) for imonth in range(12)
    ]


class Simulator(simulator.Simulator):
    def __init__(
        self,
        full_domain: pygetm.domain.Domain,
        periodic_matrix: bool = True,
        periodic_forcing: bool = True,
        calendar: str = "standard",
        fabm: Union[str, pygetm.fabm.FABM] = "fabm.yaml",
        log_level: Optional[int] = None,
    ):
        fabm_libname = os.path.join(os.path.dirname(__file__), "..", "fabm_tmm")

        domain = compress(full_domain, extra_fields=("_mask3d", "_h"))

        super().__init__(
            domain,
            nz=full_domain._nz,
            mask3d=domain._mask3d,
            fabm=fabm,
            fabm_libname=fabm_libname,
            log_level=log_level,
            use_virtual_flux=True,
        )

        slc_loc, slc_glob, _, _ = self.tiling.subdomain2slices()

        self.T.hn.scatter(domain._h)
        freeze_vertical_coordinates(self.T, self.depth, h=self.T.hn.values[slc_loc])

        self.tmm_logger = self.logger.getChild("TMM")

        # Retrieve extra TMM attributes from full domain
        order = full_domain.comm.bcast(full_domain._order)
        self._grid_file = full_domain._grid_file
        self._delta_t = full_domain._delta_t

        # 3D Boolean array for mapping between native and TMM layout
        self.wet_loc = self.T.mask3d.values[:, 0, :].T != 0

        self.nwet = (self.T.mask3d.values != 0).sum()
        self.counts = np.empty(self.tiling.n, dtype=int)
        self.offsets = np.zeros_like(self.counts)
        self.tiling.comm.Allgather(self.nwet, self.counts)
        nwets_cum = self.counts.cumsum()
        self.offsets[1:] = nwets_cum[:-1]
        self.nwet_tot = nwets_cum[-1]
        offset = self.offsets[self.tiling.rank]
        self.tmm_slice = slice(offset, offset + self.nwet)

        root = os.path.dirname(full_domain._grid_file)
        config = _read_config(os.path.join(root, "config_data.mat"))
        Ir_pre = _load_mat(
            os.path.join(root, config["matrixPath"], "Data/profile_data.mat"),
            "Ir_pre",
            dtype=int,
        )
        assert (Ir_pre - 1 == order).all()

        Aexp_files = []
        Aimp_files = []
        if periodic_matrix:
            matrix_times = climatology_times(calendar)
            exp_base = config["explicitMatrixFileBase"]
            imp_base = config["implicitMatrixFileBase"]
            for imonth in range(1, 13):
                Aexp_files.append(os.path.join(root, exp_base + f"_{imonth:02}.mat"))
                Aimp_files.append(os.path.join(root, imp_base + f"_{imonth:02}.mat"))
            Aexp_name = "Aexp"
            Aimp_name = "Aimp"
        else:
            matrix_times = None
            exp_base = config["explicitAnnualMeanMatrixFile"]
            imp_base = config["implicitAnnualMeanMatrixFile"]
            Aexp_files.append(os.path.join(root, exp_base + ".mat"))
            Aimp_files.append(os.path.join(root, imp_base + ".mat"))
            Aexp_name = "Aexpms"
            Aimp_name = "Aimpms"

        self.Aexp_src = TransportMatrix(
            Aexp_files,
            Aexp_name,
            self.offsets,
            self.counts,
            self.tiling.rank,
            self.tiling.comm,
            logger=self.tmm_logger,
            order=order,
        )
        self.Aimp_src = TransportMatrix(
            Aimp_files,
            Aimp_name,
            self.offsets,
            self.counts,
            self.tiling.rank,
            self.tiling.comm,
            logger=self.tmm_logger,
            order=order,
            localize=True,
        )
        self.Aexp = self.Aexp_src.create_sparse_array()
        self.Aimp = self.Aimp_src.create_sparse_array()
        if self.Aexp_src.ndim == 2:
            self.Aexp_tip = pygetm.input.TemporalInterpolation(
                self.Aexp_src,
                0,
                matrix_times,
                climatology=True,
                comm=self.tiling.comm,
                logger=self.tmm_logger,
            )
            self.input_manager._all_fields.append(
                (self.Aexp_tip.name, self.Aexp_tip, self.Aexp.data)
            )
        if self.Aimp_src.ndim == 2:
            self.Aimp_tip = pygetm.input.TemporalInterpolation(
                self.Aimp_src,
                0,
                matrix_times,
                climatology=True,
                comm=self.tiling.comm,
                logger=self.tmm_logger,
            )
            self.input_manager._all_fields.append(
                (self.Aimp_tip.name, self.Aimp_tip, self.Aimp.data)
            )

        self.load_environment(periodic_forcing, calendar)
        self._redistribute: List[
            Tuple[
                Array,
                Array,
                Array,
                Optional[Array],
                Union[datetime.timedelta, int],
                int,
            ]
        ] = []
        self._atmospheric_gases: List[Tuple[Array, Array, float, float]] = []

        self.tot_area = self.T.area.global_sum(where=self.T.mask == 1, to_all=True)

    def _get_existing_fabm_variable(self, name: str) -> Array:
        for v, array in self.fabm._variable2array.items():
            if v.name == name:
                break
        else:
            names = [v.name for v in self.fabm._variable2array]
            raise Exception(
                f"Variable {name} not found in FABM model."
                f" Available: {', '.join(names)}"
            )
        array.saved = True
        return array

    def request_redistribution(
        self,
        source: str,
        target: str,
        weights: Optional[Array],
        update_interval: Union[datetime.timedelta, int],
        initial_value: float = 0.0,
    ):
        """Distribute the whole-domain-integral of a 2D (horizontal-only) variable from
        a FABM model to another 2D FABM variable (typically a horizontal dependency).

        This is typically used to re-insert a permanent loss term such as burial in
        sediments back into the model, often as a surface flux representing rivers or
        atmopsheric depositon.

        Args:
            source: name of the FABM variable to integrate and distribute
            target: name of the FABM variable to distribute the integral to
            weights: weights per cell to use when redistributing. Weighting by cell
                area will be done afterwards, so if weights are equal across the
                domain, each cell will receive the same value per unit area.
            update_interval: how often to update the redistributed flux. In between
                updates, the horizontally-integrated source will be accumulated.
            initial_value: initial value for the redistributed flux, to be used
                until the first update_interval is complete.
        """
        source_array = self._get_existing_fabm_variable(source)
        target_array = self.fabm.get_dependency(target)
        target_array.fill(initial_value)
        cum_source = source_array.grid.array(fill=0.0)
        iupdate = update_interval if isinstance(update_interval, int) else None
        self._redistribute.append(
            [source_array, target_array, cum_source, weights, update_interval, iupdate]
        )

    def add_atmospheric_gas(
        self,
        atmospheric_name: str,
        underwater_name: str,
        atmospheric_scale_factor: float = 1e-6,
        underwater_scale_factor: float = 1.0,
    ) -> Array:
        atm_array = self.fabm.get_dependency(atmospheric_name)
        atm_array.attrs["_part_of_state"] = True
        flux_array = self._get_existing_fabm_variable(underwater_name + "_sfl")
        self.logger.info(
            f"Atmospheric gas: {atm_array.name} will be updated"
            f" based on air-sea flux of {underwater_name}."
        )
        self._atmospheric_gases.append(
            [
                atm_array,
                flux_array,
                atmospheric_scale_factor,
                underwater_scale_factor,
            ]
        )
        return atm_array

    def start(
        self,
        time: Union[cftime.datetime, datetime.datetime],
        timestep: Union[float, datetime.timedelta],
        transport_timestep: Optional[Union[float, datetime.timedelta]] = None,
        report: datetime.timedelta = datetime.timedelta(days=1),
        report_totals: Union[int, datetime.timedelta] = datetime.timedelta(days=10),
        profile: Optional[str] = None,
    ):
        if isinstance(timestep, datetime.timedelta):
            timestep = timestep.total_seconds()
        if isinstance(transport_timestep, datetime.timedelta):
            transport_timestep = transport_timestep.total_seconds()

        dt = transport_timestep or timestep
        nphys = dt / self._delta_t
        self.tmm_logger.info(
            f"Transport timestep of {dt} s is {nphys} *"
            f" the original online timestep of {self._delta_t} s."
        )
        if nphys % 1.0 > 1e-8:
            raise Exception(
                f"The transport timestep of {dt} s must be an exact multiple"
                f" of the original online timestep of {self._delta_t} s"
            )

        self.Aexp_src.scale_factor = dt
        self.Aexp_src.add_identity = True
        self.Aimp_src.power = int(round(nphys))
        if self.Aexp_src.ndim == 1:
            self.Aexp.data[:] = self.Aexp_src
        else:
            self.Aexp_tip._cache.clear()
        if self.Aimp_src.ndim == 1:
            self.Aimp.data[:] = self.Aimp_src
        else:
            self.Aimp_tip._cache.clear()

        super().start(
            time, timestep, transport_timestep, report, report_totals, profile
        )

        ntracer = len(self.tracers)
        tracer_dtype = self.T.H.dtype
        self.global_tracers = np.empty((self.nwet_tot, ntracer), tracer_dtype)
        self.local_tracers = self.global_tracers[self.tmm_slice, :]
        for i, tracer in enumerate(self.tracers):
            self.local_tracers[:, i] = tracer.values[:, 0, :].T[self.wet_loc]
        recvbuf = (
            self.global_tracers,
            self.counts * ntracer,
            self.offsets * ntracer,
            MPI.DOUBLE,
        )
        self.gather_tracer = functools.partial(
            self.tiling.comm.Iallgatherv, MPI.IN_PLACE, recvbuf
        )
        self.gather_req = self.gather_tracer()

        for info in self._redistribute:
            interval = info[-2]
            if isinstance(interval, datetime.timedelta):
                info[-1] = max(1, int(round(interval.total_seconds() / timestep)))
            self.logger.info(
                f"Flux redistribution: {info[1].name} will be updated "
                f"every {info[-1]} steps."
            )

    def advance_fabm(self, timestep: float):
        for source, target, cum_source, w, _, freq in self._redistribute:
            cum_source.all_values += source.all_values
            if self.istep % freq == 0:
                self.logger.info(
                    f"Deriving {target.name} from horizontally integrated {source.name}"
                )
                np.divide(cum_source.all_values, freq, out=target.all_values)
                target.all_values *= target.grid.area.all_values
                unmasked = target.grid.mask == 1
                int_flux = target.global_sum(where=unmasked, to_all=True)
                if w is not None:
                    w_int = (w * w.grid.area).global_sum(where=unmasked, to_all=True)
                    scale_factor = int_flux / w_int
                    self.logger.info(
                        f"  global scale factor for {target.name} "
                        f"= {scale_factor} {source.units} ({w.units})-1"
                    )
                    values = w.values * scale_factor
                else:
                    values = int_flux / self.tot_area
                    self.logger.info(
                        f"  global mean {source.name} = {values} {source.units}"
                    )
                target.fill(values)
                cum_source.all_values.fill(0.0)

        for atm_array, flux_array, atm_scale, wat_scale in self._atmospheric_gases:
            unmasked = flux_array.grid.mask == 1
            flux_area = flux_array * flux_array.grid.area
            int_flux = flux_area.global_sum(where=unmasked, to_all=True)
            content = atm_array.values[0, 0] * atm_scale * ATMOSPHERIC_GAS_CONTENT
            content -= int_flux * wat_scale * timestep
            volume_fraction = content / (atm_scale * ATMOSPHERIC_GAS_CONTENT)
            self.logger.debug(
                f"  integrated flux of {atm_array.name} = {int_flux} mol/s."
                f" Updated volume fraction = {volume_fraction}"
            )
            atm_array.fill(volume_fraction)

        return super().advance_fabm(timestep)

    def transport(self, timestep: float):
        # Compute the change in tracers due to biogeochemical [FABM] processes
        # This is done by collecting the updated compressed state
        # and subtracting the previous compressed state
        fabm_tracer_change = np.empty_like(self.local_tracers)
        for i, tracer in enumerate(self.tracers):
            fabm_tracer_change[:, i] = tracer.values[:, 0, :].T[self.wet_loc]
        fabm_tracer_change -= self.local_tracers

        # Ensure the compressed global tracer state is synchronized across subdomains
        MPI.Request.Wait(self.gather_req)

        # Do the explicit step
        new_local_tracers = self.Aexp.dot(self.global_tracers)

        # Add the biogeochemical/FABM tracer increment
        new_local_tracers += fabm_tracer_change

        # Do the implicit step
        self.local_tracers[:, :] = self.Aimp.dot(new_local_tracers)

        # Start synchronizing the compressed tracer state across subdomains
        self.gather_req = self.gather_tracer()

        # Copy the updated compressed state back to the uncompressed state
        for i, tracer in enumerate(self.tracers):
            tracer.values[:, 0, :].T[self.wet_loc] = self.local_tracers[:, i]

    def load_environment(self, periodic: bool, calendar: str):
        root = os.path.dirname(self._grid_file)

        times = climatology_times(calendar)

        def _get_variable(
            path: str, varname: str, **kwargs
        ) -> Tuple[Array, xr.DataArray]:
            array = self.T.array(**kwargs)
            src = get_mat_array(path, varname, self._grid_file, times=times)
            _set(array, src)
            return array, src

        def _set(array, src):
            if not periodic:
                src = src.mean(dim="time")
                array.attrs["_time_varying"] = False
            array.set(src, on_grid=pygetm.input.OnGrid.ALL, climatology=periodic)

        self.temp, temp_src = _get_variable(
            os.path.join(root, "GCM/Theta_gcm.mat"),
            "Tgcm",
            z=CENTERS,
            name="temp",
            units="degrees_Celsius",
            long_name="temperature",
            fabm_standard_name="temperature",
        )
        self.salt, salt_src = _get_variable(
            os.path.join(root, "GCM/Salt_gcm.mat"),
            "Sgcm",
            z=CENTERS,
            name="salt",
            units="PSU",
            long_name="salinity",
            fabm_standard_name="practical_salinity",
        )
        if self.fabm.has_dependency("density"):
            self.rho = self.T.array(
                z=CENTERS,
                name="rho",
                units="kg m-3",
                long_name="density",
                fabm_standard_name="density",
            )
            self.rho.set(
                environment.density(salt_src, temp_src),
                on_grid=pygetm.input.OnGrid.ALL,
                climatology=periodic,
            )

        self.wind, _ = _get_variable(
            os.path.join(root, "BiogeochemData/wind_speed.mat"),
            "windspeed",
            name="wind",
            units="m s-1",
            long_name="wind speed",
            fabm_standard_name="wind_speed",
        )
        self.fice, _ = _get_variable(
            os.path.join(root, "BiogeochemData/ice_fraction.mat"),
            "Fice",
            name="ice",
            units="1",
            long_name="ice cover",
            fabm_standard_name="ice_area_fraction",
        )

        fwf_path = os.path.join(root, "GCM/FreshWaterForcing_gcm.mat")
        tau = _load_mat(fwf_path, "saltRelaxTimegcm")
        da = _load_mat(self._grid_file, "da", slc=(0, Ellipsis))
        unmasked = da != 0.0
        if np.any(tau):
            self.logger.info(
                "Reconstructing net freshwater flux from salinity relaxation"
                " in original hydrodynamic simulation"
            )
            assert np.ndim(tau) == 0
            Srelax_src = MatArray(fwf_path, "Srelaxgcm")
            dz = _load_mat(self._grid_file, "dz", slc=(0, Ellipsis))
            salt_sf_src = pygetm.input.isel(salt_src, z=0)

            # The change in surface salinity due to relaxation is
            #   (S_relax - S) / tau (# m-3 s-1)
            # Multiply with top layer thickness dz to obtain a surface flux (# m-2 s-1)
            # This must equal the virtual salt flux of -S * pe
            # Thus, pe = (1 - S_relax / S) * dz / tau
            pe_src = (1.0 - Srelax_src / salt_sf_src) * (dz / tau)
        else:
            self.logger.info(
                "Using net freshwater flux from original hydrodynamic simulation"
            )
            pe_src = -MatArray(fwf_path, "EmPgcm")
        intpe = (np.mean(np.asarray(pe_src), axis=0) * da).sum(where=unmasked)
        offset = np.array(-intpe / da.sum())
        self.logger.info(
            f"Adjusting net freshwater flux by {offset:.4} m s-1 to close"
            f" global freshwater budget"
        )
        pe_src = pe_src + offset
        pe_src = _wrap_ongrid_array(pe_src, self._grid_file, times=times)
        _set(self.pe, pe_src)
