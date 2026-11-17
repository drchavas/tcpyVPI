"""
ERA5 data loader for the tcpyVPI package.

This module loads ERA5 reanalysis data remotely, over OPeNDAP, from NCAR's
GDEX THREDDS server (formerly RDA): the monthly mean dataset d633001 and the
hourly dataset d633000.

    monthly: {base}/d633001/e5.moda.an.{sfc|pl}/{year}/<file>
    hourly:  {base}/d633000/e5.oper.an.{sfc|pl}/{yyyymm}/<file>

with ``base = https://tds.gdex.ucar.edu/thredds/dodsC/files/g`` by default.

The hourly dataset has different file structures for surface vs pressure level data:
- Surface variables: monthly files containing all hours (YYYYMM0100_YYYYMM{last_day}23)
- Pressure level variables: daily files containing 24 hours (YYYYMMDD00_YYYYMMDD23)

Server location
---------------
NCAR has moved these files before (the 2026 RDA -> GDEX migration renamed the
monthly folder from ``d633001_nc`` to ``d633001``). So that a future move does
not break the package, the location is not hardcoded:

* Each file is tried at a short list of known locations, in order, and the
  first that works is remembered for the rest of the session. Transient network
  errors are retried; "file not found" moves straight on to the next location.
* The location can be overridden with :func:`set_era5_thredds_base` or the
  environment variable ``TCPYVPI_ERA5_THREDDS``; an override is then the only
  location tried.
* If every location fails, :class:`ERA5DataUnavailableError` lists each URL
  tried and how to override the location.

Dataset pages: https://gdex.ucar.edu/datasets/d633001/ (monthly) and
https://gdex.ucar.edu/datasets/d633000/ (hourly).

Author: Jose Ocegueda Sanchez (2025)
Based on original monthly data loader by Dan Chavas (2025)
"""

import calendar
import os
import re
import time
from typing import Optional, Literal, Union, Callable, List, Tuple
from datetime import datetime

import xarray as xr
import numpy as np

# ============================================================================
# THREDDS server location
# ============================================================================
GDEX_THREDDS_BASE = "https://tds.gdex.ucar.edu/thredds/dodsC/files/g"
RDA_THREDDS_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g"   # alias of GDEX since 2026
DEFAULT_THREDDS_BASE = GDEX_THREDDS_BASE

MONTHLY_FOLDER = "d633001"     # monthly means; was "d633001_nc" before the GDEX migration
HOURLY_FOLDER = "d633000"      # hourly

ERA5_THREDDS_ENV_VAR = "TCPYVPI_ERA5_THREDDS"

GDEX_MONTHLY_PAGE = "https://gdex.ucar.edu/datasets/d633001/"
GDEX_HOURLY_PAGE = "https://gdex.ucar.edu/datasets/d633000/"

# Known locations, tried in order: (base, dataset folder).
_DEFAULT_CANDIDATES = {
    'monthly': [
        (GDEX_THREDDS_BASE, MONTHLY_FOLDER),
        (RDA_THREDDS_BASE, MONTHLY_FOLDER),
        (RDA_THREDDS_BASE, "d633001_nc"),
    ],
    'hourly': [
        (GDEX_THREDDS_BASE, HOURLY_FOLDER),
        (RDA_THREDDS_BASE, HOURLY_FOLDER),
    ],
}

# Retry policy for transient failures at one location.
MAX_ATTEMPTS = 3               # attempts per location
BACKOFF_SECONDS = (2.0, 5.0)   # waits before the 2nd and 3rd attempt

# Session state: an explicit override (set_era5_thredds_base) and the location
# that last worked for each dataset.
_override = None               # (base, monthly_folder, hourly_folder) or None
_working = {}                  # {'monthly': (base, folder), 'hourly': (base, folder)}

# Indirection so tests can substitute the opener and the clock.
_open_dataset = xr.open_dataset
_sleep = time.sleep


class ERA5DataUnavailableError(OSError):
    """Raised when an ERA5 file cannot be opened at any candidate location."""


def set_era5_thredds_base(
    url: Optional[str] = None,
    monthly_folder: str = MONTHLY_FOLDER,
    hourly_folder: str = HOURLY_FOLDER,
) -> None:
    """Point the ERA5 loader at a different THREDDS server.

    Use this if NCAR moves the data again. While set, it is the only location
    tried. Call with no argument to go back to the defaults (the environment
    variable ``TCPYVPI_ERA5_THREDDS`` if set, otherwise the built-in list).

    Parameters
    ----------
    url : str, optional
        OPeNDAP base that contains the dataset folders, e.g.
        ``"https://tds.gdex.ucar.edu/thredds/dodsC/files/g"``.
    monthly_folder, hourly_folder : str
        Dataset folder names under ``url``. Defaults ``"d633001"`` and
        ``"d633000"``.

    Examples
    --------
    >>> import tcpyVPI
    >>> tcpyVPI.set_era5_thredds_base("https://new.server.ucar.edu/thredds/dodsC/files/g")
    >>> tcpyVPI.set_era5_thredds_base()      # back to the defaults
    """
    global _override
    _override = None if url is None else (url.rstrip('/'), monthly_folder, hourly_folder)
    _working.clear()


def get_era5_thredds_candidates(kind: Literal['monthly', 'hourly'] = 'monthly') -> List[Tuple[str, str]]:
    """Return the (base, folder) locations that will be tried, in order.

    An override from :func:`set_era5_thredds_base` takes precedence, then the
    environment variable ``TCPYVPI_ERA5_THREDDS``, then the built-in list. A
    location that has already worked this session is moved to the front.
    """
    if kind not in _DEFAULT_CANDIDATES:
        raise ValueError(f"kind must be 'monthly' or 'hourly', not {kind!r}")
    if _override is not None:
        base, mfolder, hfolder = _override
        return [(base, mfolder if kind == 'monthly' else hfolder)]
    env = os.environ.get(ERA5_THREDDS_ENV_VAR, '').strip()
    if env:
        return [(env.rstrip('/'), MONTHLY_FOLDER if kind == 'monthly' else HOURLY_FOLDER)]
    cands = list(_DEFAULT_CANDIDATES[kind])
    if kind in _working and _working[kind] in cands:
        cands.remove(_working[kind])
        cands.insert(0, _working[kind])
    return cands


def _default_location(kind):
    return get_era5_thredds_candidates(kind)[0]


# ============================================================================
# Variable Mapping Tables
# ============================================================================
# Maps variable names to ERA5 file codes and suffixes
# Format: {var_name: (code, suffix)}

SURFACE_VARS = {
    'SSTK': ('034_sstk', 'sc'),   # Sea surface temperature
    'SP': ('134_sp', 'sc'),       # Surface pressure
    'U10': ('165_10u', 'sc'),     # 10m U wind component
    'V10': ('166_10v', 'sc'),     # 10m V wind component
    'T2M': ('167_2t', 'sc'),      # 2m temperature
    'D2M': ('168_2d', 'sc'),      # 2m dewpoint temperature
}

PRESSURE_VARS = {
    'T': ('130_t', 'sc'),         # Temperature
    'U': ('131_u', 'uv'),         # U component of wind
    'V': ('132_v', 'uv'),         # V component of wind
    'Q': ('133_q', 'sc'),         # Specific humidity
    'VO': ('138_vo', 'sc'),       # Relative vorticity
}


# ============================================================================
# URL Construction Functions
# ============================================================================
# Each builder takes the server base and dataset folder as arguments. If they
# are omitted, the first candidate location is used.

def _get_monthly_mean_url(var_name: str, year: int, is_surface: bool = True,
                          base: Optional[str] = None, folder: Optional[str] = None) -> str:
    """
    Construct THREDDS URL for ERA5 monthly mean data.

    Monthly mean files contain all 12 months of a year in a single file.

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'SSTK', 'T', 'U')
    year : int
        Year of data
    is_surface : bool
        True for surface variables, False for pressure level variables
    base, folder : str, optional
        Server base and dataset folder; default to the first candidate location.

    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    if base is None or folder is None:
        b, f = _default_location('monthly')
        base, folder = base or b, folder or f
    level = 'sfc' if is_surface else 'pl'
    code, suffix = SURFACE_VARS[var_name] if is_surface else PRESSURE_VARS[var_name]
    filename = f"e5.moda.an.{level}.128_{code}.ll025{suffix}.{year}010100_{year}120100.nc"
    return f"{base}/{folder}/e5.moda.an.{level}/{year}/{filename}"


def _get_hourly_surface_url(var_name: str, year: int, month: int,
                            base: Optional[str] = None, folder: Optional[str] = None) -> str:
    """
    Construct THREDDS URL for ERA5 hourly surface data.

    Surface files contain an entire month of hourly data.
    Path structure: {base}/{folder}/e5.oper.an.sfc/YYYYMM/filename.nc

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'SSTK', 'SP', 'T2M')
    year : int
        Year of data
    month : int
        Month of data (1-12)
    base, folder : str, optional
        Server base and dataset folder; default to the first candidate location.

    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    if base is None or folder is None:
        b, f = _default_location('hourly')
        base, folder = base or b, folder or f
    code, suffix = SURFACE_VARS[var_name]
    ym = f"{year}{month:02d}"
    last_day = calendar.monthrange(year, month)[1]
    filename = f"e5.oper.an.sfc.128_{code}.ll025{suffix}.{ym}0100_{ym}{last_day:02d}23.nc"
    return f"{base}/{folder}/e5.oper.an.sfc/{ym}/{filename}"


def _get_hourly_pressure_url(var_name: str, year: int, month: int, day: int,
                             base: Optional[str] = None, folder: Optional[str] = None) -> str:
    """
    Construct THREDDS URL for ERA5 hourly pressure level data.

    Pressure level files contain a single day of hourly data (24 hours).
    Path structure: {base}/{folder}/e5.oper.an.pl/YYYYMM/filename.nc

    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'T', 'U', 'V', 'Q', 'VO')
    year : int
        Year of data
    month : int
        Month of data (1-12)
    day : int
        Day of data (1-31)
    base, folder : str, optional
        Server base and dataset folder; default to the first candidate location.

    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    if base is None or folder is None:
        b, f = _default_location('hourly')
        base, folder = base or b, folder or f
    code, suffix = PRESSURE_VARS[var_name]
    ymd = f"{year}{month:02d}{day:02d}"
    filename = f"e5.oper.an.pl.128_{code}.ll025{suffix}.{ymd}00_{ymd}23.nc"
    return f"{base}/{folder}/e5.oper.an.pl/{ymd[:6]}/{filename}"


# ============================================================================
# Opening a file, with fallback and retries
# ============================================================================

_NOT_FOUND_ERRNO = -90        # NetCDF: file not found
_TRANSIENT_ERRNO = -68        # NetCDF: I/O failure (often curl/SSL underneath)
_NOT_FOUND_RE = re.compile(r"file ?not ?found|no such file|\b404\b")
_TRANSIENT_RE = re.compile(r"i/o failure|ssl|timed out|timeout|connection (?:reset|refused|aborted)|"
                           r"temporary failure|curl error|could(?:n't| not) (?:connect|resolve)")


def _classify(exc: BaseException, url: str = '') -> str:
    """'not_found', 'transient' or 'other' for an exception from the opener.

    Uses the NetCDF error number when there is one; otherwise the message, with
    the URL removed first so that a date such as 20220404 is not read as a 404.
    """
    errno = getattr(exc, 'errno', None)
    if errno == _NOT_FOUND_ERRNO:
        return 'not_found'
    if errno == _TRANSIENT_ERRNO:
        return 'transient'
    msg = str(exc).replace(url, '').lower() if url else str(exc).lower()
    if _NOT_FOUND_RE.search(msg):
        return 'not_found'
    if _TRANSIENT_RE.search(msg):
        return 'transient'
    return 'other'


def _open_era5(kind: Literal['monthly', 'hourly'], url_for: Callable[[str, str], str]) -> xr.Dataset:
    """Open one ERA5 file, trying each candidate location in turn.

    ``url_for(base, folder)`` builds the file's URL at a location. Transient
    failures are retried up to ``MAX_ATTEMPTS`` times with a short backoff;
    "file not found" and other errors move straight on to the next location.
    The first location that works is remembered for the rest of the session.
    """
    tried = []
    for base, folder in get_era5_thredds_candidates(kind):
        url = url_for(base, folder)
        for attempt in range(MAX_ATTEMPTS):
            try:
                ds = _open_dataset(url)
            except (OSError, RuntimeError, ValueError) as exc:
                kind_of_error = _classify(exc, url)
                if kind_of_error == 'transient' and attempt < MAX_ATTEMPTS - 1:
                    _sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                label = {'not_found': 'not found', 'transient': 'network error',
                         'other': 'error'}[kind_of_error]
                tried.append(f"{url}\n      -> {label} after {attempt + 1} attempt(s): {exc}")
                break
            else:
                _working[kind] = (base, folder)
                return ds

    page = GDEX_MONTHLY_PAGE if kind == 'monthly' else GDEX_HOURLY_PAGE
    raise ERA5DataUnavailableError(
        f"Could not open ERA5 {kind} data at any known location. Tried:\n  "
        + "\n  ".join(tried)
        + "\n\nIf every location says 'not found', NCAR may have moved the data. "
        f"Check the dataset page ({page}) for the current OPeNDAP location, then "
        "point tcpyVPI at it with\n"
        "    tcpyVPI.set_era5_thredds_base('https://<server>/thredds/dodsC/files/g')\n"
        f"or the environment variable {ERA5_THREDDS_ENV_VAR}. If the errors are network "
        "errors, the server may be down or unreachable from here; try again later."
    )


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_era5_monthly(year: int, month: int, verbose: bool = True) -> xr.Dataset:
    """
    Load ERA5 monthly mean data for GPIv calculation.
    
    Loads variables from the monthly mean dataset (d633001) via THREDDS; see
    the module docstring for where the files are read from.
    All 12 months are in a single file; this function extracts the specified month.
    
    Parameters
    ----------
    year : int
        Year of interest
    month : int
        Month of interest (1-12)
    verbose : bool
        Print loading progress messages
        
    Returns
    -------
    xr.Dataset
        Merged dataset containing all required fields for GPIv:
        SSTK, SP, T, Q, U, V, VO
    """
    if verbose:
        print(f"Loading ERA5 monthly mean data for {year}-{month:02d}...")
    
    datasets = {}
    
    # Load surface variables
    for var_name in ['SSTK', 'SP']:
        if verbose:
            print(f"  Loading {var_name}...")
        datasets[var_name] = _open_era5(
            'monthly', lambda b, f, v=var_name: _get_monthly_mean_url(v, year, True, b, f))
    
    # Load pressure level variables
    for var_name in ['T', 'Q', 'U', 'V', 'VO']:
        if verbose:
            print(f"  Loading {var_name}...")
        datasets[var_name] = _open_era5(
            'monthly', lambda b, f, v=var_name: _get_monthly_mean_url(v, year, False, b, f))
    
    # Extract the specified month (0-indexed)
    idx = month - 1
    
    # Merge all variables into a single dataset
    ds = xr.merge([
        datasets['SSTK'].SSTK.isel(time=idx),
        datasets['SP'].SP.isel(time=idx),
        datasets['T'].T.isel(time=idx),
        datasets['Q'].Q.isel(time=idx),
        datasets['U'].U.isel(time=idx),
        datasets['V'].V.isel(time=idx),
        datasets['VO'].VO.isel(time=idx),
    ], compat='override', join='outer')   # explicit: silences xarray's FutureWarning; the variables do not overlap
    
    if verbose:
        print("  Data loading complete.")
    
    return ds


def load_era5_hourly(
    year: int,
    month: int,
    day: int,
    hour: Optional[int] = None,
    verbose: bool = True
) -> xr.Dataset:
    """
    Load ERA5 hourly data for GPIv calculation.
    
    Loads variables from the hourly dataset (d633000) via THREDDS; see the
    module docstring for where the files are read from.
    Surface variables are in monthly files, pressure level variables in daily files.
    
    Parameters
    ----------
    year : int
        Year of interest
    month : int
        Month of interest (1-12)
    day : int
        Day of interest (1-31)
    hour : int, optional
        Specific hour to extract (0-23). If None, returns all 24 hours.
    verbose : bool
        Print loading progress messages
        
    Returns
    -------
    xr.Dataset
        Merged dataset containing all required fields for GPIv:
        SSTK, SP, T, Q, U, V, VO
        
    Notes
    -----
    The hourly dataset has different file structures:
    - Surface variables: monthly files (all hours of the month)
    - Pressure level variables: daily files (24 hours)
    
    Time selection is applied after loading to extract the specific hour if requested.
    """
    if verbose:
        hour_str = f" hour {hour:02d}" if hour is not None else ""
        print(f"Loading ERA5 hourly data for {year}-{month:02d}-{day:02d}{hour_str}...")
    
    # Create target time for selection
    target_time = datetime(year, month, day, hour if hour is not None else 0)
    
    datasets = {}
    
    # Load surface variables (monthly files)
    for var_name in ['SSTK', 'SP']:
        if verbose:
            print(f"  Loading {var_name} (surface, monthly file)...")
        ds = _open_era5(
            'hourly', lambda b, f, v=var_name: _get_hourly_surface_url(v, year, month, b, f))
        
        # Select the specific day/hour
        if hour is not None:
            ds = ds.sel(time=target_time, method='nearest')
        else:
            # Select all hours of the specified day
            ds = ds.sel(time=ds.time.dt.day == day)
        
        datasets[var_name] = ds
    
    # Load pressure level variables (daily files)
    for var_name in ['T', 'Q', 'U', 'V', 'VO']:
        if verbose:
            print(f"  Loading {var_name} (pressure level, daily file)...")
        ds = _open_era5(
            'hourly', lambda b, f, v=var_name: _get_hourly_pressure_url(v, year, month, day, b, f))
        
        # Select specific hour if requested
        if hour is not None:
            ds = ds.sel(time=target_time, method='nearest')
        
        datasets[var_name] = ds
    
    # Get the appropriate variable name from each dataset
    # ERA5 hourly files may have different naming conventions
    def get_var(ds, expected_name):
        """Extract variable, handling different naming conventions."""
        if expected_name in ds:
            return ds[expected_name]
        # Try uppercase
        if expected_name.upper() in ds:
            return ds[expected_name.upper()]
        # Try the first data variable
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]].rename(expected_name)
        raise KeyError(f"Could not find variable {expected_name} in dataset. "
                      f"Available: {data_vars}")
    
    # Merge all variables into a single dataset
    merged_vars = []
    for var_name, ds in datasets.items():
        try:
            var = get_var(ds, var_name)
            merged_vars.append(var.rename(var_name) if var.name != var_name else var)
        except KeyError:
            # If exact name not found, try to get the main variable
            data_vars = list(ds.data_vars)
            if data_vars:
                merged_vars.append(ds[data_vars[0]].rename(var_name))
    
    result = xr.merge(merged_vars, compat='override', join='outer')   # explicit, as in load_era5_monthly
    
    if verbose:
        print("  Data loading complete.")
    
    return result


def load_era5_hourly_range(
    start_date: datetime,
    end_date: datetime,
    hours: Optional[list] = None,
    verbose: bool = True
) -> xr.Dataset:
    """
    Load ERA5 hourly data for a date range.
    
    This function loads and concatenates hourly data across multiple days.
    Useful for computing daily or multi-day averages.
    
    Parameters
    ----------
    start_date : datetime
        Start date (inclusive)
    end_date : datetime
        End date (inclusive)
    hours : list of int, optional
        Specific hours to load (0-23). If None, loads all 24 hours.
    verbose : bool
        Print loading progress messages
        
    Returns
    -------
    xr.Dataset
        Concatenated dataset for the date range
        
    Example
    -------
    >>> from datetime import datetime
    >>> ds = load_era5_hourly_range(
    ...     datetime(2020, 8, 1),
    ...     datetime(2020, 8, 31),
    ...     hours=[0, 6, 12, 18]
    ... )
    """
    import pandas as pd
    
    all_datasets = []
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    for date in date_range:
        if hours is None:
            # Load all hours for this day
            ds = load_era5_hourly(
                date.year, date.month, date.day,
                hour=None, verbose=verbose
            )
            all_datasets.append(ds)
        else:
            # Load specific hours
            for h in hours:
                ds = load_era5_hourly(
                    date.year, date.month, date.day,
                    hour=h, verbose=verbose
                )
                all_datasets.append(ds)
    
    # Concatenate along time dimension
    result = xr.concat(all_datasets, dim='time')
    
    return result


# ============================================================================
# Data Source Selection
# ============================================================================

def load_era5_data(
    year: int,
    month: int,
    day: Optional[int] = None,
    hour: Optional[int] = None,
    data_source: Literal['monthly', 'hourly'] = 'monthly',
    verbose: bool = True
) -> xr.Dataset:
    """
    Unified interface for loading ERA5 data.
    
    This is the main entry point for loading ERA5 data. It automatically
    selects the appropriate loading function based on the data_source parameter.
    
    Parameters
    ----------
    year : int
        Year of interest
    month : int
        Month of interest (1-12)
    day : int, optional
        Day of interest (1-31). Required for hourly data.
    hour : int, optional
        Hour of interest (0-23). If None with hourly data, returns all 24 hours.
    data_source : {'monthly', 'hourly'}
        Which ERA5 dataset to use:
        - 'monthly': Monthly mean dataset (d633001)
        - 'hourly': Hourly dataset (d633000)
    verbose : bool
        Print loading progress messages
        
    Returns
    -------
    xr.Dataset
        Dataset containing all required fields for GPIv calculation
        
    Raises
    ------
    ValueError
        If day is not specified for hourly data
        
    Examples
    --------
    >>> # Load monthly mean for September 2022
    >>> ds_monthly = load_era5_data(2022, 9, data_source='monthly')
    
    >>> # Load hourly data for August 15, 2020 at 12Z
    >>> ds_hourly = load_era5_data(2020, 8, day=15, hour=12, data_source='hourly')
    
    >>> # Load all hours for August 15, 2020
    >>> ds_hourly_day = load_era5_data(2020, 8, day=15, data_source='hourly')
    """
    if data_source == 'monthly':
        return load_era5_monthly(year, month, verbose=verbose)
    
    elif data_source == 'hourly':
        if day is None:
            raise ValueError("day must be specified for hourly data")
        return load_era5_hourly(year, month, day, hour=hour, verbose=verbose)
    
    else:
        raise ValueError(f"Invalid data_source: {data_source}. "
                        "Must be 'monthly' or 'hourly'.")
