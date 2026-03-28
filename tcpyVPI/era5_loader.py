"""
ERA5 data loader for the tcvpigpiv package.

This module provides functions to load ERA5 reanalysis data from NCAR RDA
THREDDS servers for both monthly mean (d633001) and hourly (d633000) datasets.

The hourly dataset has different file structures for surface vs pressure level data:
- Surface variables: monthly files containing all hours (YYYYMM0100_YYYYMM{last_day}23)
- Pressure level variables: daily files containing 24 hours (YYYYMMDD00_YYYYMMDD23)

Author: Jose Ocegueda Sanchez (2025)
Based on original monthly data loader by Dan Chavas (2025)
"""

import calendar
from typing import Optional, Literal, Union
from datetime import datetime

import xarray as xr
import numpy as np

# ============================================================================
# THREDDS Base URLs
# ============================================================================
THREDDS_BASE = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g"

# Monthly mean dataset (d633001)
BASE_MMEAN_SFC = f"{THREDDS_BASE}/d633001_nc/e5.moda.an.sfc"
BASE_MMEAN_PL = f"{THREDDS_BASE}/d633001_nc/e5.moda.an.pl"

# Hourly dataset (d633000)
BASE_HOURLY_SFC = f"{THREDDS_BASE}/d633000/e5.oper.an.sfc"
BASE_HOURLY_PL = f"{THREDDS_BASE}/d633000/e5.oper.an.pl"


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

def _get_monthly_mean_url(var_name: str, year: int, is_surface: bool = True) -> str:
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
        
    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    if is_surface:
        var_info = SURFACE_VARS[var_name]
        base = BASE_MMEAN_SFC
    else:
        var_info = PRESSURE_VARS[var_name]
        base = BASE_MMEAN_PL
    
    code, suffix = var_info
    filename = f"e5.moda.an.{'sfc' if is_surface else 'pl'}.128_{code}.ll025{suffix}.{year}010100_{year}120100.nc"
    
    return f"{base}/{year}/{filename}"


def _get_hourly_surface_url(var_name: str, year: int, month: int) -> str:
    """
    Construct THREDDS URL for ERA5 hourly surface data.
    
    Surface files contain an entire month of hourly data.
    Path structure: BASE_URL/YYYYMM/filename.nc
    
    Parameters
    ----------
    var_name : str
        Variable name (e.g., 'SSTK', 'SP', 'T2M')
    year : int
        Year of data
    month : int
        Month of data (1-12)
        
    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    code, suffix = SURFACE_VARS[var_name]
    year_str = str(year)
    month_str = str(month).zfill(2)
    last_day = calendar.monthrange(year, month)[1]
    
    filename = (f"e5.oper.an.sfc.128_{code}.ll025{suffix}."
                f"{year_str}{month_str}0100_{year_str}{month_str}{last_day:02d}23.nc")
    
    return f"{BASE_HOURLY_SFC}/{year_str}{month_str}/{filename}"


def _get_hourly_pressure_url(var_name: str, year: int, month: int, day: int) -> str:
    """
    Construct THREDDS URL for ERA5 hourly pressure level data.
    
    Pressure level files contain a single day of hourly data (24 hours).
    Path structure: BASE_URL/YYYYMM/filename.nc
    
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
        
    Returns
    -------
    str
        Complete THREDDS URL for the variable
    """
    code, suffix = PRESSURE_VARS[var_name]
    year_str = str(year)
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    
    filename = (f"e5.oper.an.pl.128_{code}.ll025{suffix}."
                f"{year_str}{month_str}{day_str}00_{year_str}{month_str}{day_str}23.nc")
    
    return f"{BASE_HOURLY_PL}/{year_str}{month_str}/{filename}"


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_era5_monthly(year: int, month: int, verbose: bool = True) -> xr.Dataset:
    """
    Load ERA5 monthly mean data for GPIv calculation.
    
    Loads variables from the monthly mean dataset (d633001) via THREDDS.
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
        url = _get_monthly_mean_url(var_name, year, is_surface=True)
        if verbose:
            print(f"  Loading {var_name}...")
        datasets[var_name] = xr.open_dataset(url)
    
    # Load pressure level variables
    for var_name in ['T', 'Q', 'U', 'V', 'VO']:
        url = _get_monthly_mean_url(var_name, year, is_surface=False)
        if verbose:
            print(f"  Loading {var_name}...")
        datasets[var_name] = xr.open_dataset(url)
    
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
    ])
    
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
    
    Loads variables from the hourly dataset (d633000) via THREDDS.
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
        url = _get_hourly_surface_url(var_name, year, month)
        if verbose:
            print(f"  Loading {var_name} (surface, monthly file)...")
        ds = xr.open_dataset(url)
        
        # Select the specific day/hour
        if hour is not None:
            ds = ds.sel(time=target_time, method='nearest')
        else:
            # Select all hours of the specified day
            ds = ds.sel(time=ds.time.dt.day == day)
        
        datasets[var_name] = ds
    
    # Load pressure level variables (daily files)
    for var_name in ['T', 'Q', 'U', 'V', 'VO']:
        url = _get_hourly_pressure_url(var_name, year, month, day)
        if verbose:
            print(f"  Loading {var_name} (pressure level, daily file)...")
        ds = xr.open_dataset(url)
        
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
    
    result = xr.merge(merged_vars)
    
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
