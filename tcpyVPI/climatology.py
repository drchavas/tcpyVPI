"""
Climatology computation module for tcvpigpiv.

This module provides functions for computing monthly climatologies of
potential intensity (PI), ventilated PI (vPI), GPIv, and their components.
Climatologies can be computed from either monthly mean or hourly ERA5 data.

The climatology is essential for computing anomalies from instantaneous
(hourly) fields.

Author: Jose Ocegueda Sanchez (2025)
"""

import os
from pathlib import Path
from typing import Optional, List, Union, Literal
from datetime import datetime

import numpy as np
import xarray as xr


def compute_monthly_climatology(
    compute_func,
    months: List[int] = list(range(1, 13)),
    years: List[int] = list(range(1980, 2020)),
    data_source: Literal['monthly', 'hourly'] = 'monthly',
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **compute_kwargs
) -> xr.Dataset:
    """
    Compute monthly climatology of GPIv and its components.
    
    This function computes the long-term monthly mean for each calendar month
    by averaging over multiple years. The result is a 12-month climatology
    that can be used to compute anomalies.
    
    Parameters
    ----------
    compute_func : callable
        Function that computes GPIv from a dataset. Should accept an xr.Dataset
        and return an xr.Dataset with computed fields.
        Typically: `compute_gpiv_from_dataset` from vpigpiv_module.
    months : list of int
        Which months to include (1-12). Default is all months.
    years : list of int
        Years to average over for climatology. Default is 1991-2020 (30 years).
    data_source : {'monthly', 'hourly'}
        Which ERA5 dataset to use as input.
    output_path : str or Path, optional
        If provided, save the climatology to this NetCDF file.
    verbose : bool
        Print progress messages.
    **compute_kwargs
        Additional keyword arguments passed to compute_func.
        
    Returns
    -------
    xr.Dataset
        Dataset with dimensions (month, latitude, longitude) containing
        the climatological mean of GPIv and its components.
        
    Examples
    --------
    >>> from tcvpigpiv.vpigpiv_module import compute_gpiv_from_dataset
    >>> clim = compute_monthly_climatology(
    ...     compute_gpiv_from_dataset,
    ...     years=range(2000, 2020),
    ...     output_path='gpiv_climatology.nc'
    ... )
    """
    from .era5_loader import load_era5_data
    
    all_results = []
    
    for month in months:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Computing climatology for month {month:02d}")
            print(f"{'='*60}")
        
        monthly_results = []
        
        for year in years:
            if verbose:
                print(f"  Processing {year}-{month:02d}...")
            
            try:
                # Load data
                if data_source == 'monthly':
                    ds = load_era5_data(year, month, data_source='monthly', verbose=False)
                else:
                    # For hourly data, compute monthly mean from all hours
                    # This requires loading and averaging all hours of all days
                    import calendar
                    last_day = calendar.monthrange(year, month)[1]
                    
                    # Load representative times (e.g., 4 times per day)
                    day_results = []
                    for day in range(1, last_day + 1):
                        for hour in [0, 6, 12, 18]:
                            try:
                                ds_hour = load_era5_data(
                                    year, month, day=day, hour=hour,
                                    data_source='hourly', verbose=False
                                )
                                result_hour = compute_func(ds_hour, **compute_kwargs)
                                day_results.append(result_hour)
                            except Exception as e:
                                if verbose:
                                    print(f"    Warning: Could not load {year}-{month:02d}-{day:02d} {hour:02d}Z: {e}")
                    
                    if day_results:
                        # Compute monthly mean from hourly results
                        result = xr.concat(day_results, dim='time').mean(dim='time')
                        monthly_results.append(result.expand_dims(year=[year]))
                    continue
                
                # Compute GPIv for this month
                result = compute_func(ds, **compute_kwargs)
                monthly_results.append(result.expand_dims(year=[year]))
                
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not process {year}-{month:02d}: {e}")
                continue
        
        if monthly_results:
            # Average over years for this month
            month_stack = xr.concat(monthly_results, dim='year')
            month_clim = month_stack.mean(dim='year')
            month_clim = month_clim.expand_dims(month=[month])
            all_results.append(month_clim)
    
    if not all_results:
        raise RuntimeError("No data could be processed for climatology")
    
    # Combine all months
    climatology = xr.concat(all_results, dim='month')
    
    # Add metadata
    climatology.attrs['title'] = 'GPIv Monthly Climatology'
    climatology.attrs['source'] = f'ERA5 {data_source} data'
    climatology.attrs['years'] = f'{min(years)}-{max(years)}'
    climatology.attrs['created'] = datetime.now().isoformat()
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        climatology.to_netcdf(output_path)
        if verbose:
            print(f"\nClimatology saved to: {output_path}")
    
    return climatology


def load_climatology(path: Union[str, Path]) -> xr.Dataset:
    """
    Load a pre-computed climatology from a NetCDF file.
    
    Parameters
    ----------
    path : str or Path
        Path to the climatology NetCDF file.
        
    Returns
    -------
    xr.Dataset
        Climatology dataset with dimensions (month, latitude, longitude).
    """
    return xr.open_dataset(path)


def compute_anomalies(
    instantaneous: xr.Dataset,
    climatology: xr.Dataset,
    month: int,
    variables: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Compute anomalies relative to monthly climatology.
    
    This function subtracts the climatological mean for the given month
    from instantaneous fields to produce anomalies.
    
    Parameters
    ----------
    instantaneous : xr.Dataset
        Dataset containing instantaneous (e.g., hourly) computed fields.
    climatology : xr.Dataset
        Climatology dataset with 'month' dimension.
    month : int
        Calendar month (1-12) for selecting the appropriate climatology.
    variables : list of str, optional
        Which variables to compute anomalies for. If None, computes for
        all variables present in both datasets.
        
    Returns
    -------
    xr.Dataset
        Dataset containing anomaly fields (same structure as instantaneous).
        
    Examples
    --------
    >>> clim = load_climatology('gpiv_climatology.nc')
    >>> ds_hourly = load_era5_data(2020, 8, day=15, hour=12, data_source='hourly')
    >>> results = compute_gpiv_from_dataset(ds_hourly)
    >>> anomalies = compute_anomalies(results, clim, month=8)
    """
    # Get climatology for this month
    clim_month = climatology.sel(month=month)
    
    # Determine which variables to process
    if variables is None:
        variables = list(set(instantaneous.data_vars) & set(climatology.data_vars))
    
    anomaly_data = {}
    
    for var in variables:
        if var in instantaneous and var in clim_month:
            # Ensure coordinates align (interpolate if needed)
            clim_var = clim_month[var]
            inst_var = instantaneous[var]
            
            # Handle potential coordinate mismatches
            if not (set(clim_var.dims) <= set(inst_var.dims) | {'month'}):
                # Try to interpolate climatology to instantaneous grid
                clim_var = clim_var.interp_like(inst_var)
            
            anomaly = inst_var - clim_var
            anomaly.attrs = inst_var.attrs.copy()
            anomaly.attrs['long_name'] = f"{inst_var.attrs.get('long_name', var)} Anomaly"
            anomaly_data[f"{var}_anom"] = anomaly
            
            # Also keep the original values
            anomaly_data[var] = inst_var
            anomaly_data[f"{var}_clim"] = clim_var
    
    result = xr.Dataset(anomaly_data)
    result.attrs['anomaly_reference'] = f"Month {month} climatology"
    
    return result


def compute_standardized_anomalies(
    instantaneous: xr.Dataset,
    climatology_mean: xr.Dataset,
    climatology_std: xr.Dataset,
    month: int,
    variables: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Compute standardized anomalies (z-scores) relative to climatology.
    
    Standardized anomalies are computed as:
        z = (x - μ) / σ
    where x is the instantaneous value, μ is the climatological mean,
    and σ is the climatological standard deviation.
    
    Parameters
    ----------
    instantaneous : xr.Dataset
        Dataset containing instantaneous computed fields.
    climatology_mean : xr.Dataset
        Climatology of the mean with 'month' dimension.
    climatology_std : xr.Dataset
        Climatology of the standard deviation with 'month' dimension.
    month : int
        Calendar month (1-12).
    variables : list of str, optional
        Which variables to compute standardized anomalies for.
        
    Returns
    -------
    xr.Dataset
        Dataset containing standardized anomaly fields.
    """
    clim_mean = climatology_mean.sel(month=month)
    clim_std = climatology_std.sel(month=month)
    
    if variables is None:
        variables = list(
            set(instantaneous.data_vars) & 
            set(climatology_mean.data_vars) & 
            set(climatology_std.data_vars)
        )
    
    anomaly_data = {}
    
    for var in variables:
        if var in instantaneous and var in clim_mean and var in clim_std:
            inst_var = instantaneous[var]
            mean_var = clim_mean[var]
            std_var = clim_std[var]
            
            # Compute standardized anomaly
            z = (inst_var - mean_var) / std_var
            z = z.where(std_var > 0)  # Avoid division by zero
            
            z.attrs = inst_var.attrs.copy()
            z.attrs['long_name'] = f"{inst_var.attrs.get('long_name', var)} Standardized Anomaly"
            z.attrs['units'] = 'standard deviations'
            anomaly_data[f"{var}_zanom"] = z
    
    result = xr.Dataset(anomaly_data)
    result.attrs['anomaly_reference'] = f"Month {month} standardized anomaly"
    
    return result


def compute_climatology_statistics(
    compute_func,
    months: List[int] = list(range(1, 13)),
    years: List[int] = list(range(1980, 2020)),
    data_source: Literal['monthly', 'hourly'] = 'monthly',
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **compute_kwargs
) -> tuple:
    """
    Compute both mean and standard deviation climatologies.
    
    This is useful for computing standardized anomalies.
    
    Parameters
    ----------
    compute_func : callable
        Function that computes GPIv from a dataset.
    months : list of int
        Which months to include (1-12).
    years : list of int
        Years to average over.
    data_source : {'monthly', 'hourly'}
        Which ERA5 dataset to use.
    output_path : str or Path, optional
        Base path for saving. Will create _mean.nc and _std.nc files.
    verbose : bool
        Print progress messages.
    **compute_kwargs
        Additional keyword arguments passed to compute_func.
        
    Returns
    -------
    tuple of xr.Dataset
        (climatology_mean, climatology_std) datasets.
    """
    from .era5_loader import load_era5_data
    
    all_results = {month: [] for month in months}
    
    for year in years:
        for month in months:
            if verbose:
                print(f"Processing {year}-{month:02d}...")
            
            try:
                ds = load_era5_data(year, month, data_source=data_source, verbose=False)
                result = compute_func(ds, **compute_kwargs)
                all_results[month].append(result)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not process {year}-{month:02d}: {e}")
    
    # Compute statistics for each month
    mean_results = []
    std_results = []
    
    for month in months:
        if all_results[month]:
            stacked = xr.concat(all_results[month], dim='year')
            
            month_mean = stacked.mean(dim='year').expand_dims(month=[month])
            month_std = stacked.std(dim='year').expand_dims(month=[month])
            
            mean_results.append(month_mean)
            std_results.append(month_std)
    
    clim_mean = xr.concat(mean_results, dim='month')
    clim_std = xr.concat(std_results, dim='month')
    
    # Add metadata
    for ds, stat_type in [(clim_mean, 'mean'), (clim_std, 'std')]:
        ds.attrs['title'] = f'GPIv Monthly Climatology ({stat_type})'
        ds.attrs['source'] = f'ERA5 {data_source} data'
        ds.attrs['years'] = f'{min(years)}-{max(years)}'
        ds.attrs['created'] = datetime.now().isoformat()
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        mean_path = output_path.parent / f"{output_path.stem}_mean{output_path.suffix}"
        std_path = output_path.parent / f"{output_path.stem}_std{output_path.suffix}"
        
        clim_mean.to_netcdf(mean_path)
        clim_std.to_netcdf(std_path)
        
        if verbose:
            print(f"\nClimatology mean saved to: {mean_path}")
            print(f"Climatology std saved to: {std_path}")
    
    return clim_mean, clim_std
