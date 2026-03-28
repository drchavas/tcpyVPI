"""
tcpyVPI - Tropical Cyclone Ventilated Potential Intensity and Genesis Potential Index

A Python package to calculate the tropical cyclone ventilated Potential
Intensity (vPI), and also the Genesis Potential Index using vPI (GPIv), from
gridded datafiles.

This package supports both monthly mean and hourly ERA5 reanalysis data
accessed remotely via THREDDS servers.

References:
    Chavas, Camargo, & Tippett (2025, J. Clim.)

Authors:
    Dan Chavas (2025) - Original implementation
    Jose Ocegueda Sanchez (2025) - Hourly data and anomaly support
"""

from .vpigpiv_module import (
    run_vpigpiv,
    run_vpigpiv_hourly,
    compute_gpiv_from_dataset,
    calculate_potential_intensity,
    calculate_vws,
    calculate_entropy_deficit,
    calculate_etac,
    plot_vpigpiv,
)

from .era5_loader import (
    load_era5_data,
    load_era5_monthly,
    load_era5_hourly,
    load_era5_hourly_range,
)

from .climatology import (
    compute_monthly_climatology,
    compute_climatology_statistics,
    load_climatology,
    compute_anomalies,
    compute_standardized_anomalies,
)

__version__ = "0.1.0"
__author__ = "Dan Chavas, Jose Ocegueda Sanchez"

__all__ = [
    # Main computation functions
    "run_vpigpiv",
    "run_vpigpiv_hourly",
    "compute_gpiv_from_dataset",
    
    # Component calculation functions
    "calculate_potential_intensity",
    "calculate_vws",
    "calculate_entropy_deficit",
    "calculate_etac",
    
    # Data loading
    "load_era5_data",
    "load_era5_monthly",
    "load_era5_hourly",
    "load_era5_hourly_range",
    
    # Climatology
    "compute_monthly_climatology",
    "compute_climatology_statistics",
    "load_climatology",
    "compute_anomalies",
    "compute_standardized_anomalies",
    
    # Plotting
    "plot_vpigpiv",
]
