# tcpyVPI # note the capitalization 

A Python package to calculate the tropical cyclone ventilated Potential Intensity (vPI) and the Genesis Potential Index using vPI (GPIv) from gridded datafiles. 

See Chavas, Camargo, & Tippett (2025, J. Clim.) for details.

**Author:** Dan Chavas (2025)  
**Collaborators:** Aaron Kruskie, Jose Ocegueda Sanchez (2025)

## Installation

```bash
pip install tcpyVPI
```

Or install from source:
```bash
git clone https://github.com/drchavas/tcpyVPI.git
cd tcpyVPI
pip install -e .
```

## Features

- **Monthly Mean Data**: Compute GPIv from ERA5 monthly mean reanalysis (d633001)
- **Hourly Data**: Compute GPIv from ERA5 hourly reanalysis (d633000) via THREDDS remote access
- **Climatology**: Compute and store monthly climatologies of GPIv and its components
- **Anomalies**: Calculate anomalies relative to climatological means
- **Standardized Anomalies**: Compute z-scores for statistical analysis

## Quick Start

### Monthly Mean Computation

```python
from tcpyVPI import run_vpigpiv

# Compute GPIv for September 2022
results = run_vpigpiv(2022, 9)
```

### Hourly Computation

```python
from tcpyVPI import run_vpigpiv_hourly

# Compute GPIv for August 15, 2020 at 12Z
results = run_vpigpiv_hourly(2020, 8, 15, hour=12)
```

### With Anomalies

```python
from tcpyVPI import run_vpigpiv_hourly

# First, compute or load a climatology
results = run_vpigpiv_hourly(
    2020, 8, 15, hour=12,
    compute_anomalies=True,
    climatology_path='gpiv_climatology.nc'
)
```

## Data Loading

The package provides flexible data loading from NCAR RDA THREDDS servers:

```python
from tcpyVPI import load_era5_data, load_era5_hourly

# Load monthly mean data
ds_monthly = load_era5_data(2022, 9, data_source='monthly')

# Load hourly data for a specific time
ds_hourly = load_era5_data(2020, 8, day=15, hour=12, data_source='hourly')

# Load all hours of a day
ds_day = load_era5_hourly(2020, 8, 15)
```

### ERA5 Dataset Structure

The package accesses ERA5 data via THREDDS with the following structure:

**Monthly Mean (d633001):**
- All 12 months in a single file per variable per year
- Both surface and pressure level variables

**Hourly (d633000):**
- **Surface variables**: Monthly files containing all hours
  - Example: `e5.oper.an.sfc.128_165_10u.ll025sc.2020080100_2020083123.nc`
- **Pressure level variables**: Daily files containing 24 hours
  - Example: `e5.oper.an.pl.128_131_u.ll025uv.2020081500_2020081523.nc`

## Climatology Computation

```python
from tcpyVPI import compute_monthly_climatology, compute_gpiv_from_dataset

# Compute 40-year climatology (1980-2020)
climatology = compute_monthly_climatology(
    compute_gpiv_from_dataset,
    years=range(1980, 2020),
    output_path='gpiv_climatology.nc'
)
```

## Computing Components Individually

```python
from tcpyVPI import (
    load_era5_data,
    calculate_potential_intensity,
    calculate_vws,
    calculate_entropy_deficit,
    calculate_etac,
)

# Load data
ds = load_era5_data(2022, 9, data_source='monthly')

# Calculate individual components
PI, asdeq = calculate_potential_intensity(ds)
VWS = calculate_vws(ds)
Chi = calculate_entropy_deficit(ds, asdeq)
eta_c = calculate_etac(ds)
```

## Output Variables

The main computation returns a dataset with:

| Variable | Description | Units |
|----------|-------------|-------|
| `GPIv` | Ventilated Genesis Potential Index | - |
| `vPI` | Ventilated Potential Intensity | m/s |
| `PI` | Potential Intensity | m/s |
| `VWS` | Vertical Wind Shear (200-850 hPa) | m/s |
| `Chi` | Entropy Deficit | - |
| `eta_c` | Capped Absolute Vorticity (850 hPa) | s⁻¹ |
| `ventilation_index` | Ventilation Index | - |

When computing anomalies, additional fields are added:
- `*_anom`: Anomaly fields
- `*_clim`: Climatological values

## Dependencies

- numpy
- xarray
- tcpyPI
- matplotlib (for plotting)
- cartopy (for plotting)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package, please cite:

Chavas, D. R., Camargo, S. J., & Tippett, M. K. (2025). "Tropical cyclone genesis potential using a ventilated potential intensity". *Journal of Climate*.


