# tcpyVPI

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
- **Configurable**: Pressure levels, thresholds and PI options can be overridden
  for sensitivity testing; defaults reproduce Chavas et al. (2025)

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

## Configuration

By default the package reproduces the configuration of Chavas et al. (2025), and
you never need to touch any of this. Since v1.2.0 the choices that were
previously hardcoded can be overridden for sensitivity testing:

| Argument | Default | Meaning |
|----------|---------|---------|
| `shear_p_top` | 200 hPa | top of the bulk shear layer |
| `shear_p_bot` | 850 hPa | bottom of the bulk shear layer |
| `chi_p_mid` | 600 hPa | mid-level for the entropy deficit |
| `vort_level` | 850 hPa | level of the relative vorticity |
| `vort_cap` | 3.7e-5 s⁻¹ | cap on absolute vorticity |
| `VI_max` | 0.145 | ventilation index above which vPI = 0 |
| `gpiv_exponent` | 4.90 | exponent in `GPIv = (102.1 · vPI · η_c)^a` |
| `CKCD` | 0.9 | ratio C_k/C_d, passed to tcpyPI |
| `ascent_flag` | 0 | tcpyPI: 0 = reversible, 1 = pseudo-adiabatic |
| `diss_flag` | 1 | tcpyPI: 1 = dissipative heating on, 0 = off |
| `ptop` | 50 hPa | sounding above this level is ignored |

```python
# defaults - reproduces the paper
results = run_vpigpiv(2022, 9)

# override any subset
results = run_vpigpiv(2022, 9, chi_p_mid=500, shear_p_top=250)
```

They work on `run_vpigpiv`, `run_vpigpiv_hourly` and `compute_gpiv_from_dataset`,
and the level/threshold arguments are also on the individual `calculate_*`
functions. Defaults are importable as module constants
(`DEFAULT_SHEAR_P_TOP`, etc.).

**See `python_notebooks/tcpyVPI_ERA5builtin_example.ipynb` section 3** for a
runnable demo of passing these values. For a side-by-side default-vs-custom
comparison with difference maps, see
`github_tests/tcpyVPI_ERA5_from_github.ipynb`.

Two caveats:

- Pressure levels are selected **exactly**, not by nearest neighbour. Asking for
  a level the dataset does not contain raises a `KeyError` listing what is
  available; interpolate onto that level first.
- `gpiv_exponent` is tied to the normalising constant 102.1, which was
  calibrated jointly with the default exponent 4.90 to match the observed global
  mean genesis count. Changing the exponent alone leaves GPIv un-normalised:
  spatial patterns remain meaningful, absolute values do not.

## Example Notebooks

| Notebook | What it shows |
|----------|---------------|
| `python_notebooks/tcpyVPI_ERA5builtin_example.ipynb` | Shortest path: monthly and hourly via the wrapper functions, plus the configuration options |
| `python_notebooks/tcpyVPI_ERA5_example.ipynb` | Loading and computing in separate steps, with custom plots |
| `python_notebooks/tcpyVPI_CESM2_example.ipynb` | Climate-model output instead of reanalysis |
| `python_notebooks/tcpyVPI_ERA5climoanomaly_Jose.ipynb` | Climatologies and anomalies |
| `github_tests/tcpyVPI_ERA5_from_github.ipynb` | Installing from GitHub rather than PyPI, and comparing output across versions or configurations |

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


