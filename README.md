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
| `vort_cap` | 3.7e-5 s⁻¹ | cap on the magnitude of absolute vorticity |
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
| `eta_c` | Capped absolute vorticity (850 hPa), **signed** | s⁻¹ |
| `eta_c_cyclonic` | Same, hemisphere-mirrored and clipped at zero — the form GPIv uses | s⁻¹ |
| `ventilation_index` | Ventilation Index | - |

### Grid assumptions

**The code assumes a regular, fixed-spacing latitude–longitude grid.** The GPIv
grid-box area term is `cos(lat) · Δlon · Δlat`, and `Δlon` and `Δlat` are
computed internally as the **median spacing** of the `longitude` and `latitude`
coordinates of your input data. They are constants, evaluated once per call.

If you have a variable-resolution grid, an irregular grid, or anything that is
not a plain lat–lon mesh, that single constant is not meaningful. **Interpolate
your data onto a fixed-spacing grid first** — 2° × 2° matches the published
calibration — or modify the code to compute the area term appropriately for your
grid. A `RuntimeWarning` is raised if the spacing varies by more than 1%; it is
silent on a regular grid at any resolution.

Two input shapes are rejected outright, with a `ValueError`, because they used to
produce a finite but meaningless number:

- **A collapsed coordinate**, e.g. after `ds.sel(latitude=0.5)` — a natural thing
  to do when checking a single location. The coordinate is then 0-d, and taking
  its difference returns the latitude itself, so `Δlat` silently became 0.5.
- **A single-point coordinate**, which has no spacing to measure.

Rolled or re-centred longitudes (`0…179.75` then `-180…-0.25`) are handled
correctly: longitude differences are unwrapped onto (−180°, 180°], so the ±360°
seam does not distort `Δlon`.

### GPIv is calibrated for 2°, and is not resolution-invariant

The area weighting is correct at any uniform spacing, so GPIv **sums are
grid-consistent**. That is not the same as being resolution-independent, and the
distinction matters if you compare totals against Chavas et al. (2025).

The coefficient 102.1 and exponent 4.90 were fit on 2° fields. Off that grid
there are **two distinct effects, pointing in opposite directions**:

**1. Per-gridbox values scale with gridbox area.** GPIv is defined per unit
gridbox area, so a 0.25° value is ~1/64 of the 2° value at the same location —
finer grid, *lower* pointwise values. This is the `dx · dy` factor doing its job,
and it is the reason a 0.25° GPIv map will not match a published colorbar.

**2. Sums run high.** GPIv goes as roughly the fifth power of `vPI · η_c`, which
is convex, so by Jensen's inequality evaluating it on fine-grid fields and
summing gives a systematically **larger** total than evaluating it on the same
fields averaged to 2°: the fine grid resolves peaks that 2° averaging smooths,
and the exponent amplifies them.

The size of effect 2 is set by the coefficient of variation `c` of the sub-2°
variability of `vPI · η_c` **within** a 2° box — not the total variance of the
field, since between-box variance survives the coarsening and cancels. To second
order the ratio is `1 + p(p−1)c²/2` with `p = 4.90`, i.e. `1 + 9.56 c²`:

| within-box CV `c` | Gaussian | lognormal | `1 + 9.56 c²` |
|---|---|---|---|
| 0.10 | 1.10× | 1.10× | 1.10× |
| 0.15 | 1.22× | 1.24× | 1.21× |
| 0.25 | 1.65× | 1.78× | 1.60× |
| 0.40 | 2.86× | 4.13× | 2.53× |

It is one-directional — finer always sums higher — and it grows fast, so at
realistic sub-2° variability this is a factor-of-2-or-more correction, not a
rounding error. The spread between the columns is the tail sensitivity: the
heavier the distribution of `vPI · η_c`, the larger the effect.

`run_vpigpiv()` loads **native 0.25° ERA5**, so its output is off-calibration by
default. **Coarsen to 2° to reproduce or compare against published values.**

Rather than warn at runtime, the grid is recorded on the output so it travels
into a saved netCDF and is still there when the file is reopened:

```python
r = compute_gpiv_from_dataset(ds)
r['GPIv'].attrs['grid_dx_deg']           # 0.25
r['GPIv'].attrs['calibration_grid_deg']  # 2.0
r['GPIv'].attrs['calibrated']            # 'no'
r['GPIv'].attrs['comment']               # the full caveat
```

`verbose=True` also flags it on the grid-spacing line.

### A note on `eta_c`

Two vorticity fields are returned, and the difference matters if you plot or
analyse them:

- **`eta_c`** is the *signed* absolute vorticity, clipped by magnitude to
  ±`vort_cap`. It is **negative in the Southern Hemisphere**, which is physically
  correct and is what you want on a map.
- **`eta_c_cyclonic`** mirrors the hemispheres so cyclonic rotation is positive
  everywhere, then clips anticyclonic values to zero. **This is the field GPIv is
  built from**, because GPIv raises it to a non-integer power (4.90) and a
  negative base would be NaN.

`calculate_etac()` returns the signed form by default; pass `cyclonic=True` for
the mirrored one.

```python
eta_signed   = calculate_etac(ds)                  # negative in the SH
eta_cyclonic = calculate_etac(ds, cyclonic=True)   # non-negative everywhere
```

Through v1.2.0 there was only one field and it was wrong in the SH; see the
v1.3.0 entry in [CHANGELOG.md](CHANGELOG.md).

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


