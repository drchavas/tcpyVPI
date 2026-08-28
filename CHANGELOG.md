# Changelog

All notable changes to tcpyVPI will be documented in this file.

## [1.0.2] - 2026-08-24

### Fixed

**These are results-changing bug fixes. Output from v1.0.1 and earlier should be
regarded as incorrect and recomputed.** Both bugs were in the inputs handed to
`tcpyPI.pi()` inside `calculate_potential_intensity()`, so they propagate to
every downstream quantity: `PI`, `asdeq`, `Chi`, `ventilation_index`, `vPI`,
and `GPIv`.

- **Surface pressure units.** `tcpyPI.pi()` expects the surface/sea-level
  pressure in hPa, matching its pressure-level argument, but gridded surface
  pressure is archived in Pa (ERA5 `SP`, CESM/CAM `PS`) and was being passed
  through unconverted. Added a `to_hPa()` helper that reads the CF `units`
  attribute (falling back to a magnitude test when units are absent) and
  normalises the field. The conversion happens at the point of use, so it
  covers the ERA5 monthly and hourly loaders, hand-built datasets such as the
  CESM2 example notebook, and user-supplied data, and is a no-op if the input
  is already in hPa.

  On an idealised tropical sounding (SST 27 °C, `V_reduc=1.0`) this alone
  changed PI from 80.5 to 67.1 m/s, roughly a 25% high bias. It also caused
  `tcpyPI`'s convergence flag to return `IFL=0`, i.e. the values were being
  flagged as invalid and the flag was not being checked.

- **Specific humidity used as mixing ratio.** `tcpyPI.pi()` expects mixing
  ratio in g/kg, but specific humidity was being passed as `q * 1000.0`. Now
  converted with the existing `get_rv_from_q()` helper, consistent with how
  the 600 hPa moisture is already handled in `calculate_entropy_deficit()`.
  Worth roughly a further 2% on PI (≤0.35 g/kg on the profile above).

  Combined effect on the same sounding: PI 80.5 → 64.4 m/s.

### Changed

- Corrected the `calculate_entropy_deficit()` docstring, which described a
  denominator computed from 2 m temperature and dewpoint (and previously
  925 hPa). The implementation in fact uses the `asdeq` term inverted from the
  potential intensity; the docstring now documents that, along with the fact
  that the 600 hPa numerator uses the environmental temperature for both the
  saturation and actual entropy, and the weak-temperature-gradient argument
  that justifies substituting it for the eyewall value.

### Notes

- Any stored climatologies or anomaly fields produced with v1.0.1 or earlier
  must be recomputed; anomalies are not protected by cancellation, since the
  bias is state-dependent through the sounding.

## [0.3.0] - 2025-XX-XX

### Added

#### Hourly ERA5 Data Support
- New `era5_loader.py` module for loading ERA5 data from THREDDS servers
- `load_era5_hourly()` function for loading hourly ERA5 data (d633000 dataset)
- `load_era5_hourly_range()` function for loading multiple days/hours
- `run_vpigpiv_hourly()` convenience function for hourly GPIv computation
- Support for different file structures:
  - Surface variables: monthly files containing all hours
  - Pressure level variables: daily files containing 24 hours

#### Climatology Module
- New `climatology.py` module for computing and storing climatologies
- `compute_monthly_climatology()` for computing long-term means
- `compute_climatology_statistics()` for computing both mean and standard deviation
- `load_climatology()` for loading pre-computed climatologies

#### Anomaly Calculation
- `compute_anomalies()` function for calculating anomalies relative to climatology
- `compute_standardized_anomalies()` for z-score calculation
- `compute_anomalies=True` parameter in `run_vpigpiv()` and `run_vpigpiv_hourly()`

#### Unified Data Loading
- `load_era5_data()` unified interface supporting both monthly and hourly data
- `data_source` parameter to select between 'monthly' and 'hourly' datasets

### Changed
- Updated `run_vpigpiv()` to accept `data_source`, `day`, `hour` parameters
- Improved module organization with separate files for data loading and climatology
- Updated `__init__.py` with comprehensive exports
- Enhanced documentation and examples

### Files Added
- `tcvpigpiv/era5_loader.py` - ERA5 data loading functions
- `tcvpigpiv/climatology.py` - Climatology computation and anomalies
- `python_notebooks/tcvpigpiv_hourly_example.ipynb` - Example notebook
- `CHANGELOG.md` - This file

### Technical Details

#### THREDDS URL Structure

**Monthly Mean (d633001):**
```
https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633001_nc/e5.moda.an.{sfc|pl}/{year}/
  e5.moda.an.{sfc|pl}.128_{code}_{var}.ll025{suffix}.{year}010100_{year}120100.nc
```

**Hourly Surface (d633000):**
```
https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.sfc/{YYYYMM}/
  e5.oper.an.sfc.128_{code}_{var}.ll025{suffix}.{YYYYMM}0100_{YYYYMM}{last_day}23.nc
```

**Hourly Pressure Level (d633000):**
```
https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.pl/{YYYYMM}/
  e5.oper.an.pl.128_{code}_{var}.ll025{suffix}.{YYYYMMDD}00_{YYYYMMDD}23.nc
```

## [0.2.2] - Previous Release

- Initial PyPI release
- Monthly mean ERA5 data support
- Basic GPIv computation

---

## Migration Guide

### From 0.2.x to 0.3.0

The existing API remains fully compatible. New features are additive:

```python
# Old usage (still works)
from tcvpigpiv.vpigpiv_module import run_vpigpiv
results = run_vpigpiv(2022, 9)

# New usage with hourly data
from tcvpigpiv import run_vpigpiv_hourly
results = run_vpigpiv_hourly(2020, 8, 15, hour=12)

# New usage with anomalies
results = run_vpigpiv_hourly(
    2020, 8, 15, hour=12,
    compute_anomalies=True,
    climatology_path='gpiv_climatology.nc'
)
```
