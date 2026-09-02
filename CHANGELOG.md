# Changelog

All notable changes to tcpyVPI will be documented in this file.

## [1.2.0] - 2026-09-02

### Added

Configuration options for the previously hardcoded choices. **Every default
reproduces v1.1.0 exactly**, so existing code and existing results are
unchanged; the arguments exist so the choices can be varied for sensitivity
testing without editing the source.

| Argument | Default | Function |
| --- | --- | --- |
| `shear_p_top` | 200 hPa | `calculate_vws` (as `p_top`), `compute_gpiv_from_dataset` |
| `shear_p_bot` | 850 hPa | `calculate_vws` (as `p_bot`), `compute_gpiv_from_dataset` |
| `chi_p_mid` | 600 hPa | `calculate_entropy_deficit` (as `p_mid`), `compute_gpiv_from_dataset` |
| `vort_level` | 850 hPa | `calculate_etac` (as `p_level`), `compute_gpiv_from_dataset` |
| `vort_cap` | 3.7e-5 s⁻¹ | `calculate_etac`, `compute_gpiv_from_dataset` |
| `VI_max` | 0.145 | `compute_gpiv_from_dataset` |
| `gpiv_exponent` | 4.90 | `compute_gpiv_from_dataset` |
| `CKCD` | 0.9 | `calculate_potential_intensity`, `compute_gpiv_from_dataset` |
| `ascent_flag` | 0 | `calculate_potential_intensity`, `compute_gpiv_from_dataset` |
| `diss_flag` | 1 | `calculate_potential_intensity`, `compute_gpiv_from_dataset` |
| `ptop` | 50 hPa | `calculate_potential_intensity`, `compute_gpiv_from_dataset` |

`run_vpigpiv()` and `run_vpigpiv_hourly()` forward any of these through
`**params`, e.g. `run_vpigpiv(2022, 9, chi_p_mid=500)`.

Defaults are also available as module constants (`DEFAULT_SHEAR_P_TOP` etc.).

`ptop` is the pressure below which `tcpyPI` ignores the sounding. It also
interacts with missing data: `tcpyPI` sets the output to missing (`IFL=3`) if
any level between the lowest valid level and `ptop` is NaN, so raising `ptop`
can rescue profiles that do not extend cleanly into the stratosphere.

- `tests/check_config_defaults.py` verifies that the defaults are bit-identical
  to the hardcoded behaviour and that each argument actually propagates.

### Changed

- The mid-level pressure used in the entropy calculation is now derived from
  `chi_p_mid` rather than being a second independent literal. Through v1.1.0 the
  level (`sel(level=600)`) and the pressure (`p=60000.` Pa) were separate
  constants that had to be kept in sync by hand; changing one without the other
  would have silently evaluated the entropy at the wrong pressure.
- Level selection now raises a `KeyError` naming the missing level and listing
  the available ones, instead of a bare xarray error. Selection remains **exact**,
  not nearest-neighbour, matching all previous versions: a dataset must contain
  the requested level. Interpolate first if it does not, as the CESM2 example
  notebook does.

### Notes

- `gpiv_exponent` interacts with the normalising constant 102.1, which was
  calibrated jointly with the exponent 4.90 by matching the observed global
  annual mean genesis count. Changing the exponent alone leaves GPIv
  un-normalised: spatial patterns and relative comparisons remain valid,
  absolute values do not. The constant is available as `DEFAULT_GPIV_COEFF`
  if you refit.
- `calculate_potential_intensity` still defaults to `V_reduc=0.8` while
  `compute_gpiv_from_dataset` overrides it to 1.0, so calling the former
  directly gives a PI 20% lower than the pipeline reports. Unchanged from
  previous versions, but worth knowing.

## [1.1.0] - 2026-08-28

Both bugs below were reported independently by Yulian Tang, who spotted them
while working with the code. Thank you.

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

### Added

- `validate_pi_inputs()`, called from `calculate_potential_intensity()` just
  before `tcpyPI.pi()`. It raises `ValueError` with a message naming the
  suspected unit problem if surface pressure or the pressure levels look like
  Pa rather than hPa, if SST or the temperature profile look like Kelvin rather
  than Celsius, or if the mixing ratio looks like kg/kg rather than g/kg. The
  thresholds are order-of-magnitude, so only a genuine unit mistake trips them.
  Checked on a single sampled column, so it stays cheap for lazy/remote data.

  Motivated by the two bugs above: both produced plausible-looking output and
  survived several releases. This class of error should now fail loudly at the
  call rather than silently biasing the result.

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
