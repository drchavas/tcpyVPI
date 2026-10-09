# --- tcpyVPI unit-handling regression check: no data download required ---
#
# Guards the two bugs fixed in v1.1.0, where the inputs handed to tcpyPI.pi()
# had the wrong units and produced plausible-looking but badly biased output:
#   1. surface pressure passed in Pa where hPa was expected
#   2. specific humidity passed where mixing ratio in g/kg was expected
# and the v1.4.1 fix to the input guards, which checked only the first grid
# column and so rejected valid ERA5 data whose first row (90N) is dry in winter.
#
# Confirms that a dataset giving surface pressure in Pa and one giving it in
# hPa now produce identical results, and that PI lands in a sane range.
# Needs no pip install: the repo root goes first on sys.path, which also means
# it always exercises the working copy rather than an installed release.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, xarray as xr, tcpyVPI
from tcpyVPI.vpigpiv_module import calculate_potential_intensity, calculate_entropy_deficit

print("tcpyVPI version:", tcpyVPI.__version__)
print("loaded from    :", tcpyVPI.__file__)

lev = np.array([1000,975,950,925,900,850,800,750,700,650,600,550,500,
                450,400,350,300,250,200,150,100,70,50], float)
TC  = np.array([26.5,25.2,23.9,22.6,21.3,18.6,15.8,12.8,9.6,6.1,2.3,-1.9,-6.6,
                -11.9,-18.0,-25.0,-33.3,-43.5,-56.0,-70.5,-79.5,-70.0,-62.0])
RH  = np.array([.85,.85,.84,.82,.80,.72,.62,.55,.50,.45,.42,.40,.38,
                .35,.30,.25,.20,.15,.10,.05,.02,.01,.01])
esl = 6.112*np.exp(17.67*TC/(TC+243.5)); e = RH*esl
rv  = 0.622*e/(lev-e); q = rv/(1.0+rv)
lat, lon = np.array([15.]), np.array([300.])
c2 = {'latitude': lat, 'longitude': lon}
c3 = {'level': lev, **c2}


def build(sp_val, sp_units):
    return xr.Dataset({
        'SSTK': xr.DataArray(np.full((1, 1), 300.15),
                             dims=['latitude', 'longitude'], coords=c2),
        'SP':   xr.DataArray(np.full((1, 1), sp_val),
                             dims=['latitude', 'longitude'], coords=c2,
                             attrs={'units': sp_units}),
        'T':    xr.DataArray((TC + 273.15)[:, None, None] * np.ones((1, 1, 1)),
                             dims=['level', 'latitude', 'longitude'], coords=c3),
        'Q':    xr.DataArray(q[:, None, None] * np.ones((1, 1, 1)),
                             dims=['level', 'latitude', 'longitude'], coords=c3),
    })


def scalar(x):
    """Pull a Python float out of a single-point DataArray.

    float() on a size-1 array with ndim > 0 is a hard error on newer numpy,
    so ravel explicitly rather than relying on implicit conversion.
    """
    a = np.asarray(x).ravel()
    assert a.size == 1, f"expected a single grid point, got shape {np.shape(x)}"
    return float(a[0])


out = {}
for lbl, ds in [('SP in Pa ', build(101000., 'Pa')),
                ('SP in hPa', build(1010., 'hPa'))]:
    PI, asdeq = calculate_potential_intensity(ds, V_reduc=1.0, verbose=False)
    Chi = calculate_entropy_deficit(ds, asdeq, verbose=False)
    out[lbl] = (scalar(PI), scalar(asdeq), scalar(Chi))
    print(f"  {lbl}:  PI={out[lbl][0]:6.2f} m/s   asdeq={out[lbl][1]:6.2f}   Chi={out[lbl][2]:.4f}")

a, b = out['SP in Pa '], out['SP in hPa']
assert abs(a[0] - b[0]) < 1e-6, "FAIL: Pa and hPa inputs disagree -> to_hPa() not applied"
assert 55.0 < a[0] < 75.0, f"FAIL: PI={a[0]:.1f} out of sane range (pre-fix gave ~80.5)"

# The input guards should reject the exact mistake that shipped in v1.0.1:
# surface pressure still in Pa but labelled hPa, so to_hPa() cannot catch it.
mislabelled = build(101000., 'hPa')
try:
    calculate_potential_intensity(mislabelled, V_reduc=1.0, verbose=False)
    raise AssertionError("FAIL: validate_pi_inputs() did not reject Pa mislabelled as hPa")
except ValueError as exc:
    assert 'surface pressure' in str(exc), f"unexpected error: {exc}"

# --- v1.4.1: the guards must sample the whole domain, not the first column ---
# Through v1.4.0 the checks looked only at index 0 of every dimension. ERA5 runs
# latitude 90 -> -90, so that is the North Pole, whose Dec-Mar mixing ratio is
# often below the 0.5 g/kg floor (0.40 g/kg max in the Jan 1979 monthly mean),
# and valid input was rejected as "mixing ratio implausibly small".
from tcpyVPI.vpigpiv_module import validate_pi_inputs, _sample_profile

rv_gkg_max = 1000.0 * float(np.max(q / (1.0 - q)))
q_polar = q * (0.30 / rv_gkg_max)          # same shape, ~0.3 g/kg at most
lat2 = np.array([90., 0.])                  # ERA5 order: pole first
c2b = {'latitude': lat2, 'longitude': lon}
c3b = {'level': lev, **c2b}


def build_two_rows(q_scale=1.0):
    qq = np.stack([q_polar, q], axis=1)[:, :, None] * q_scale
    return xr.Dataset({
        'SSTK': xr.DataArray(np.full((2, 1), 300.15),
                             dims=['latitude', 'longitude'], coords=c2b),
        'SP':   xr.DataArray(np.full((2, 1), 101000.),
                             dims=['latitude', 'longitude'], coords=c2b,
                             attrs={'units': 'Pa'}),
        'T':    xr.DataArray((TC + 273.15)[:, None, None] * np.ones((1, 2, 1)),
                             dims=['level', 'latitude', 'longitude'], coords=c3b),
        'Q':    xr.DataArray(qq, dims=['level', 'latitude', 'longitude'],
                             coords=c3b),
    })


ds2 = build_two_rows()

# Confirm the fixture still reproduces the v1.4.0 failure: the first column alone
# is dry enough to trip the floor.
first_col_rv = 1000.0 * ds2['Q'].isel(latitude=0, longitude=0).values
first_col_rv = first_col_rv / (1.0 - first_col_rv / 1000.0)
try:
    validate_pi_inputs(np.array([27.]), np.array([1010.]), lev,
                       TC, first_col_rv)
    raise AssertionError("fixture no longer reproduces the v1.4.0 polar-column failure")
except ValueError as exc:
    assert 'implausibly small' in str(exc), f"unexpected error: {exc}"

# The real pipeline must accept the dataset, and the sample must span the grid.
try:
    calculate_potential_intensity(ds2, V_reduc=1.0, verbose=False)
except ValueError as exc:
    raise AssertionError(f"FAIL: valid data with a dry first column was rejected: {exc}")
assert _sample_profile(ds2['Q']).shape == (len(lev), 2, 1), \
    "FAIL: _sample_profile() no longer spans the horizontal domain"

# ...while a genuine kg/kg mistake (the *1000 lost) is still caught everywhere.
try:
    calculate_potential_intensity(build_two_rows(q_scale=1e-3), V_reduc=1.0, verbose=False)
    raise AssertionError("FAIL: validate_pi_inputs() did not reject mixing ratio in kg/kg")
except ValueError as exc:
    assert 'implausibly small' in str(exc), f"unexpected error: {exc}"

print("\n  PASS - unit handling is consistent, PI is in range, and the input")
print("         guards reject mislabelled surface pressure.")
print("         They also sample the whole grid, so a dry polar first column")
print("         (ERA5 Dec-Mar) is no longer mistaken for a kg/kg unit error.")
print("  (pre-fix v1.0.1 gave PI = 80.46 m/s with tcpyPI convergence flag IFL=0)")
