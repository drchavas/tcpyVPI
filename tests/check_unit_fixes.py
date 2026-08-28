# --- tcpyVPI unit-handling regression check: no data download required ---
#
# Guards the two bugs fixed in v1.1.0, where the inputs handed to tcpyPI.pi()
# had the wrong units and produced plausible-looking but badly biased output:
#   1. surface pressure passed in Pa where hPa was expected
#   2. specific humidity passed where mixing ratio in g/kg was expected
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

print("\n  PASS - unit handling is consistent, PI is in range, and the input")
print("         guards reject mislabelled surface pressure.")
print("  (pre-fix v1.0.1 gave PI = 80.46 m/s with tcpyPI convergence flag IFL=0)")
