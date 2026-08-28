# --- tcpyVPI 1.0.2 fix verification: no data download required ---
#
# Run this AFTER installing the local working copy (pip install -e .),
# not the PyPI release. Confirms:
#   1. the version actually imported is 1.0.2 from this repo
#   2. surface pressure in Pa and in hPa now give identical results
#   3. PI lands in a sane range (v1.0.1 gave ~80.5 m/s here, with IFL=0)

import os
import sys

# Always test the working copy in this repo, never a pip-installed tcpyVPI.
# Putting the repo root first also guarantees we don't silently test an older
# release that happens to be installed in the environment.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, xarray as xr, tcpyVPI
from tcpyVPI.vpigpiv_module import calculate_potential_intensity, calculate_entropy_deficit

print("tcpyVPI version:", tcpyVPI.__version__, "  <-- must be 1.0.2")
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
print("\n  PASS - unit handling is consistent and PI is in the expected range.")
print("  (pre-fix v1.0.1 gave PI = 80.46 m/s with tcpyPI convergence flag IFL=0)")
