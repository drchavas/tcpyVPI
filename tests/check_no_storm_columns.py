# --- tcpyVPI no-storm column regression check: no data download required ---
#
# Guards the v1.4.2 handling of ocean columns where no storm is possible:
#   1. no ventilation (VWS = 0) gives VI = 0 and vPI = PI exactly
#   2. a fully ventilated column (VI > VI_max) gives vPI = 0 and GPIv = 0, not
#      NaN, so NaN-skipping means (climatologies, daily means) count it as zero
#   3. a land column (SST missing) stays NaN everywhere
#   4. a column with no heat engine (tcpyPI returns vmax > 0 with To >= SST)
#      gets PI = 0, vPI = 0, GPIv = 0
#   5. cold ocean (SST <= 5 C, where tcpyPI returns no PI) gets PI = 0,
#      vPI = 0, GPIv = 0, as pcmin.m in Chavas et al. (2025) does (v1.4.3)
#   6. no output field contains +/-inf
# Needs no pip install: the repo root goes first on sys.path, so it always
# exercises the working copy rather than an installed release.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, xarray as xr, tcpyVPI
import tcpyVPI.vpigpiv_module as m

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

# Three latitude rows, each its own case; two identical longitudes so the grid
# spacing is measurable.
CALM, SHEARED, LAND, COLD = 0, 1, 2, 3
lat = np.array([16., 15., 14., 13.])
lon = np.array([300., 302.])
c2 = {'latitude': lat, 'longitude': lon}
c3 = {'level': lev, **c2}
shape2, shape3 = (4, 2), (len(lev), 4, 2)

sst = np.full(shape2, 300.15)
sst[LAND] = np.nan
sst[COLD] = 273.15 + 3.0                  # 3 C, below tcpyPI's 5 C cutoff
u = np.zeros(shape3)
u[lev == 200, SHEARED, :] = 60.0           # 60 m/s of 850-200 hPa shear

ds = xr.Dataset({
    'SSTK': xr.DataArray(sst, dims=['latitude', 'longitude'], coords=c2),
    'SP':   xr.DataArray(np.full(shape2, 101000.), dims=['latitude', 'longitude'],
                         coords=c2, attrs={'units': 'Pa'}),
    'T':    xr.DataArray((TC + 273.15)[:, None, None] * np.ones(shape3),
                         dims=['level', 'latitude', 'longitude'], coords=c3),
    'Q':    xr.DataArray(q[:, None, None] * np.ones(shape3),
                         dims=['level', 'latitude', 'longitude'], coords=c3),
    'U':    xr.DataArray(u, dims=['level', 'latitude', 'longitude'], coords=c3),
    'V':    xr.DataArray(np.zeros(shape3), dims=['level', 'latitude', 'longitude'], coords=c3),
    'VO':   xr.DataArray(np.full(shape3, 1e-5), dims=['level', 'latitude', 'longitude'], coords=c3),
})

FIELDS = ['PI', 'Chi', 'ventilation_index', 'vPI', 'GPIv']


def no_infs(r, label):
    for k in FIELDS:
        assert not np.isinf(r[k].values).any(), f"FAIL ({label}): {k} contains +/-inf"


r = m.compute_gpiv_from_dataset(ds, verbose=False)
no_infs(r, 'real tcpyPI')
row = lambda k, i: r[k].isel(latitude=i).values

# 1. No ventilation: VI = 0 is valid and leaves PI unreduced.
assert np.all(row('PI', CALM) > 30), f"FAIL: unexpected PI {row('PI', CALM)}"
assert np.all(row('ventilation_index', CALM) == 0), "FAIL: VWS = 0 should give VI = 0"
assert np.allclose(row('vPI', CALM), row('PI', CALM), rtol=1e-9), \
    "FAIL: VI = 0 should give vPI = PI"
assert np.all(row('GPIv', CALM) > 0), "FAIL: calm column should have GPIv > 0"
print(f"  calm     : PI = vPI = {row('PI', CALM)[0]:.2f} m/s, VI = 0, GPIv > 0")

# 2. Fully ventilated: zero, not missing.
assert np.all(row('ventilation_index', SHEARED) > m.DEFAULT_VI_MAX), \
    "fixture no longer exceeds VI_max; strengthen the shear"
assert np.all(row('vPI', SHEARED) == 0), f"FAIL: ventilated vPI is {row('vPI', SHEARED)}, expected 0"
assert np.all(row('GPIv', SHEARED) == 0), f"FAIL: ventilated GPIv is {row('GPIv', SHEARED)}, expected 0"
print(f"  sheared  : VI = {row('ventilation_index', SHEARED)[0]:.3f} > {m.DEFAULT_VI_MAX} -> vPI = 0, GPIv = 0")

# 3. Land stays missing.
for k in FIELDS:
    assert np.all(np.isnan(row(k, LAND))), f"FAIL: land column {k} should be NaN"
print("  land     : every field NaN")

# 5. Cold ocean: zero, like pcmin.m, rather than tcpyPI's missing value.
assert np.all(np.isnan([m.pi(3.0, 1010., lev, TC, rv * 1000.)[0]])), \
    "fixture assumption broken: tcpyPI now computes PI below 5 C"
for k in ['PI', 'vPI', 'GPIv']:
    assert np.all(row(k, COLD) == 0), f"FAIL: cold ocean {k} is {row(k, COLD)}, expected 0"
assert np.all(np.isfinite(row('Chi', COLD))), "FAIL: cold ocean Chi should be finite"
assert np.all(np.isnan(row('ventilation_index', COLD))), "FAIL: cold ocean VI should be NaN (PI = 0)"
print("  cold     : SST 3 C -> PI = 0, vPI = 0, GPIv = 0 (tcpyPI alone gives NaN)")

# 4. No heat engine. tcpyPI is replaced by a stub that reproduces what the real
# routine returns under a shallow trade inversion: a small positive vmax with
# the outflow temperature 1 K above the SST.
real_pi = m.pi


def stub_pi(SSTC, MSL, P, TC_, R, **kw):
    if not np.isfinite(SSTC):
        return np.nan, np.nan, 0, np.nan, np.nan
    return 2.8, MSL - 1.0, 1, SSTC + 273.15 + 1.0, 970.0


m.pi = stub_pi
try:
    r = m.compute_gpiv_from_dataset(ds, verbose=False)
finally:
    m.pi = real_pi
no_infs(r, 'no heat engine')
for i in (CALM, SHEARED):
    assert np.all(r['PI'].isel(latitude=i).values == 0), "FAIL: no heat engine should give PI = 0"
    assert np.all(np.isfinite(r['Chi'].isel(latitude=i).values)), "FAIL: Chi should be finite"
    assert np.all(np.isnan(r['ventilation_index'].isel(latitude=i).values)), "FAIL: VI should be NaN where PI = 0"
    assert np.all(r['vPI'].isel(latitude=i).values == 0), "FAIL: no heat engine should give vPI = 0"
    assert np.all(r['GPIv'].isel(latitude=i).values == 0), "FAIL: no heat engine should give GPIv = 0"
print("  no engine: tcpyPI vmax 2.8 m/s with To > SST -> PI = 0, vPI = 0, GPIv = 0")

print("\n  PASS - no-storm ocean columns are zero rather than missing, land stays")
print("         missing, and no field contains +/-inf.")
