# --- tcpyVPI outflow temperature regression check: no data download required ---
#
# Guards the v1.5.0 outflow temperature output (T_o):
#   1. return_To=False (the default) keeps the two-element return
#   2. asking for T_o leaves PI and the air-sea term bit-for-bit unchanged
#   3. T_o < SST wherever PI > 0, and T_o is NaN where PI is 0 or missing
#   4. compute_gpiv_from_dataset() outputs T_o with units K
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

# Rows: warm ocean (PI > 0), warmer ocean, cold ocean (PI = 0), land (NaN).
WARM, WARMER, COLD, LAND = range(4)
lat = np.array([16., 15., 14., 13.]); lon = np.array([300., 302.])
c2 = {'latitude': lat, 'longitude': lon}; c3 = {'level': lev, **c2}
shape2, shape3 = (4, 2), (len(lev), 4, 2)
sst = np.full(shape2, 300.15); sst[WARMER] = 302.15; sst[COLD] = 276.15; sst[LAND] = np.nan
ds = xr.Dataset({
    'SSTK': xr.DataArray(sst, dims=['latitude', 'longitude'], coords=c2),
    'SP':   xr.DataArray(np.full(shape2, 101000.), dims=['latitude', 'longitude'],
                         coords=c2, attrs={'units': 'Pa'}),
    'T':    xr.DataArray((TC + 273.15)[:, None, None] * np.ones(shape3),
                         dims=['level', 'latitude', 'longitude'], coords=c3),
    'Q':    xr.DataArray(q[:, None, None] * np.ones(shape3),
                         dims=['level', 'latitude', 'longitude'], coords=c3),
    'U':    xr.DataArray(np.zeros(shape3), dims=['level', 'latitude', 'longitude'], coords=c3),
    'V':    xr.DataArray(np.zeros(shape3), dims=['level', 'latitude', 'longitude'], coords=c3),
    'VO':   xr.DataArray(np.full(shape3, 1e-5), dims=['level', 'latitude', 'longitude'], coords=c3),
})

# 1-2. Default return unchanged; T_o request changes nothing else.
two = m.calculate_potential_intensity(ds, V_reduc=1.0, verbose=False)
assert len(two) == 2, "FAIL: default return must stay (vmax, asdeq)"
PI, asdeq, T_o = m.calculate_potential_intensity(ds, V_reduc=1.0, verbose=False, return_To=True)
for a, b, name in [(two[0], PI, 'PI'), (two[1], asdeq, 'asdeq')]:
    assert np.array_equal(a.values, b.values, equal_nan=True), f"FAIL: {name} changed when T_o requested"
print("  default return unchanged; PI and the air-sea term identical with return_To=True")

# 3. Physical consistency.
pos = PI.values > 0
assert pos[WARM].all() and pos[WARMER].all(), "fixture assumption broken: warm rows should have PI > 0"
assert np.all(T_o.values[pos] < sst[pos]), "FAIL: T_o must be colder than the SST where PI > 0"
assert np.all((T_o.values[pos] > 150) & (T_o.values[pos] < 260)), f"FAIL: implausible T_o {T_o.values[pos]}"
assert np.all(np.isnan(T_o.values[~pos])), "FAIL: T_o must be NaN where PI is 0 or missing"
assert T_o.attrs.get('units') == 'K' and T_o.attrs.get('long_name') == 'Outflow Temperature'
print(f"  T_o = {T_o.values[WARM, 0]:.1f} K < SST where PI > 0; NaN on cold ocean and land")

# 4. Pipeline output.
r = m.compute_gpiv_from_dataset(ds, verbose=False)
assert 'T_o' in r and r['T_o'].attrs.get('units') == 'K', "FAIL: compute_gpiv_from_dataset lacks T_o"
assert np.array_equal(r['T_o'].values, T_o.values, equal_nan=True), "FAIL: pipeline T_o differs"
print("  compute_gpiv_from_dataset() returns T_o (K)")

print("\n  PASS - T_o is returned on request, is physically consistent, and changes no other output.")
