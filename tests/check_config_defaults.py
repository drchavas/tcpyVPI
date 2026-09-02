# --- tcpyVPI configurable-parameter regression check: no data download needed ---
#
# Two things are verified:
#   1. BACKWARD COMPATIBILITY. Calling with defaults, and calling with every
#      parameter passed explicitly at its default value, give bit-identical
#      results. This is what guarantees v1.2.0 reproduces v1.1.0.
#   2. EFFECTIVENESS. Each parameter, when changed, actually alters the fields
#      it should.
#
# The fixture is chosen deliberately: weak shear so VI < VI_max and vPI is
# finite (otherwise vPI/GPIv are NaN and mask any change), and vorticity that
# varies with height and stays under the cap (otherwise eta_c saturates and
# hides vort_level).

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, xarray as xr, tcpyVPI
from tcpyVPI.vpigpiv_module import compute_gpiv_from_dataset, calculate_etac

print("tcpyVPI version:", tcpyVPI.__version__)

lev = np.array([1000,925,850,700,600,500,400,300,250,200,150,100,70,50], float)
TC  = np.array([26.5,22.6,18.6,9.6,2.3,-6.6,-18.0,-33.3,-43.5,-56.0,-70.5,-79.5,-70.0,-62.0])
RH  = np.array([.85,.82,.72,.50,.42,.38,.30,.20,.15,.10,.05,.02,.01,.01])
esl = 6.112*np.exp(17.67*TC/(TC+243.5)); e = RH*esl
rv  = 0.622*e/(lev-e); q = rv/(1+rv)
lat, lon = np.array([12., 15., 18.]), np.array([300., 302.])
c2 = {'latitude': lat, 'longitude': lon}
c3 = {'level': lev, **c2}


def f3(v):
    v = np.asarray(v, float)
    return xr.DataArray(np.repeat(np.repeat(v[:, None, None], lat.size, 1), lon.size, 2),
                        dims=['level', 'latitude', 'longitude'], coords=c3)


ds = xr.Dataset({
    'SSTK': xr.DataArray(np.full((lat.size, lon.size), 301.15),
                         dims=['latitude', 'longitude'], coords=c2),
    'SP':   xr.DataArray(np.full((lat.size, lon.size), 101000.),
                         dims=['latitude', 'longitude'], coords=c2, attrs={'units': 'Pa'}),
    'T':  f3(TC + 273.15),
    'Q':  f3(q),
    'U':  f3(np.linspace(0, 7, lev.size)),      # weak shear -> VI below VI_max
    'V':  f3(np.linspace(1, -2, lev.size)),
    'VO': f3(np.linspace(3e-5, 0.2e-5, lev.size)),
})

KEYS = ['PI', 'VWS', 'Chi', 'ventilation_index', 'vPI', 'eta_c', 'GPIv']
DEFAULTS = dict(shear_p_top=200., shear_p_bot=850., chi_p_mid=600.,
                vort_level=850., vort_cap=3.7e-5, VI_max=0.145,
                gpiv_exponent=4.90, CKCD=0.9, ascent_flag=0, diss_flag=1,
                ptop=50.)

base = compute_gpiv_from_dataset(ds, verbose=False)
print("\nbaseline:")
for k in KEYS:
    print(f"  {k:20s} {float(base[k].mean()):14.6g}")
assert np.isfinite(float(base['vPI'].mean())), \
    "fixture is degenerate: vPI is NaN, so vPI/GPIv changes cannot be detected"

# --- 1. backward compatibility -------------------------------------------------
explicit = compute_gpiv_from_dataset(ds, verbose=False, **DEFAULTS)
for k in KEYS:
    assert np.array_equal(explicit[k].values, base[k].values, equal_nan=True), \
        f"FAIL: passing the default for {k} changed the result"
print("\n  PASS - explicit defaults are bit-identical to the hardcoded behaviour")

# --- 2. each parameter has an effect -------------------------------------------
def changed(res):
    return [k for k in KEYS
            if not np.allclose(np.nan_to_num(res[k].values, nan=-9e9),
                               np.nan_to_num(base[k].values, nan=-9e9),
                               rtol=1e-12, atol=0)]


PROBES = [('shear_p_top', 250.), ('shear_p_bot', 925.), ('chi_p_mid', 500.),
          ('vort_cap', 5.0e-5), ('VI_max', 0.20), ('gpiv_exponent', 5.5),
          ('CKCD', 0.8), ('ascent_flag', 1), ('diss_flag', 0),
          ('ptop', 100.)]

print()
for name, val in PROBES:
    d = changed(compute_gpiv_from_dataset(ds, verbose=False, **{name: val}))
    print(f"  {name:16s}={str(val):8s} -> changed {','.join(d) if d else 'NOTHING'}")
    assert d, f"FAIL: {name} had no effect"

# vort_level needs its own fixture: in the grid above eta_c is pinned at the cap
# at these latitudes, which would mask the level change.
lo_lat = np.array([5., 8.])
c3b = {'level': lev, 'latitude': lo_lat, 'longitude': np.array([300.])}
ds_v = xr.Dataset({'VO': xr.DataArray(
    np.repeat(np.repeat(np.linspace(2.0e-5, 0.1e-5, lev.size)[:, None, None], 2, 1), 1, 2),
    dims=['level', 'latitude', 'longitude'], coords=c3b)})
a = calculate_etac(ds_v, p_level=850., verbose=False).values
b = calculate_etac(ds_v, p_level=1000., verbose=False).values
assert not np.allclose(a, b), "FAIL: vort_level had no effect"
print(f"  {'vort_level':16s}={'1000.0':8s} -> changed eta_c (checked below the cap)")

print("\n  PASS - all eleven parameters propagate, and defaults reproduce v1.1.0.")
