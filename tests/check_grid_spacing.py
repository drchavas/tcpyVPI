# --- tcpyVPI grid-spacing regression check: no data download needed ---
#
# The GPIv area term weights every grid box by a single constant dx*dy measured
# from the coordinates. This checks that the measurement is right on grids that
# are unusual but legitimate, and that grids on which it CANNOT be right fail
# loudly rather than returning a plausible-looking number.
#
# Each case below is a bug that the v1.4.0 `mean(diff(...))` implementation had.

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, xarray as xr, tcpyVPI
from tcpyVPI.vpigpiv_module import _spacing_deg, compute_gpiv_from_dataset

print("tcpyVPI version:", tcpyVPI.__version__)


def coord(values, name):
    v = np.asarray(values, float)
    if v.ndim == 0:                       # what .sel(name=scalar) leaves behind
        return xr.DataArray(v, dims=(), name=name)
    return xr.DataArray(v, dims=[name], coords={name: v}, name=name)


def spacing(values, name, wrap=False):
    """Return (value, n_warnings)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = _spacing_deg(coord(values, name), name, wrap=wrap)
    return s, len(w)


# --- 1. correct spacing, no warning, on every legitimate grid -----------------
ROLLED = np.concatenate([np.arange(0, 180, 0.25), np.arange(-180, 0, 0.25)])
LEGIT = [
    ("0.25 deg ascending lon",  np.arange(0, 360, 0.25),      'longitude', True,  0.25),
    ("rolled lon (0..180,-180..0)", ROLLED,                   'longitude', True,  0.25),
    ("ERA5 lat, descending",    np.arange(90, -90.01, -0.25), 'latitude',  False, 0.25),
    ("2 deg lat",               np.arange(90, -90.01, -2.0),  'latitude',  False, 2.00),
    ("4 deg lon",               np.arange(0, 360, 4.0),       'longitude', True,  4.00),
]
print()
for name, v, cname, wrap, want in LEGIT:
    got, nw = spacing(v, cname, wrap)
    print(f"  {name:30s} -> {got:7.4f} deg  ({nw} warnings)")
    assert np.isclose(got, want), f"FAIL: {name}: expected {want}, got {got}"
    assert nw == 0, f"FAIL: {name} warned on a perfectly regular grid"
print("\n  PASS - regular grids measure correctly and silently, including a rolled longitude")

# The rolled case is the one the mean got badly wrong. Because the old code took
# the absolute value OUTSIDE the mean, the single ~-360 step nearly cancelled all
# the +0.25 steps: dx came out ~0.00017 rather than 0.25, low by ~1400x.
old = float(np.abs(xr.DataArray(ROLLED, dims=['longitude']).diff('longitude').mean()))
assert old < 0.01, "fixture no longer reproduces the v1.4.0 rolled-longitude failure"
print(f"         (v1.4.0 gave {old:.6f} deg here, low by a factor of {0.25/old:.0f})")

# --- 2. degenerate coordinates raise ------------------------------------------
# Each of these returned a finite, wrong number or NaN in v1.4.0.
print()
DEGENERATE = [
    ("0-d, i.e. after .sel(latitude=0.5)", 0.5),
    ("single-element coordinate",          [0.5]),
    ("duplicate values",                   [1.0, 1.0, 1.0]),
]
for name, v in DEGENERATE:
    try:
        got = _spacing_deg(coord(v, 'latitude'), 'latitude')
        raise AssertionError(f"FAIL: {name} returned {got} instead of raising")
    except ValueError:
        print(f"  {name:38s} -> ValueError")

# the 0-d case specifically: .diff() returns the VALUE, so dy became the latitude
zerod = xr.Dataset({'x': (('latitude',), np.ones(3))},
                   coords={'latitude': [0.5, 1.0, 1.5]}).sel(latitude=0.5)['latitude']
assert float(np.abs(zerod.diff('latitude').mean())) == 0.5, \
    "fixture no longer reproduces the v1.4.0 0-d behaviour"
print("  (v1.4.0 returned 0.5 for the 0-d case - the latitude itself, not a spacing)")
print("\n  PASS - degenerate coordinates raise instead of returning a wrong number")

# --- 3. non-uniform spacing warns ---------------------------------------------
got, nw = spacing([0, .25, .5, .75, 5., 5.25, 5.5], 'longitude', wrap=True)
assert nw == 1, "FAIL: a non-uniform coordinate did not warn"
assert np.isclose(got, 0.25), f"FAIL: median should still be 0.25, got {got}"
print(f"\n  PASS - non-uniform spacing warns (and the median is still robust: {got})")

# --- 4. end-to-end: GPIv per square degree is grid-independent -----------------
lev = np.array([1000,925,850,700,600,500,400,300,250,200,150,100,70,50], float)
TC  = np.array([26.5,22.6,18.6,9.6,2.3,-6.6,-18.0,-33.3,-43.5,-56.0,-70.5,-79.5,-70.0,-62.0])
RH  = np.array([.85,.82,.72,.50,.42,.38,.30,.20,.15,.10,.05,.02,.01,.01])
esl = 6.112 * np.exp(17.67 * TC / (TC + 243.5))
rv  = 0.622 * (RH * esl) / (lev - RH * esl)
q   = rv / (1 + rv)


def fixture(step, roll=False):
    lat = np.arange(16, -16.01, -step)
    # A true 360 deg wrap for the roll case: 0..180 then -180..0, as produced by
    # Dataset.roll() or by subsetting onto a -180..180 convention. The seam is a
    # single ~-360 step, which is exactly what the unwrap is there to absorb.
    lon = (np.concatenate([np.arange(0, 180, step), np.arange(-180, 0, step)])
           if roll else np.arange(0, 360, step))
    nz = lev.size

    def f3(prof):
        return np.broadcast_to(np.asarray(prof, float)[:, None, None],
                               (nz, lat.size, lon.size)).copy()

    ds = xr.Dataset({
        'SSTK': (('latitude', 'longitude'), np.full((lat.size, lon.size), 301.15)),
        'SP':   (('latitude', 'longitude'), np.full((lat.size, lon.size), 101000.)),
        'T':  (('level', 'latitude', 'longitude'), f3(TC + 273.15)),
        'Q':  (('level', 'latitude', 'longitude'), f3(q)),
        'U':  (('level', 'latitude', 'longitude'), f3(np.linspace(0, 7, nz))),
        'V':  (('level', 'latitude', 'longitude'), f3(np.linspace(1, -2, nz))),
        'VO': (('level', 'latitude', 'longitude'), f3(np.linspace(3e-5, 0.2e-5, nz))),
    }, coords={'level': lev, 'latitude': lat, 'longitude': lon})
    ds['SP'].attrs['units'] = 'Pa'
    return ds


print()
per_deg2 = {}
for step in [4.0, 2.0, 1.0, 0.25]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = float(compute_gpiv_from_dataset(fixture(step), verbose=False)['GPIv'].mean())
    per_deg2[step] = g / step**2
    print(f"  {step:5.2f} deg   mean GPIv = {g:11.5g}   per deg^2 = {g/step**2:.6g}   "
          f"({len(w)} warnings)")
    assert not w, f"FAIL: warned on a uniform {step} deg grid"

spread = (max(per_deg2.values()) - min(per_deg2.values())) / np.mean(list(per_deg2.values()))
assert spread < 0.05, f"FAIL: GPIv per deg^2 varies by {spread:.1%} across grids"
print(f"\n  PASS - GPIv per square degree is grid-independent to {spread:.1%} "
      f"(residual is cos(lat) sampling, not the area term)")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rolled = float(compute_gpiv_from_dataset(fixture(2.0, roll=True), verbose=False)['GPIv'].mean())
clean = float(compute_gpiv_from_dataset(fixture(2.0), verbose=False)['GPIv'].mean())
assert np.isclose(rolled, clean), f"FAIL: rolled lon {rolled} != clean {clean}"
assert not w, "FAIL: a wrapped longitude warned; the 360 deg unwrap is not working"
print(f"  PASS - a wrapped longitude gives the same GPIv as the equivalent clean "
      f"grid, silently")

try:
    compute_gpiv_from_dataset(fixture(2.0).sel(latitude=0.0, method='nearest'), verbose=False)
    raise AssertionError("FAIL: a point-selected dataset did not raise")
except ValueError:
    print("  PASS - a point-selected dataset raises instead of inventing a dy")

# --- 5. calibration metadata --------------------------------------------------
# GPIv sums are grid-consistent but NOT resolution-invariant: the ~5th power is
# convex, so a fine grid sums higher than the same fields averaged to 2 deg.
# There is deliberately no runtime warning (it would fire on run_vpigpiv()'s own
# default 0.25 deg path), so the metadata is the only durable signal - it has to
# be right, and it has to survive a netCDF round-trip.
print()
for step, expect in [(2.0, 'yes'), (0.25, 'no'), (4.0, 'no')]:
    a = compute_gpiv_from_dataset(fixture(step), verbose=False)['GPIv'].attrs
    assert a['calibrated'] == expect, \
        f"FAIL: {step} deg reported calibrated={a['calibrated']}, expected {expect}"
    assert np.isclose(a['grid_dx_deg'], step) and np.isclose(a['grid_dy_deg'], step), \
        f"FAIL: {step} deg recorded grid ({a['grid_dx_deg']}, {a['grid_dy_deg']})"
    assert a['calibration_grid_deg'] == 2.0
    print(f"  {step:5.2f} deg -> calibrated='{a['calibrated']}', "
          f"grid recorded as ({a['grid_dx_deg']:g}, {a['grid_dy_deg']:g})")

import tempfile
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, 'roundtrip.nc')
    compute_gpiv_from_dataset(fixture(0.25), verbose=False).to_netcdf(path)
    with xr.open_dataset(path) as reopened:
        a = reopened['GPIv'].attrs
        assert a['calibrated'] == 'no', "FAIL: calibration flag lost through netCDF"
        assert np.isclose(a['grid_dx_deg'], 0.25), "FAIL: grid spacing lost through netCDF"
        assert 'Coarsen' in a['comment'], "FAIL: the caveat comment did not survive"
print("\n  PASS - calibration metadata is correct and survives a netCDF round-trip")

print("\n  PASS - grid spacing is measured robustly, and unmeasurable grids fail loudly.")
