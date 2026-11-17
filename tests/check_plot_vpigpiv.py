# --- tcpyVPI plotting regression check: offline, no data download ---
#
# Guards the v1.5.1 fix to plot_vpigpiv(). Since v1.4.2 the ventilation index
# can be exactly 0 (no shear, or chi floored at 0), and the VI map's LogNorm()
# autoscaled to vmin = 0, so matplotlib raised "Invalid vmin or vmax". This
# draws every figure plot_vpigpiv() makes, with the Agg backend, on a small
# dataset that contains a VI = 0 cell.
#
# Skipped (exit 0) if matplotlib or cartopy is not installed: plotting is an
# optional extra (pip install "tcpyVPI[plot]"). Coastlines are stubbed out so
# cartopy does not try to download Natural Earth data.
# Needs no pip install of tcpyVPI: the repo root goes first on sys.path.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes
except ImportError as exc:
    print(f"  SKIP - plotting libraries not installed ({exc}); pip install 'tcpyVPI[plot]'")
    sys.exit(0)

import warnings
import numpy as np, xarray as xr, tcpyVPI
import tcpyVPI.vpigpiv_module as m

warnings.filterwarnings('ignore')
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

# 4 x 4 grid. Shear rises with longitude (0, 5, 15, 60 m/s), so the VI map has
# exactly 0, values inside the plotted 0.1-10 range, and values above VI_max.
# One land cell (SST missing) as well.
lat = np.array([20., 15., 10., 5.])
lon = np.array([300., 305., 310., 315.])
c2 = {'latitude': lat, 'longitude': lon}; c3 = {'level': lev, **c2}
shape2, shape3 = (4, 4), (len(lev), 4, 4)
sst = np.full(shape2, 300.15); sst[0, 3] = np.nan
u = np.zeros(shape3)
u[lev == 200] = np.array([0., 5., 15., 60.])[None, None, :]
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
r = m.compute_gpiv_from_dataset(ds, verbose=False)
vi = r['ventilation_index'].values
assert np.nanmin(vi) == 0, f"fixture assumption broken: no VI = 0 cell (min {np.nanmin(vi)})"
assert np.nanmax(vi) > m.DEFAULT_VI_MAX, "fixture assumption broken: no ventilated cell"
print(f"  fixture: VI from {np.nanmin(vi):g} to {np.nanmax(vi):.3g}, one land cell")

# No coastline download, no window.
real_coast, real_show = GeoAxes.coastlines, plt.show
GeoAxes.coastlines = lambda self, *a, **k: None
plt.show = lambda *a, **k: None
plt.close('all')
try:
    try:
        m.plot_vpigpiv(ds, r, 2022, 9, 1, 12)
        for num in plt.get_fignums():
            plt.figure(num).canvas.draw()          # force full rendering
    except ValueError as exc:
        raise AssertionError(f"FAIL: plot_vpigpiv raised {exc!r} on data with VI = 0")
    nfig = len(plt.get_fignums())
    assert nfig == 4, f"FAIL: expected 4 figures, got {nfig}"
    # The VI map's colour scale must be the fixed plotted range, not the data's.
    vi_ax = plt.figure(plt.get_fignums()[0]).axes[0]
    norms = [c.norm for c in vi_ax.collections if getattr(c, 'norm', None) is not None]
    assert norms and np.isclose(norms[0].vmin, 0.1) and np.isclose(norms[0].vmax, 10.0), \
        f"FAIL: VI colour scale should span 0.1-10, got {[(n.vmin, n.vmax) for n in norms]}"
finally:
    GeoAxes.coastlines, plt.show = real_coast, real_show
    plt.close('all')
print(f"  plot_vpigpiv drew all {nfig} figures; VI colour scale fixed at 0.1-10")

print("\n  PASS - plot_vpigpiv works when the ventilation index is exactly 0.")
