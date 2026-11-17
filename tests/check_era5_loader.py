# --- tcpyVPI ERA5 loader regression check: offline, no data download ---
#
# Guards the v1.5.0 loader changes made after NCAR's RDA -> GDEX migration
# renamed the monthly folder (d633001_nc -> d633001):
#   1. the URL builders produce the expected GDEX URLs
#   2. the fallback moves on after "file not found", retries transient errors,
#      stops at the first location that works, and remembers it
#   3. if every location fails, one error lists every URL tried
#   4. set_era5_thredds_base() and TCPYVPI_ERA5_THREDDS override the location
# The opener is replaced by a stub, so nothing touches the network.
# Needs no pip install: the repo root goes first on sys.path, so it always
# exercises the working copy rather than an installed release.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tcpyVPI
import tcpyVPI.era5_loader as e

print("tcpyVPI version:", tcpyVPI.__version__)
print("loaded from    :", tcpyVPI.__file__)

os.environ.pop(e.ERA5_THREDDS_ENV_VAR, None)
tcpyVPI.set_era5_thredds_base()

GDEX = "https://tds.gdex.ucar.edu/thredds/dodsC/files/g"
RDA = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g"

# 1. URL builders -----------------------------------------------------------
expected = {
    e._get_monthly_mean_url('SSTK', 2022, True):
        f"{GDEX}/d633001/e5.moda.an.sfc/2022/e5.moda.an.sfc.128_034_sstk.ll025sc.2022010100_2022120100.nc",
    e._get_monthly_mean_url('U', 2022, False):
        f"{GDEX}/d633001/e5.moda.an.pl/2022/e5.moda.an.pl.128_131_u.ll025uv.2022010100_2022120100.nc",
    e._get_hourly_surface_url('SSTK', 2022, 9):
        f"{GDEX}/d633000/e5.oper.an.sfc/202209/e5.oper.an.sfc.128_034_sstk.ll025sc.2022090100_2022093023.nc",
    e._get_hourly_pressure_url('T', 2022, 9, 1):
        f"{GDEX}/d633000/e5.oper.an.pl/202209/e5.oper.an.pl.128_130_t.ll025sc.2022090100_2022090123.nc",
    e._get_monthly_mean_url('SSTK', 2022, True, RDA, 'd633001_nc'):
        f"{RDA}/d633001_nc/e5.moda.an.sfc/2022/e5.moda.an.sfc.128_034_sstk.ll025sc.2022010100_2022120100.nc",
}
for got, want in expected.items():
    assert got == want, f"FAIL: URL builder\n  got  {got}\n  want {want}"
print("  URL builders produce the GDEX paths (and accept an explicit base/folder)")


# 2. Fallback and retries ---------------------------------------------------
class StubOpener:
    """Fail each URL according to a rule; record every call."""
    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def __call__(self, url):
        self.calls.append(url)
        err = self.rule(url, self.calls.count(url))
        if err is not None:
            raise err
        return f"dataset from {url}"


def not_found(url):
    return OSError(-90, "NetCDF: file not found", url)


def io_failure(url):
    return OSError(-68, "NetCDF: I/O failure", url)


real_open, real_sleep = e._open_dataset, e._sleep
waits = []
e._sleep = waits.append
url_for = lambda b, f: e._get_monthly_mean_url('SSTK', 2022, True, b, f)
try:
    # GDEX path missing, old RDA alias works: one try at GDEX, then RDA.
    e._working.clear()
    stub = StubOpener(lambda url, n: not_found(url) if url.startswith(GDEX) else None)
    e._open_dataset = stub
    ds = e._open_era5('monthly', url_for)
    assert ds.startswith(f"dataset from {RDA}/d633001/"), ds
    assert len(stub.calls) == 2 and not waits, f"FAIL: 'not found' must not be retried: {stub.calls}"
    assert e._working['monthly'] == (RDA, 'd633001'), "FAIL: working location not remembered"
    # The remembered location is tried first next time.
    stub.calls.clear()
    e._open_dataset = stub
    e._open_era5('monthly', url_for)
    assert len(stub.calls) == 1 and stub.calls[0].startswith(RDA), \
        f"FAIL: remembered location not tried first: {stub.calls}"
    print("  'file not found' moves on without retrying; the working location is remembered")

    # Transient error twice, then success at the same location.
    e._working.clear(); waits.clear()
    stub = StubOpener(lambda url, n: io_failure(url) if n <= 2 else None)
    e._open_dataset = stub
    ds = e._open_era5('monthly', url_for)
    assert ds.startswith(f"dataset from {GDEX}/") and len(stub.calls) == 3, stub.calls
    assert waits == list(e.BACKOFF_SECONDS), f"FAIL: expected backoff {e.BACKOFF_SECONDS}, got {waits}"
    print(f"  transient errors are retried with backoff {e.BACKOFF_SECONDS} s, then succeed")

    # A date like 20220404 in the URL must not be read as an HTTP 404.
    assert e._classify(io_failure("x/e5.oper.an.pl.20220404.nc"), "x/e5.oper.an.pl.20220404.nc") == 'transient'
    assert e._classify(RuntimeError("HTTP 404 Not Found"), "") == 'not_found'
    print("  error classification ignores digits in the URL")

    # Everything fails: one error that lists every URL tried.
    e._working.clear(); waits.clear()
    e._open_dataset = StubOpener(lambda url, n: not_found(url))
    try:
        e._open_era5('monthly', url_for)
        raise AssertionError("FAIL: expected ERA5DataUnavailableError")
    except tcpyVPI.ERA5DataUnavailableError as exc:
        msg = str(exc)
        for base, folder in e._DEFAULT_CANDIDATES['monthly']:
            assert url_for(base, folder) in msg, f"FAIL: error does not list {url_for(base, folder)}"
        for needle in ('set_era5_thredds_base', e.ERA5_THREDDS_ENV_VAR, 'gdex.ucar.edu/datasets/d633001'):
            assert needle in msg, f"FAIL: error message lacks {needle!r}"
    print("  when every location fails, one error lists each URL and how to override")

    # 4. Overrides ----------------------------------------------------------
    NEW = "https://example.org/thredds/dodsC/files/g"
    tcpyVPI.set_era5_thredds_base(NEW + "/")
    assert e.get_era5_thredds_candidates('monthly') == [(NEW, 'd633001')]
    assert e.get_era5_thredds_candidates('hourly') == [(NEW, 'd633000')]
    assert e._get_hourly_pressure_url('T', 2022, 9, 1).startswith(f"{NEW}/d633000/")
    stub = StubOpener(lambda url, n: not_found(url))
    e._open_dataset = stub
    try:
        e._open_era5('monthly', url_for)
    except tcpyVPI.ERA5DataUnavailableError:
        pass
    assert len(stub.calls) == 1 and stub.calls[0].startswith(NEW), \
        f"FAIL: with an override only the override is tried: {stub.calls}"
    tcpyVPI.set_era5_thredds_base()
    os.environ[e.ERA5_THREDDS_ENV_VAR] = NEW
    assert e.get_era5_thredds_candidates('monthly') == [(NEW, 'd633001')], "FAIL: env var ignored"
    os.environ.pop(e.ERA5_THREDDS_ENV_VAR)
    assert e.get_era5_thredds_candidates('monthly')[0] == (GDEX, 'd633001'), "FAIL: reset to defaults"
    print("  set_era5_thredds_base() and TCPYVPI_ERA5_THREDDS override the location; reset works")
finally:
    e._open_dataset, e._sleep = real_open, real_sleep
    e._working.clear()
    tcpyVPI.set_era5_thredds_base()

print("\n  PASS - URLs, fallback, retries, error message and overrides behave as intended.")
