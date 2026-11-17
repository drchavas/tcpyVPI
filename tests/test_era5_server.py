"""Live check that NCAR's ERA5 server still serves the files tcpyVPI expects.

Opens the metadata of one monthly and one hourly file through the package's
own URL builder and fallback, and checks the variable is there. This is the
check that notices when NCAR moves the data, as the 2026 RDA -> GDEX migration
did.

Network access is required, and the hourly server is slow (~30 s per file), so
it is skipped unless TCPYVPI_NETWORK_TESTS=1:

    TCPYVPI_NETWORK_TESTS=1 python -m pytest -m network tests/test_era5_server.py -v

The weekly GitHub Actions workflow .github/workflows/era5-server.yml runs it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tcpyVPI.era5_loader as e

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get('TCPYVPI_NETWORK_TESTS') != '1',
                       reason='network test; set TCPYVPI_NETWORK_TESTS=1 to run'),
]


def _check(ds, var, expected_url):
    assert var in ds.data_vars, f"{var} missing from {expected_url}; found {list(ds.data_vars)}"
    assert ds[var].sizes.get('latitude') == 721 and ds[var].sizes.get('longitude') == 1440, \
        f"unexpected grid for {var}: {dict(ds[var].sizes)}"


def test_monthly_file_opens():
    """Monthly SSTK 2022: e5.moda.an.sfc.128_034_sstk.ll025sc.2022010100_2022120100.nc"""
    url_for = lambda b, f: e._get_monthly_mean_url('SSTK', 2022, True, b, f)
    ds = e._open_era5('monthly', url_for)
    _check(ds, 'SSTK', url_for(*e._working['monthly']))
    assert ds.sizes['time'] == 12
    print("monthly OK at", e._working['monthly'])


def test_hourly_file_opens():
    """Hourly T 2022-09-01: e5.oper.an.pl.128_130_t.ll025sc.2022090100_2022090123.nc"""
    url_for = lambda b, f: e._get_hourly_pressure_url('T', 2022, 9, 1, b, f)
    ds = e._open_era5('hourly', url_for)
    _check(ds, 'T', url_for(*e._working['hourly']))
    assert ds.sizes['time'] == 24
    print("hourly OK at", e._working['hourly'])


def test_default_location_is_first():
    """Warn (fail) if the files only open at a fallback, so the default can be updated."""
    if 'monthly' not in e._working or 'hourly' not in e._working:
        pytest.skip('run with the other tests in this file')
    for kind in ('monthly', 'hourly'):
        assert e._working[kind] == e._DEFAULT_CANDIDATES[kind][0], (
            f"{kind} data opened only at fallback {e._working[kind]}; "
            f"update the default location in tcpyVPI/era5_loader.py")
