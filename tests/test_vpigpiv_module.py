"""
Test script for the ``tcvpigpiv`` package.

This script demonstrates how to use the package with both monthly mean
and hourly ERA5 data. It includes tests for:
- Monthly mean GPIv computation
- Hourly GPIv computation
- Climatology calculation
- Anomaly computation

By default, it uses September 2022 as a test case for monthly data
and August 15, 2020 for hourly data.

Authors:
    Dan Chavas (2025) - Original tests
    Jose Ocegueda Sanchez (2025) - Hourly and climatology tests
"""

# -----------------------------------------------------------------------------
# User-defined inputs
YEAR_MONTHLY = 2022
MONTH_MONTHLY = 9

YEAR_HOURLY = 2020
MONTH_HOURLY = 8
DAY_HOURLY = 15
HOUR_HOURLY = 12
# -----------------------------------------------------------------------------

import os
import sys
import numpy as np

# Add the project root to Python path
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


def test_monthly_gpiv():
    """Test GPIv computation with monthly mean data."""
    print("\n" + "="*60)
    print("Testing Monthly Mean GPIv Computation")
    print("="*60)
    
    try:
        from tcvpigpiv import run_vpigpiv
    except ImportError:
        from tcvpigpiv.vpigpiv_module import run_vpigpiv
    
    print(f"\nComputing GPIv for {YEAR_MONTHLY:04d}-{MONTH_MONTHLY:02d}...")
    results = run_vpigpiv(YEAR_MONTHLY, MONTH_MONTHLY, data_source='monthly', plot=False)
    
    print("\nResults summary:")
    for name, arr in results.items():
        arr_np = arr.values
        print(f"  {name}: min={np.nanmin(arr_np):.4g}, max={np.nanmax(arr_np):.4g}, "
              f"mean={np.nanmean(arr_np):.4g}")
    
    print("\n✓ Monthly mean test passed!")
    return results


def test_hourly_gpiv():
    """Test GPIv computation with hourly data."""
    print("\n" + "="*60)
    print("Testing Hourly GPIv Computation")
    print("="*60)
    
    try:
        from tcvpigpiv import run_vpigpiv_hourly
    except ImportError:
        from tcvpigpiv.vpigpiv_module import run_vpigpiv_hourly
    
    print(f"\nComputing GPIv for {YEAR_HOURLY:04d}-{MONTH_HOURLY:02d}-{DAY_HOURLY:02d} {HOUR_HOURLY:02d}Z...")
    results = run_vpigpiv_hourly(
        YEAR_HOURLY, MONTH_HOURLY, DAY_HOURLY,
        hour=HOUR_HOURLY, plot=False
    )
    
    print("\nResults summary:")
    for name, arr in results.items():
        arr_np = arr.values
        print(f"  {name}: min={np.nanmin(arr_np):.4g}, max={np.nanmax(arr_np):.4g}, "
              f"mean={np.nanmean(arr_np):.4g}")
    
    print("\n✓ Hourly test passed!")
    return results


def test_data_loading():
    """Test direct data loading functions."""
    print("\n" + "="*60)
    print("Testing Data Loading Functions")
    print("="*60)
    
    try:
        from tcvpigpiv import load_era5_data, load_era5_monthly, load_era5_hourly
    except ImportError:
        from tcvpigpiv.era5_loader import load_era5_data, load_era5_monthly, load_era5_hourly
    
    # Test monthly loading
    print(f"\nLoading monthly mean data for {YEAR_MONTHLY:04d}-{MONTH_MONTHLY:02d}...")
    ds_monthly = load_era5_monthly(YEAR_MONTHLY, MONTH_MONTHLY, verbose=True)
    print(f"  Variables: {list(ds_monthly.data_vars)}")
    print(f"  Dimensions: {dict(ds_monthly.dims)}")
    
    # Test hourly loading
    print(f"\nLoading hourly data for {YEAR_HOURLY:04d}-{MONTH_HOURLY:02d}-{DAY_HOURLY:02d} {HOUR_HOURLY:02d}Z...")
    ds_hourly = load_era5_hourly(YEAR_HOURLY, MONTH_HOURLY, DAY_HOURLY, hour=HOUR_HOURLY, verbose=True)
    print(f"  Variables: {list(ds_hourly.data_vars)}")
    print(f"  Dimensions: {dict(ds_hourly.dims)}")
    
    # Test unified interface
    print("\nTesting unified load_era5_data interface...")
    ds1 = load_era5_data(YEAR_MONTHLY, MONTH_MONTHLY, data_source='monthly', verbose=False)
    ds2 = load_era5_data(YEAR_HOURLY, MONTH_HOURLY, day=DAY_HOURLY, hour=HOUR_HOURLY, 
                         data_source='hourly', verbose=False)
    print("  ✓ Both data sources loaded successfully")
    
    print("\n✓ Data loading test passed!")
    return ds_monthly, ds_hourly


def test_individual_components():
    """Test computation of individual GPIv components."""
    print("\n" + "="*60)
    print("Testing Individual Component Calculations")
    print("="*60)
    
    try:
        from tcvpigpiv import (
            load_era5_data,
            calculate_potential_intensity,
            calculate_vws,
            calculate_entropy_deficit,
            calculate_etac,
        )
    except ImportError:
        from tcvpigpiv.era5_loader import load_era5_data
        from tcvpigpiv.vpigpiv_module import (
            calculate_potential_intensity,
            calculate_vws,
            calculate_entropy_deficit,
            calculate_etac,
        )
    
    # Load data
    print(f"\nLoading data for {YEAR_MONTHLY:04d}-{MONTH_MONTHLY:02d}...")
    ds = load_era5_data(YEAR_MONTHLY, MONTH_MONTHLY, data_source='monthly', verbose=False)
    
    # Calculate components individually
    print("\nCalculating components...")
    
    PI, asdeq = calculate_potential_intensity(ds, verbose=False)
    print(f"  PI: min={PI.min().values:.2f}, max={PI.max().values:.2f} m/s")
    
    VWS = calculate_vws(ds, verbose=False)
    print(f"  VWS: min={VWS.min().values:.2f}, max={VWS.max().values:.2f} m/s")
    
    Chi = calculate_entropy_deficit(ds, asdeq, verbose=False)
    print(f"  Chi: min={Chi.min().values:.4f}, max={Chi.max().values:.4f}")
    
    eta_c = calculate_etac(ds, verbose=False)
    print(f"  eta_c: min={eta_c.min().values:.2e}, max={eta_c.max().values:.2e} s^-1")
    
    print("\n✓ Individual components test passed!")
    return PI, VWS, Chi, eta_c


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# tcvpigpiv Package Test Suite")
    print("#"*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Monthly Mean GPIv", test_monthly_gpiv),
        ("Individual Components", test_individual_components),
        ("Hourly GPIv", test_hourly_gpiv),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED: {e}")
            results[test_name] = None
    
    # Summary
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result is not None else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == '__main__':
    main()
