"""
Ventilated Genesis Potential Index (GPIv) computation module.

This module computes the ventilated Genesis Potential Index (GPIv)
using ERA5 reanalysis data and the tcpyPI package. It supports both
monthly mean and hourly data via THREDDS remote access.

Steps:
1. Load ERA5 variables from remote THREDDS URLs
2. Compute Potential Intensity (PI) with tcpyPI
3. Calculate ventilation-related modifiers (VWS, Chi, eta_c)
4. Combine into vPI and GPIv fields
5. Optionally compute anomalies relative to climatology

References:
    Chavas, Camargo, & Tippett (2025, J. Clim.)

Authors:
    Dan Chavas (2025) - Original implementation
    Jose Ocegueda Sanchez (2025) - Hourly data and anomaly support
"""

# ==== IMPORTS ====
from typing import Optional, Literal, Union
from pathlib import Path

import xarray as xr
import numpy as np
import tcpyPI
from tcpyPI import pi


# ==== GPIv CALCULATION UTILITIES ====
# Thermodynamic and GPIv helper functions

def get_rv_from_q(q):
    """Get mixing ratio rv from specific humidity q."""
    return q / (1.0 - q)


def to_hPa(p):
    """Return a pressure field in hPa, converting from Pa if needed.

    ``tcpyPI.pi`` requires the surface/sea-level pressure and the pressure
    levels in hPa, but gridded surface pressure is normally archived in Pa
    (ERA5 ``SP``, CESM/CAM ``PS``).  Units are taken from the CF ``units``
    attribute when present; otherwise they are inferred from magnitude, since
    surface pressure is ~1e5 Pa but only ~1e3 hPa.

    Parameters
    ----------
    p : xr.DataArray or array-like
        Pressure field in Pa or hPa.

    Returns
    -------
    Same type as ``p``, in hPa.
    """
    units = str(getattr(p, 'attrs', {}).get('units', '')).strip().lower()
    if units in ('pa', 'pascal', 'pascals'):
        scale = 1.0 / 100.0
    elif units in ('hpa', 'mb', 'mbar', 'millibar', 'millibars',
                   'hectopascal', 'hectopascals'):
        scale = 1.0
    else:
        # Unknown/absent units: sample one finite value and infer.
        sample = np.asarray(p).ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return p
        scale = 1.0 / 100.0 if sample[0] > 1200.0 else 1.0

    if scale == 1.0:
        return p
    out = p * scale
    if hasattr(out, 'attrs'):
        out.attrs = dict(getattr(p, 'attrs', {}))
        out.attrs['units'] = 'hPa'
    return out


def get_entropy(p, T, rv):
    """Calculates moist entropy.
    
    Parameters
    ----------
    p : float or array-like
        Pressure in Pa
    T : float or array-like
        Temperature in K
    rv : float or array-like
        Mixing ratio in kg/kg
        
    Returns
    -------
    float or array-like
        Moist entropy in J/kg/K
    """
    cp = 1005.7
    R = 287.04
    Rv = 461.5
    Lv0 = 2.501e6
    T_trip = 273.15
    p00 = 100000.0  # Pa

    p_vals = np.asanyarray(p)
    T_vals = np.asanyarray(T)
    rv_vals = np.asanyarray(rv)

    rho_d = p_vals / T_vals / (R + rv_vals * Rv)
    p_v = rv_vals * rho_d * Rv * T_vals
    esl = 611.2 * np.exp(17.67 * (T_vals - 273.15) / (T_vals - 29.65))
    RH = p_v / esl
    RH = np.maximum(RH, 1e-10)  # Avoid log(0)
    p_d = p_vals - p_v

    s = cp * np.log(T_vals / T_trip) - R * np.log(p_d / p00) + Lv0 * rv_vals / T_vals - Rv * rv_vals * np.log(RH)
    return s


def get_saturation_entropy(p, T):
    """Calculates saturation moist entropy.
    
    Parameters
    ----------
    p : float or array-like
        Pressure in Pa
    T : float or array-like
        Temperature in K
        
    Returns
    -------
    float or array-like
        Saturation moist entropy in J/kg/K
    """
    cp = 1005.7
    R = 287.04
    Rv = 461.5
    Lv0 = 2.501e6
    T_trip = 273.15
    p00 = 100000.0  # Pa

    p_vals = np.asanyarray(p)
    T_vals = np.asanyarray(T)

    esl = 611.2 * np.exp(17.67 * (T_vals - 273.15) / (T_vals - 29.65))
    p_d = p_vals - esl
    rvs = R / Rv * esl / p_d

    s_sat = cp * np.log(T_vals / T_trip) - R * np.log(p_d / p00) + Lv0 * rvs / T_vals
    return s_sat


# --- Core Component Calculation Functions ---

def calculate_potential_intensity(
    ds: xr.Dataset,
    sst_var: str = 'SSTK',
    sp_var: str = 'SP',
    t_var: str = 'T',
    q_var: str = 'Q',
    V_reduc: float = 0.8,
    verbose: bool = True
) -> tuple:
    """
    Compute the maximum potential intensity (PI) for each grid point.

    This function follows the logic of the original Colab notebook closely.  The
    ``tcpyPI.pi`` routine requires:

    * Sea‑surface temperature (SST) in degrees Celsius;
    * Surface pressure (MSL) in hectoPascals (hPa); in principle the PI algorithm
      expects the mean sea-level pressure in hPa, but to reproduce the
      reference implementation the ERA5 surface pressure (which is in Pa)
      is passed through without conversion.
    * A vector of pressure levels in hPa ordered from the lowest model level (highest
      pressure) to the top of the atmosphere;
    * A temperature profile in degrees Celsius ordered consistently with the
      pressure levels;
    * A mixing ratio profile in grams per kilogram (g/kg) ordered consistently
      with the pressure levels.

    The ERA5 inputs are provided in Kelvin for temperature, Pascals for
    surface pressure, and kg/kg for specific humidity.  The ``level`` coordinate
    is typically ordered with the smallest value (lowest pressure) first.  To
    satisfy the ``pi`` requirements we convert the units appropriately and
    reverse the level order so that index 0 corresponds to the highest pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the required variables.
    sst_var, sp_var, t_var, q_var : str
        Variable names in ``ds`` for sea surface temperature, surface pressure,
        temperature profile and specific humidity, respectively.
    V_reduc : float
        Reduction factor for maximum wind speed (default 0.8).
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple
        (vmax, asdeq) where:
        - vmax: Maximum potential intensity in m/s
        - asdeq: Air–sea entropy disequilibrium term
    """
    if verbose:
        print("  Calculating Potential Intensity (PI)...")

    CKCD = 0.9  #0.9 is default in tcpypi; assumed constant ratio of enthalpy to momentum exchange coefficients
    
    # Convert SST from Kelvin to Celsius
    sst_c = ds[sst_var] - 273.15
    sst_k = ds[sst_var]

    # tcpyPI.pi expects the surface pressure in hPa, matching the units of the
    # pressure-level coordinate. Gridded surface pressure is archived in Pa
    # (ERA5 SP, CAM PS), so convert when needed.
    sp_hpa = to_hPa(ds[sp_var])

    # Convert temperature from Kelvin to Celsius
    t_c = ds[t_var] - 273.15

    # Specific humidity; converted to mixing ratio below
    q = ds[q_var]

    # Ensure pressure levels are in descending order (highest pressure first)
    levels = ds['level']
    level_vals = levels.values
    if level_vals.size > 1 and level_vals[0] < level_vals[-1]:
        level_desc = levels[::-1]
    else:
        level_desc = levels

    t_c_desc = t_c.reindex(level=level_desc)
    # tcpyPI.pi expects MIXING RATIO in g/kg, not specific humidity.
    rv_gkg_desc = (get_rv_from_q(q) * 1000.0).reindex(level=level_desc)

    # Apply the potential intensity calculation
    vmax, _, _, To_k, _ = xr.apply_ufunc(
        pi,
        sst_c,
        sp_hpa,
        level_desc,
        t_c_desc,
        rv_gkg_desc,
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1, V_reduc=V_reduc),
        input_core_dims=[[], [], ['level'], ['level'], ['level']],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        output_dtypes=[float] * 5
    )

    # Calculate air-sea disequilibrium term
    asdeq = vmax**2 * (1.0 / CKCD) * To_k / (sst_k * (sst_k - To_k))

    #Add metadata
    vmax.attrs = {'long_name': 'Potential Intensity', 'units': 'm/s'}
    asdeq.attrs = {'long_name': 'Air-Sea Entropy Disequilibrium Term', 'units': 'J/kg/K'}

    return vmax, asdeq


def calculate_vws(
    ds: xr.Dataset,
    u_var: str = 'U',
    v_var: str = 'V',
    verbose: bool = True
) -> xr.DataArray:
    """
    Calculate Vertical Wind Shear (VWS) between 200 and 850 hPa.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing wind components.
    u_var : str
        Name of the zonal wind variable.
    v_var : str
        Name of the meridional wind variable.
    verbose : bool
        Print progress messages.
        
    Returns
    -------
    xr.DataArray
        Calculated VWS in m/s.
    """
    if verbose:
        print("  Calculating Vertical Wind Shear (VWS)...")
        
    u200 = ds[u_var].sel(level=200)
    v200 = ds[v_var].sel(level=200)
    u850 = ds[u_var].sel(level=850)
    v850 = ds[v_var].sel(level=850)
    
    vws = np.sqrt((u200 - u850)**2 + (v200 - v850)**2)
    vws.attrs = {'long_name': 'Vertical Wind Shear (200-850 hPa)', 'units': 'm/s'}
    return vws


def calculate_entropy_deficit(
    ds: xr.Dataset,
    asdeq: xr.DataArray,
    sp_var: str = 'SP',
    t_var: str = 'T',
    q_var: str = 'Q',
    verbose: bool = True
) -> xr.DataArray:
    r"""
    Calculate the entropy deficit parameter (Chi).

    The entropy deficit quantifies mid-level dryness relative to the air-sea
    entropy disequilibrium that fuels the storm.  Following Chavas et al.
    (2025) we evaluate

    .. math::
       \chi_m = \frac{s^*_m(600) - s_m(600)}{s^*_{\mathrm{SST}} - s_b},

    where :math:`s_m` and :math:`s^*_m` are the moist and saturation entropies.

    **Numerator.**  Both terms are evaluated at 600 hPa using the *same*
    environmental temperature from the input dataset, so the numerator is
    purely a measure of subsaturation of the environmental sounding at that
    level.  In the theory :math:`s^*_m` is the saturation entropy inside the
    saturated eyewall; substituting the environmental value is justified by
    weak temperature gradient balance, since 600 hPa temperature is nearly
    horizontally uniform in the tropics.  No inner-core sounding is used.

    **Denominator.**  The air-sea entropy disequilibrium is not computed from
    a boundary-layer parcel here.  It is supplied as ``asdeq``, obtained by
    inverting the potential intensity in :func:`calculate_potential_intensity`,

    .. math::
       s^*_{\mathrm{SST}} - s_b = \frac{V_p^2}{C_k/C_d}
                                  \frac{T_o}{T_s\,(T_s - T_o)},

    with :math:`T_o` the outflow temperature returned by ``tcpyPI``.  This
    keeps the denominator exactly consistent with the PI used to normalise the
    ventilation index.  (Earlier revisions computed it from 925 hPa, and then
    from 2 m temperature and dewpoint; that code is retained below, commented
    out, for reference.)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with thermodynamic variables.
    asdeq : xr.DataArray
        Air-sea entropy disequilibrium term from the PI calculation, J/kg/K.
    sp_var, t_var, q_var : str
        Variable names for surface pressure, temperature, and specific humidity.
        ``sp_var`` is used only to supply coordinates for the output.
    verbose : bool
        Print progress messages.

    Returns
    -------
    xr.DataArray
        Calculated entropy deficit (Chi), dimensionless.
    """
    if verbose:
        print("  Calculating Entropy Deficit (Chi)...")
    
    T = ds[t_var]
    q = ds[q_var]
    psfc = ds[sp_var]
    
    # Get values at specific levels.  For the mid‑tropospheric values we
    # retain the 600 hPa level, while the boundary layer values are now
    # derived from 2‑metre near‐surface variables rather than 925 hPa.  These
    # are loaded into the dataset as 'T2M' (2‑m temperature) and 'D2M'
    # (2‑m dewpoint temperature) in the data loading section.
    T_600 = T.sel(level=600)
    q_600 = q.sel(level=600)
    
    # Convert specific humidity at 600hPa to mixing ratio using the helper
    rv_600 = get_rv_from_q(q_600)
    if verbose:
        print("T600: min =", np.nanmin(T_600).item(), ", max =", np.nanmax(T_600).item())
        print("rv600: min =", np.nanmin(rv_600).item(), ", max =", np.nanmax(rv_600).item())

    # Retrieve 2‑metre temperature and dewpoint from the dataset.  Depending
    # on the remote file format these may originally have been named '2T' or
    # 'T2M' (and '2D' or 'D2M'); they are renamed to 'T2M' and 'D2M' when
    # merging into ``ds``.
    # T2m = ds['T2M']
    # Td2m = ds['D2M']
    # print("T2m: min =", np.nanmin(T2m).item(), ", max =", np.nanmax(T2m).item())
    # print("Td2m: min =", np.nanmin(Td2m).item(), ", max =", np.nanmax(Td2m).item())
    
    # Compute the near‑surface (2 m) mixing ratio from the dewpoint and
    # surface pressure.  The vapor pressure at the dewpoint is the saturation
    # vapour pressure, computed using the same Clausius–Clapeyron expression
    # employed in the entropy functions.  The mixing ratio is given by
    # r_v = 0.622 * e / (p_sfc - e).  Note that ``psfc`` is the surface
    # pressure in Pa and broadcasts over the horizontal dimensions.
    # e_surf = 611.2 * np.exp(17.67 * (Td2m - 273.15) / (Td2m - 29.65))
    # rv_2m = 0.622 * e_surf / (psfc - e_surf)
    # print("esurf: min =", np.nanmin(e_surf).item(), ", max =", np.nanmax(e_surf).item())
    # print("rv_2m: min =", np.nanmin(rv_2m).item(), ", max =", np.nanmax(rv_2m).item())

    # Calculate entropy components using the helper functions

    sm_600 = get_entropy(p=60000., T=T_600, rv=rv_600)
    sm_star_600 = get_saturation_entropy(p=60000., T=T_600)
    if verbose:
        print("sm_600: min =", np.nanmin(sm_600).item(), ", max =", np.nanmax(sm_600).item())
        print("sm_star_600: min =", np.nanmin(sm_star_600).item(), ", max =", np.nanmax(sm_star_600).item())
    
    # Calculate Chi
    numerator = sm_star_600 - sm_600
    chi = numerator / asdeq
    
    chi = xr.DataArray(chi, coords=psfc.coords, dims=psfc.dims,
                       attrs={'long_name': 'Entropy Deficit (Chi)', 'units': ''})
    return chi


def calculate_etac(
    ds: xr.Dataset,
    vo_var: str = 'VO',
    verbose: bool = True
) -> xr.DataArray:
    """
    Calculate the capped low-level absolute vorticity (eta_c).
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing relative vorticity.
    vo_var : str
        Name of the relative vorticity variable.
    verbose : bool
        Print progress messages.
        
    Returns
    -------
    xr.DataArray
        Capped absolute vorticity at 850 hPa.
    """
    if verbose:
        print("  Calculating Capped Vorticity (eta_c)...")
        
    vo_850 = ds[vo_var].sel(level=850)

    # Coriolis parameter
    omega = 2 * np.pi / (24 * 3600)
    f = 2 * omega * np.sin(np.deg2rad(ds['latitude']))

    # Absolute vorticity
    abs_vo_850 = vo_850 + f

    # Cap the absolute vorticity at ±3.7e-5 s^-1 as in the original script.  When
    # the magnitude exceeds the cap, we set the value to +3.7e-5 rather than
    # preserving the sign.  For magnitudes below the cap we keep the signed
    # absolute vorticity.
    capped = xr.where(np.abs(abs_vo_850) > 3.7e-5, 3.7e-5, abs_vo_850)
    capped.attrs = {
        'long_name': 'Capped 850 hPa Absolute Vorticity',
        'units': 's**-1'
    }
    return capped


def compute_gpiv_from_dataset(
    ds: xr.Dataset,
    verbose: bool = True
) -> xr.Dataset:
    """
    Compute vPI and GPIv and all components from a merged dataset.
    
    This is the main computation function that computes the calculation
    of all GPIv components.
    
    Parameters
    ----------
    ds : xr.Dataset
        A merged dataset containing all necessary variables:
        SSTK, SP, T, Q, U, V, VO
    verbose : bool
        Print progress messages.

    Returns
    -------
    xr.Dataset
        Dataset containing GPIv and all intermediate products:
        PI, vPI, VWS, Chi, eta_c, ventilation_index, GPIv
    """
    # Variable names
    u_var = 'U'
    v_var = 'V'
    t_var = 'T'
    q_var = 'Q'
    vo_var = 'VO'
    sst_var = 'SSTK'
    sp_var = 'SP'
    
    # Calculate all components
    PI, asdeq = calculate_potential_intensity(
        ds, sst_var, sp_var, t_var, q_var, V_reduc=1.0, verbose=verbose
    )
    VWS = calculate_vws(ds, u_var, v_var, verbose=verbose)
    Chi = calculate_entropy_deficit(ds, asdeq, sp_var, t_var, q_var, verbose=verbose)
    eta_c = calculate_etac(ds, vo_var, verbose=verbose)
    
    # Combine components
    if verbose:
        print("  Combining components for final GPIv...")
    
    # Ventilation Index (VI)
    ventilation_index = (VWS * Chi) / PI
    ventilation_index = ventilation_index.where(ventilation_index > 0) # Set non-positives to NaN
    ventilation_index.attrs = {'long_name': 'Ventilation Index', 'units': ''}

    # Ventilated Potential Intensity (vPI)
    VI_max = 0.145
    VI = ventilation_index.where(ventilation_index <= VI_max)
    
    # Solve cubic equation for vPI factor
    with np.errstate(divide='ignore', invalid='ignore'):
        VI_complex = VI.values.astype(complex)
        ratio = VI_complex / VI_max
        term1 = (ratio**2 - 1.)**0.5
        term2 = (term1 - ratio)**(1./3.)
        x = (1. / np.sqrt(3.)) * term2
        vPI_factor_complex = x + 1. / (3. * x)
        vPI_factor = vPI_factor_complex.real
    
    vPI = xr.DataArray(vPI_factor, coords=PI.coords, dims=PI.dims) * PI
    vPI.attrs = {'long_name': 'Ventilated Potential Intensity', 'units': 'm/s'}
    
    # Final Ventilated Genesis Potential Index (GPIv)
    # Ensure latitude is a 2D array for broadcasting
    lat2d, _ = xr.broadcast(ds['latitude'], ds['longitude'])
    cos_lat = np.cos(np.deg2rad(lat2d))
    
    dx = 2.0
    dy = 2.0

    # The formula from the paper
    GPIv = (102.1 * vPI * eta_c)**4.90 * cos_lat * dx * dy
    GPIv.attrs = {'long_name': 'Ventilated Genesis Potential Index', 'units': ''}
    
    # Assemble results into a single dataset
    results_ds = xr.Dataset({
        'GPIv': GPIv,
        'vPI': vPI,
        'PI': PI,
        'ventilation_index': ventilation_index,
        'VWS': VWS,
        'Chi': Chi,
        'eta_c': eta_c,
    })
    
    return results_ds


###############################################################################
# Public API
###############################################################################

def run_vpigpiv(
    year: int,
    month: int,
    day: Optional[int] = None,
    hour: Optional[int] = None,
    data_source: Literal['monthly', 'hourly'] = 'monthly',
    compute_anomalies: bool = False,
    climatology_path: Optional[Union[str, Path]] = None,
    plot: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Compute vPI and GPIv for a given time.

    This is the main entry point for computing GPIv. It supports both
    monthly mean and hourly ERA5 data, with optional anomaly calculation.

    Parameters
    ----------
    year : int
        Year to analyse.
    month : int
        Month (1-12) to analyse.
    day : int, optional
        Day (1-31) to analyse. Required for hourly data.
    hour : int, optional
        Hour (0-23) to analyse. If None with hourly data, uses daily mean.
    data_source : {'monthly', 'hourly'}
        Which ERA5 dataset to use:
        - 'monthly': Monthly mean dataset (d633001)
        - 'hourly': Hourly dataset (d633000)
    compute_anomalies : bool
        If True, also compute anomalies relative to climatology.
        Requires climatology_path to be specified.
    climatology_path : str or Path, optional
        Path to pre-computed climatology NetCDF file.
        Required if compute_anomalies is True.
    plot : bool
        If True, generate diagnostic plots.
    verbose : bool
        Print progress messages.

    Returns
    -------
    xr.Dataset
        Dataset of computed components (PI, vPI, VWS, Chi, eta_c,
        ventilation_index and GPIv). If compute_anomalies is True,
        also includes anomaly fields.

    Examples
    --------
    >>> # Monthly mean for September 2022
    >>> results = run_vpigpiv(2022, 9)
    
    >>> # Hourly data for August 15, 2020 at 12Z
    >>> results = run_vpigpiv(2020, 8, day=15, hour=12, data_source='hourly')
    
    >>> # Hourly with anomalies
    >>> results = run_vpigpiv(
    ...     2020, 8, day=15, hour=12,
    ...     data_source='hourly',
    ...     compute_anomalies=True,
    ...     climatology_path='gpiv_climatology.nc'
    ... )
    """
    from .era5_loader import load_era5_data
    
    # Load data
    ds = load_era5_data(
        year, month, day=day, hour=hour,
        data_source=data_source, verbose=verbose
    )
    
    # Compute GPIv
    results = compute_gpiv_from_dataset(ds, verbose=verbose)
    
    # Print summaries
    if verbose:
        for name, arr in results.items():
            print(f"{name}: min={arr.min().values:.4g}, max={arr.max().values:.4g}, "
                  f"mean={arr.mean().values:.4g}")
    
    # Compute anomalies if requested
    if compute_anomalies:
        if climatology_path is None:
            raise ValueError("climatology_path must be specified when compute_anomalies=True")
        
        from .climatology import load_climatology, compute_anomalies as calc_anom
        
        if verbose:
            print(f"\nComputing anomalies relative to month {month} climatology...")
        
        clim = load_climatology(climatology_path)
        anom_results = calc_anom(results, clim, month)
        
        # Merge anomalies into results
        results = xr.merge([results, anom_results])
        
        if verbose:
            print("Anomaly fields added.")
    
    # Generate plots if requested
    if plot:
        try:
            plot_vpigpiv(ds, results, year, month, day, hour)
        except ImportError:
            if verbose:
                print("Warning: Could not generate plots (matplotlib/cartopy not available)")
    
    return results


def run_vpigpiv_hourly(
    year: int,
    month: int,
    day: int,
    hour: Optional[int] = None,
    compute_anomalies: bool = False,
    climatology_path: Optional[Union[str, Path]] = None,
    plot: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Convenience function for hourly GPIv computation.
    
    This is a wrapper around run_vpigpiv that sets data_source='hourly'.
    
    Parameters
    ----------
    year : int
        Year to analyse.
    month : int
        Month (1-12) to analyse.
    day : int
        Day (1-31) to analyse.
    hour : int, optional
        Hour (0-23) to analyse. If None, returns all hours of the day.
    compute_anomalies : bool
        If True, compute anomalies relative to climatology.
    climatology_path : str or Path, optional
        Path to climatology file.
    plot : bool
        Generate diagnostic plots.
    verbose : bool
        Print progress messages.
        
    Returns
    -------
    xr.Dataset
        Computed GPIv and components.
        
    Examples
    --------
    >>> # GPIv for August 15, 2020 at 12Z
    >>> results = run_vpigpiv_hourly(2020, 8, 15, hour=12)
    """
    return run_vpigpiv(
        year, month, day=day, hour=hour,
        data_source='hourly',
        compute_anomalies=compute_anomalies,
        climatology_path=climatology_path,
        plot=plot,
        verbose=verbose
    )

# Moved as a comment so it doesn't show up when asking for help of the functions.
#    The function mirrors the plotting in the original notebook but is
#    encapsulated here so that users of the package can easily reproduce
#    the figures.  It generates maps for the ventilation index, vPI and PI,
#    capped vorticity, GPIv, and a sanity-check map of SST minus 2 m
#    temperature.

def plot_vpigpiv(
    ds: xr.Dataset,
    results: xr.Dataset,
    year: int,
    month: int,
    day: Optional[int] = None,
    hour: Optional[int] = None
) -> None:
    """
    Generate diagnostic maps for GPIv and its components.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing SST and other fields.
    results : xr.Dataset
        Output from compute_gpiv_from_dataset.
    year, month : int
        Date for figure titles.
    day, hour : int, optional
        For hourly data, include in titles.
    """
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    
    # Build title suffix
    if day is not None:
        if hour is not None:
            time_str = f"{year}-{month:02d}-{day:02d} {hour:02d}Z"
        else:
            time_str = f"{year}-{month:02d}-{day:02d}"
    else:
        time_str = f"{year}-{month:02d}"
    
    # Unpack fields
    PI = results['PI']
    vPI = results['vPI']
    eta_c = results['eta_c']
    GPIv = results['GPIv']
    ventilation_index = results['ventilation_index']

    centlong = 180
    
    # Plot VI
    fig, ax = plt.subplots(1, figsize=(6, 3), constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=centlong)})
    lev_exp = np.linspace(-1, 1, 25)
    levs = np.power(10, lev_exp)
    xr.plot.contourf(ventilation_index, ax=ax, norm=LogNorm(), levels=levs,
                     transform=ccrs.PlateCarree(), cmap=cm.plasma)
    ax.set_title(f"Ventilation Index, {time_str}")
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot vPI and PI
    fig, ax = plt.subplots(2, figsize=(6, 6), constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=centlong)})
    xr.plot.contourf(vPI, ax=ax[0], transform=ccrs.PlateCarree())
    ax[0].set_title(f"vPI, {time_str}")
    ax[0].coastlines()
    gl = ax[0].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    xr.plot.contourf(PI, ax=ax[1], transform=ccrs.PlateCarree())
    ax[1].set_title(f"PI, {time_str}")
    ax[1].coastlines()
    gl = ax[1].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot eta_c
    fig, ax = plt.subplots(1, figsize=(6, 3), constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=centlong)})
    xr.plot.contourf(eta_c, ax=ax, transform=ccrs.PlateCarree())
    ax.set_title(f"eta_c, {time_str}")
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot GPIv
    fig, ax = plt.subplots(1, figsize=(6, 3), constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=centlong)})
    xr.plot.contourf(GPIv, ax=ax, transform=ccrs.PlateCarree())
    ax.set_title(f"GPIv, {time_str}")
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.show()
