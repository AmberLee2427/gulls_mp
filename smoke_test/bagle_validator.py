"""BAGLE physics validation for gulls binary lens astrometry."""
from __future__ import annotations
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


# Run-or-die: require BAGLE to be importable via the Python environment.
from bagle import model
from bagle import parallax as _bagle_parallax  # For direct parallax vector computation receipts.
from .gulls_io import read_gulls_observatory_settings, read_gulls_lightcurve, parse_out_file, load_lightcurve, calculate_magnification_from_lightcurve
from utils import require_finite, require_series, rotation_ne_to_xy

# Horizons Roman earliest supported ephemeris epoch (receipt: previously observed failure boundary)
# Source citation: astroquery JPL Horizons error message for target "Roman Space Telescope (spacecraft)".
_EARLIEST_ROMAN_JD = 2461343.795428987  # JD TDB
_EARLIEST_ROMAN_MJD = _EARLIEST_ROMAN_JD - 2400000.5


# --- Parallax / Observer Frame Audit Utilities ---------------------------------
def audit_parallax_alignment(ra_deg: float, dec_deg: float, times_mjd: np.ndarray,
                             bagle_model, obs_location,
                             max_rel_err: float = 0.15) -> Dict[str, Any]:
    """Fail-fast audit comparing BAGLE parallax vectors vs independent recompute.

    Purpose:
      Detect frame/epoch/observer mismatches causing temporal peak misalignment or
      systematic astrometric offsets by verifying that BAGLE's internal parallax
      application matches a fresh direct call to `bagle.parallax.parallax_in_direction`.

        Method:
            - Recompute parallax vectors pvec = (p_E, p_N) via BAGLE's
                bagle/parallax.py:parallax_in_direction (see source: constructs
                Time(mjd + 2400000.5, scale='tdb'); projects barycentric observer
                position onto East/North basis). Ordering confirmed by reading
                parallax_in_direction: e = dot(pos, _east_projected); n = dot(pos, _north_projected);
                pvec = [[e, n], ...]. We treat columns explicitly as (East, North).
      - Infer BAGLE-applied parallax displacement used inside lens/source motion by
        differencing resolved lens origin astrometry with parallax disabled.
      - Compare vector norms and component-wise values; enforce relative error bounds.

    Assumptions:
      - bagle_model has attributes: parallaxFlag, piE_amp or piL, obsLocation.
      - Times provided in MJD (receipt: BAGLE requires MJD).
      - Observer location string matches BAGLE expectations ('earth', 'sun', satellite id, etc.).

    Raises:
      RuntimeError with structured diagnostics on mismatch beyond tolerance.

    Returns:
      dict with diagnostic metrics (norm ratios, max component abs diff, peak time indices, etc.).
    """
    # Strict: required BAGLE attributes must exist; no None receipts.
    if not hasattr(bagle_model, 'parallaxFlag'):
        raise RuntimeError("BAGLE model missing required attribute 'parallaxFlag'")
    if not hasattr(bagle_model, 'obsLocation'):
        raise RuntimeError("BAGLE model missing required attribute 'obsLocation'")
    diag: Dict[str, Any] = {
        'parallaxFlag': bool(bagle_model.parallaxFlag),
        'obsLocation_model': bagle_model.obsLocation,
        'obsLocation_audit': obs_location,
        'n_times': int(len(times_mjd))
    }

    # Observer location strictness — enforced regardless of parallaxFlag so mismatches never slip through.
    # We REQUIRE a single, non-empty string for obsLocation here. Do not accept list/tuple and never index strings.
    # Prior bug: indexing a string like '-211' as if it were a list produced '-' and a Horizons URL with COMMAND='-' (HTTP 500).
    model_obs_loc = bagle_model.obsLocation
    # Accept string or single-element list/tuple[str]
    if isinstance(model_obs_loc, str):
        model_obs_str = model_obs_loc
    elif isinstance(model_obs_loc, (list, tuple)):
        if len(model_obs_loc) != 1 or not isinstance(model_obs_loc[0], str):
            raise RuntimeError(
                f"bagle_model.obsLocation must be a string or a 1-element list/tuple[str]; got {type(model_obs_loc)} with value {model_obs_loc!r}"
            )
        model_obs_str = model_obs_loc[0]
    else:
        raise RuntimeError(
            f"bagle_model.obsLocation must be a string or a 1-element list/tuple[str]; got {type(model_obs_loc)} with value {model_obs_loc!r}"
        )
    if model_obs_str == "":
        raise RuntimeError("bagle_model.obsLocation is empty string")
    if model_obs_str != obs_location:
        raise RuntimeError(f"Observer location mismatch: model='{model_obs_str}' vs audit='{obs_location}'")

    if not bagle_model.parallaxFlag:
        diag['skipped'] = True
        return diag  # No parallax to audit.

    times_mjd = np.asarray(times_mjd, dtype=float)
    if times_mjd.ndim != 1:
        raise RuntimeError("times_mjd must be 1D array for parallax audit")
    if not np.all(np.isfinite(times_mjd)):
        raise RuntimeError("times_mjd contains non-finite values")

    # Recompute raw parallax vectors (dimensionless projected barycentric offsets in AU scaled to East/North axes)
    # Source citation: BAGLE_Microlensing/src/bagle/parallax.py:parallax_in_direction
    # returns array shape (N,2) with columns (East, North). We retain that ordering.
    # (debug removed)
    pvec = _bagle_parallax.parallax_in_direction(ra_deg, dec_deg, times_mjd, obsLocation=obs_location)
    if pvec.shape != (len(times_mjd), 2):
        raise RuntimeError(f"parallax_in_direction returned shape {pvec.shape}, expected (N,2)")

    # Extract BAGLE lens parallax (piL) in mas for astrometric lens-origin displacement
    piL = None
    if hasattr(bagle_model, 'piL'):
        piL = getattr(bagle_model, 'piL')
    # Fallback: derive from dL in pc if needed: parallax (mas) ~ 1000 / dL_pc
    if (piL is None) and hasattr(bagle_model, 'dL'):
        dL_pc = getattr(bagle_model, 'dL')
        if np.isfinite(dL_pc) and dL_pc > 0:
            piL = 1000.0 / dL_pc
    if (piL is None) or (not np.isfinite(piL)):
        raise RuntimeError("Cannot determine lens parallax piL (mas) for parallax audit")

    # Expected applied displacement in astrometry (arcsec) ~ piL(mas) * pvec * 1e-3
    # We'll compare relative shape by numerically toggling parallaxFlag through lens astrometry if possible.
    try:
        bagle_model.parallaxFlag = False
        astrometry_no_par = bagle_model.get_lens_origin_astrometry(times_mjd, filt_idx=0)
    finally:
        bagle_model.parallaxFlag = True
    astrometry_with_par = bagle_model.get_lens_origin_astrometry(times_mjd, filt_idx=0)
    if astrometry_no_par.shape != astrometry_with_par.shape:
        raise RuntimeError("Astrometry shape mismatch between parallax on/off calls")
    if astrometry_no_par.shape[1] != 2:
        raise RuntimeError("Lens origin astrometry must have shape (N,2)")

    applied_disp = astrometry_with_par - astrometry_no_par  # arcsec (columns: East, North per BAGLE get_lens_origin_astrometry docstring ordering)
    # Convert applied_disp to mas for sensitivity (1 arcsec = 1000 mas)
    applied_disp_mas = applied_disp * 1e3
    # Scale pvec by piL to get expected in mas (BAGLE usage in get_lens_origin_astrometry)
    # piL (mas) * pvec (dimensionless) -> mas displacement; component alignment relies on confirmed (E,N) ordering.
    expected_disp_mas = (piL * pvec)
    # Component-wise comparison
    diff = applied_disp_mas - expected_disp_mas
    comp_abs_max = np.max(np.abs(diff))
    # Norm comparisons
    norm_applied = np.linalg.norm(applied_disp_mas, axis=1)
    norm_expected = np.linalg.norm(expected_disp_mas, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_err = np.where(norm_expected > 0, np.abs(norm_applied - norm_expected) / norm_expected, 0.0)

    diag.update({
        'piL_mas': float(piL),
        'component_abs_max_diff_mas': float(comp_abs_max),
        'median_rel_err': float(np.median(rel_err)),
        'max_rel_err': float(np.max(rel_err)),
        'norm_expected_med_mas': float(np.median(norm_expected)),
        'norm_applied_med_mas': float(np.median(norm_applied))
    })

    if np.max(rel_err) > max_rel_err:
        raise RuntimeError(
            f"Parallax alignment failure: max_rel_err={np.max(rel_err):.3f} > tolerance {max_rel_err:.3f}; "
            f"component_abs_max_diff={comp_abs_max:.3e} mas; obsLocation={model_obs_loc}; piL={piL:.3g} mas"
        )
    return diag


def _resolve_and_apply_bagle_observer(bagle_model, params: Dict[str, str], ra_deg: float, dec_deg: float, times_mjd: np.ndarray, verbose: bool) -> Tuple[str, Dict[str, Any]]:
    """Resolve GULLS observer and set bagle_model.obsLocation accordingly.

    Returns (used_obs_location, receipt_dict). Raises on unresolvable/unsupported settings.
    """
    settings = read_gulls_observatory_settings(params)
    space_flag = settings.get('SPACE')
    orbit_code = settings.get('ORBIT')
    receipt: Dict[str, Any] = {
        'gulls_obs_file': str(settings.get('file')),
        'SPACE': space_flag,
        'ORBIT': orbit_code,
    }
    if space_flag not in (0, 1, None):
        raise RuntimeError(f"Unexpected SPACE flag value: {space_flag}")

    if space_flag in (None, 0):
        used = 'earth'
    else:
        # Strict: SPACE=1 requires ORBIT=2 and Roman Horizons ID '-211'. No candidates, no probing.
        if orbit_code is None:
            raise RuntimeError("GULLS observatory has SPACE=1 but no ORBIT specified; unable to determine observer ephemeris.")
        if orbit_code != 2:
            raise RuntimeError(f"Unsupported space ORBIT code '{orbit_code}'. This validator only supports ORBIT=2 (L2/Roman).")
        used = '-211'
        receipt['candidate_order'] = ['-211']

    # Apply to model; BAGLE indexes obsLocation by filt_idx, so supply a single-element list to avoid
    # truncating strings like '-211' when treated as a sequence.
    bagle_model.obsLocation = [used]
    # Minimal observer debug: echo only the resolved/used observer string
    # (debug removed)

    receipt['bagle_obs_applied'] = used
    # Quick sanity probe via audit function later; return
    return used, receipt


def convert_to_bagle_params(out_params: Dict[str, Any], simulation_zero_time: float, event_type: str) -> Dict[str, float]:
    """Convert gulls .out parameters to BAGLE model parameters.

    This function is intentionally strict: required keys must be present in
    out_params. If any are missing, we raise a KeyError rather than silently
    fabricating values.
    """
    # Strict required keys that must exist in the .out file
    required_keys = [
        'Lens_Mass', 'Planet_q', 't0lens1', 'thetaE', 'u0lens1', 'Planet_s',
        'alpha', 'rho', 'Lens_Dist', 'Source_Dist', 'Lens_mul', 'Lens_mub',
        'Source_mul', 'Source_mub', 'piE', 'Source_W146', 'Lens_W146',
        'ra_deg', 'dec_deg', 'tref',
        # Require explicit source/lens galactic + equatorial for PM transform
        'Lens_l', 'Lens_b', 'Lens_RA2000.0', 'Lens_DEC2000.0',
        'Source_l', 'Source_b', 'Source_RA2000.0', 'Source_DEC2000.0'
    ]
    # BAGLE sign/axis conventions (copied from BAGLE model docstrings):
    #
    # Comment on sign conventions:
    # thetaS0 = xS0 - xL0
    # (difference in positions on sky, heliocentric, at t0)
    # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
    # if u0_E > 0 then the Source is to the East of the lens
    # if u0_E < 0 then the source is to the West of the lens
    # We adopt the following sign convention (same as Gould:2004):
    #    u0_amp > 0 means u0_E > 0
    #    u0_amp < 0 means u0_E < 0
    # Note that we assume beta = u0_amp (with same signs).
    #
    # Calculate the closest approach vector. Define beta sign convention
    # same as of Andy Gould does with beta > 0 means u0_E > 0
    # (lens passes to the right of the source as seen from Earth or Sun).
    # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
    # the sign of beta is always the same as the sign of u0_amp. Therefore this
    # usage of the function with u0_amp works exactly the same.
    missing = [k for k in required_keys if k not in out_params]
    if missing:
        # Fail fast with integrity – caller will see exactly what's missing
        raise KeyError(f"Missing required keys in .out file: {missing}")

    params: Dict[str, float] = {}
    mL_primary = float(out_params['Lens_Mass'])
    if not np.isfinite(mL_primary) or mL_primary <= 0:
        raise ValueError(f"Lens_Mass must be positive finite Msun; got {out_params['Lens_Mass']!r}")
    q = float(out_params['Planet_q'])
    if not np.isfinite(q) or q <= 0:
        raise ValueError(f"Planet_q must be positive finite mass ratio; got {out_params['Planet_q']!r}")
    params['mLp'] = mL_primary
    params['mLs'] = mL_primary * q

    # Convert simulation time to BJD, then BJD to MJD for BAGLE
    # Time scale assumption (explicit, no guessing):
    # - GULLS emits BJD values in the lightcurves; we treat these as BJD_TDB.
    # - We convert BJD_TDB to MJD_TDB via MJD = JD - 2400000.5.
    # - BAGLE parallax constructs astropy Time with scale='tdb' from provided MJD
    #   (see BAGLE_Microlensing/src/bagle/parallax.py: Time(mjd + 2400000.5, format='jd', scale='tdb')).
    # If GULLS adopts a different timescale (UTC/TT/TCB), this conversion must be
    # updated explicitly; do not add heuristics or silent fallbacks here.
    t0_rel = float(out_params['t0lens1'])
    if not np.isfinite(t0_rel):
        raise ValueError(f"t0lens1 must be finite (BJD-relative days); got {out_params['t0lens1']!r}")
    t0_bjd = t0_rel + simulation_zero_time
    # Receipt: BAGLE requires all time inputs in MJD.
    # See BAGLE_Microlensing/src/bagle/model.py:
    #  - Other notes: "All times must be reported in MJD."
    #  - Method docstrings (e.g., get_amplification/get_astrometry) specify t is MJD.
    t0_mjd = t0_bjd - 2400000.5
    params['t0_original'] = t0_mjd  # retain original (COM frame) for plotting/diagnostics

    # Reference time (tref) in .out used by GULLS parallax setup (geocentric transform anchor)
    tref = float(out_params['tref'])
    if not np.isfinite(tref):
        raise ValueError(f"tref must be finite (BJD-relative days); got {out_params['tref']!r}")
    tref_bjd = tref + simulation_zero_time
    tref_mjd = tref_bjd - 2400000.5  # what the actual fuck is mjd?
    params['t_ref_mjd'] = tref_mjd

    thetaE = float(out_params['thetaE'])
    if not np.isfinite(thetaE) or thetaE <= 0:
        raise ValueError(f"thetaE must be positive finite (mas); got {out_params['thetaE']!r}")
    params['thetaE'] = thetaE
    # u0 sign convention reconciliation:
    # GULLS .out provides u0lens1 which can be negative (e.g., -0.787... in sample). BAGLE allows beta (closest approach)
    # to be signed; sign encodes orientation (beta > 0 when source East of lens per BAGLE PSBL_PhotAstromParam1 docstring).
    # We map beta_signed = u0lens1 * thetaE directly, preserving sign, and record u0_amp = abs(u0lens1).
    u0_gulls = float(out_params['u0lens1'])
    if not np.isfinite(u0_gulls):
        raise ValueError(f"u0lens1 must be finite (Einstein radii); got {out_params['u0lens1']!r}")
    # UNIT NOTE: beta is an angular closest approach. We construct it in mas via beta = u0 * thetaE (mas).
    # In BAGLE's PSBL_PhotAstrom pipeline, Einstein-radii vs. arcsec unit handling is mixed:
    # - source trajectory uses Einstein radii (u),
    # - lens astrometry uses arcsec and converts mas->arcsec with 1e-3 (see model.py ~6160: xL += (piL * parallax)*1e-3).
    # If BAGLE later normalizes by thetaE internally to compute u, a mas vs arcsec mismatch would scale timescales by ~1e3.
    params['beta_mas'] = u0_gulls * thetaE  # signed beta (mas)
    # and I suppose we just hope the signs go the same way
    params['u0_original'] = u0_gulls      # retain original sign for receipts
    s_ER = float(out_params['Planet_s'])
    if not np.isfinite(s_ER) or s_ER <= 0:
        raise ValueError(f"Planet_s must be positive finite (Einstein radii); got {out_params['Planet_s']!r}")
    # UNIT NOTE (BAGLE model.py ~6160-6180):
    #   PSBL_PhotAstrom computes lens offsets as:
    #       offset = 0.5 * self.sep * [sin(alpha), cos(alpha)]
    #       offset *= 1e-3  # convert to arcsec
    #   Therefore, self.sep is in mas internally. GULLS Planet_s is dimensionless (Einstein radii),
    #   so supply sep in mas = s * thetaE.
    # ORIGIN NOTE (midpoint vs COM):
    #   The 0.5*sep midpoint placement differs from a center-of-mass reference. With tiny q, COM≈primary, while
    #   midpoint shifts primary by ~sep/2. This can alter effective u0 and t0 vs GULLS. Keep this in mind if large peak lags persist.
    params['projected_separation_mas'] = s_ER * thetaE
    # Alpha handling:
    # - GULLS generation sets Event->alpha in degrees in buildEvent.cpp:
    #   Event->alpha = 360.0 * ran2(idum);  (uniform in [0,360) deg)
    # - BAGLE PSBL PhotAstrom Param1 expects alpha in DEGREES as documented in
    #   BAGLE_Microlensing/src/bagle/fake_data.py (see docstring near the call to
    #   model.PSBL_PhotAstrom_Par_Param1 where "alpha : float (degrees)" is specified).
    # Therefore: keep alpha in degrees when passing into BAGLE.
    alpha_deg = float(out_params['alpha'])  # degrees from GULLS (angle between trajectory and binary axis)
    if not np.isfinite(alpha_deg) or not (0.0 <= alpha_deg < 360.0 + 1e-9):
        raise ValueError(
            "alpha must be finite degrees in [0, 360). "
            "Receipt: GULLS generates alpha in degrees (uniform 0..360) and BAGLE expects degrees: "
            "BAGLE_Microlensing/src/bagle/fake_data.py:597 (fake_data_PSBL docstring lists 'alpha : float (degrees)').")
    # Defer mapping to BAGLE alpha until after we compute the sky trajectory angle from proper motions.
    params['alpha_deg_rel_to_x'] = alpha_deg    # store relative angle (traj vs axis) for reparameterization
    rho = float(out_params['rho'])
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"rho must be positive finite (dimensionless); got {out_params['rho']!r}")
    params['rho_ER'] = rho

    # Separation remains the physical midpoint-frame value: sep = s * thetaE (mas).
    # We DO NOT alter sep for COM emulation now; instead we reparameterize t0 and beta below to account for origin shift.

    # Distances from gulls are in kpc; BAGLE expects pc.
    dL_kpc = float(out_params['Lens_Dist'])
    dS_kpc = float(out_params['Source_Dist'])
    if not (np.isfinite(dL_kpc) and np.isfinite(dS_kpc)):
        raise ValueError(f"Lens_Dist/Source_Dist must be finite (kpc); got Lens_Dist={out_params['Lens_Dist']!r}, Source_Dist={out_params['Source_Dist']!r}")
    params['dL'] = 1000.0 * dL_kpc  # pc
    params['dS'] = 1000.0 * dS_kpc  # pc
    if params['dS'] <= 0:
        raise ValueError("Source_Dist must be > 0")
    if params['dL'] <= 0:
        raise ValueError("Lens_Dist must be > 0")
    if not (params['dL'] < params['dS']):
        raise ValueError(f"Geometry invalid: lens must be in front of source: dL={params['dL']} pc, dS={params['dS']} pc")
    params['dL_dS'] = params['dL'] / params['dS']

    # Proper motions: gulls outputs are Galactic (l,b) mas/yr; BAGLE wants RA/Dec (E,N) mas/yr.
    # geo, helio, or L2???
    # Transform using astropy.
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Build Galactic coords with proper motions; gulls columns are in degrees.
    # are these helio centric. WTF is happening here?
    lens_coord_gal = SkyCoord(l=float(out_params['Lens_l'])*u.deg,
                              b=float(out_params['Lens_b'])*u.deg,
                              pm_l_cosb=float(out_params['Lens_mul'])*u.mas/u.yr,
                              pm_b=float(out_params['Lens_mub'])*u.mas/u.yr,
                              distance=float(out_params['Lens_Dist'])*1e3*u.pc,
                              frame='galactic')  # kpc->pc
    lens_icrs = lens_coord_gal.icrs
    params['muL_E'] = lens_icrs.pm_ra_cosdec.to_value(u.mas/u.yr)
    params['muL_N'] = lens_icrs.pm_dec.to_value(u.mas/u.yr)
    # UNIT NOTE (BAGLE model.py ~6160): xL uses muL * 1e-3 → arcsec/yr, so muL here must be in mas/yr as supplied.
    if not (np.isfinite(params['muL_E']) and np.isfinite(params['muL_N'])):
        raise ValueError("Lens proper motions transformed to ICRS are non-finite (mas/yr). Check input mul/mub.")

    source_coord_gal = SkyCoord(l=float(out_params['Source_l'])*u.deg,
                                b=float(out_params['Source_b'])*u.deg,
                                pm_l_cosb=float(out_params['Source_mul'])*u.mas/u.yr,
                                pm_b=float(out_params['Source_mub'])*u.mas/u.yr,
                                distance=float(out_params['Source_Dist'])*1e3*u.pc,
                                frame='galactic')
    source_icrs = source_coord_gal.icrs
    params['muS_E'] = source_icrs.pm_ra_cosdec.to_value(u.mas/u.yr)
    params['muS_N'] = source_icrs.pm_dec.to_value(u.mas/u.yr)
    if not (np.isfinite(params['muS_E']) and np.isfinite(params['muS_N'])):
        raise ValueError("Source proper motions transformed to ICRS are non-finite (mas/yr). Check input mul/mub.")

    # Astrometric reference – source position relative to lens at t0 (arcsec)
    # πrel = θEπE
    # µhel = µgeo + µ⊕πrel =θE/tE * πE,geo/πE + µ⊕πEθE, 
    # where µ⊕ ≡ v⊕,⊥/AU and v⊕,⊥ is the transverse velocity of Earth in the frame of the Sun at the peak of the event, projected on the plane of the sky
    # μrel = |μl − μs| mas/yr (geo, helio or L2?)
    mu_rel_E = params['muL_E'] - params['muS_E']  # mas/yr  (geo, helio or L2?)
    mu_rel_N = params['muL_N'] - params['muS_N']
    mu_rel_norm = np.hypot(mu_rel_E, mu_rel_N)
    if mu_rel_norm == 0.0:
        raise ValueError("Relative proper motion vector is zero; cannot define perpendicular u0 orientation.")
    v_hat_E = mu_rel_E / mu_rel_norm
    v_hat_N = mu_rel_N / mu_rel_norm
    # Rotate v_hat 90 deg CCW to get u_hat (perpendicular to relative motion, oriented such that
    
    

    v_hat_vec = np.array([v_hat_E, v_hat_N])
    u_hat_vec = np.array([u_hat_E, u_hat_N])
    thetaS0_mas = abs(u0_gulls) * thetaE
    # What the fuck is xS0
    params['xS0_E'] = (thetaS0_mas * u_hat_E) * 1e-3  # arcsec
    params['xS0_N'] = (thetaS0_mas * u_hat_N) * 1e-3

    # Sanity check against catalog-provided relative PM if available
    if 'murel_ref' in out_params:
        mu_rel_amp = np.hypot(params['muS_E'] - params['muL_E'], params['muS_N'] - params['muL_N'])
        mu_ref = float(out_params['murel_ref'])
        if mu_ref <= 0:
            raise ValueError("murel_ref must be positive if provided.")
        ratio = mu_rel_amp / mu_ref
        if not (0.9 <= ratio <= 1.1):
            raise ValueError(
                f"PM transform mismatch: |mu_rel|={mu_rel_amp:.3f} mas/yr vs murel_ref={mu_ref:.3f} mas/yr. "
                "Check mul convention (pm_l vs pm_l*cos(b)).")

    # Round-trip PM check (ICRS -> Galactic) to catch axis/units mistakes
    # Lens
    lens_icrs_for_rt = SkyCoord(ra=float(out_params['Lens_RA2000.0'])*u.deg,
                                 dec=float(out_params['Lens_DEC2000.0'])*u.deg,
                                 pm_ra_cosdec=params['muL_E']*u.mas/u.yr,
                                 pm_dec=params['muL_N']*u.mas/u.yr,
                                 distance=float(out_params['Lens_Dist'])*1e3*u.pc,
                                 frame='icrs')
    lens_gal_rt = lens_icrs_for_rt.galactic
    dl = abs(lens_gal_rt.pm_l_cosb.to_value(u.mas/u.yr) - float(out_params['Lens_mul']))
    db = abs(lens_gal_rt.pm_b.to_value(u.mas/u.yr) - float(out_params['Lens_mub']))
    if dl > 1e-4 or db > 1e-4:  # 0.0001 mas/yr tolerance
        raise ValueError(f"Lens PM round-trip mismatch: Δpm_l_cosb={dl:.3e}, Δpm_b={db:.3e} mas/yr")

    # Source
    source_icrs_for_rt = SkyCoord(ra=float(out_params['Source_RA2000.0'])*u.deg,
                                   dec=float(out_params['Source_DEC2000.0'])*u.deg,
                                   pm_ra_cosdec=params['muS_E']*u.mas/u.yr,
                                   pm_dec=params['muS_N']*u.mas/u.yr,
                                   distance=float(out_params['Source_Dist'])*1e3*u.pc,
                                   frame='icrs')
    source_gal_rt = source_icrs_for_rt.galactic
    dl_s = abs(source_gal_rt.pm_l_cosb.to_value(u.mas/u.yr) - float(out_params['Source_mul']))
    db_s = abs(source_gal_rt.pm_b.to_value(u.mas/u.yr) - float(out_params['Source_mub']))
    if dl_s > 1e-4 or db_s > 1e-4:
        raise ValueError(f"Source PM round-trip mismatch: Δpm_l_cosb={dl_s:.3e}, Δpm_b={db_s:.3e} mas/yr")

    # Parallax – required (scalar amplitude only for Param1). Directional components (piEN, piEE) are intentionally ignored:
    # Receipt: PSBL_PhotAstrom_Par_Param1 signature includes only |piE|. Relative proper motion provides directional info.
    params['piE'] = float(out_params['piE'])
    if not np.isfinite(params['piE']):
        raise ValueError(f"piE must be finite; got {out_params['piE']!r}")

    # Photometric scaling – required for BAGLE model construction
    params['b_sff'] = [1.0]
    mag_source_val = float(out_params['Source_W146'])  # magnitude
    if not np.isfinite(mag_source_val):
        raise ValueError(f"Source_W146 must be finite magnitude; got {out_params['Source_W146']!r}")
    params['mag_source'] = [mag_source_val]  # W146 magnitude
    # Keep lens mag for photometric blend computation (not passed into BAGLE model)
    mag_lens_val = float(out_params['Lens_W146'])
    if not np.isfinite(mag_lens_val):
        raise ValueError(f"Lens_W146 must be finite magnitude; got {out_params['Lens_W146']!r}")
    params['mag_lens'] = mag_lens_val

    # Lens light contrast between primary (star) and secondary (planet).
    # For star+planet lenses, the planet is effectively dark in W146; encode that
    # with a very large positive delta-mag so its flux contribution is ~0.
    # (BAGLE expects a list.)
    params['dmag_Lp_Ls'] = [99.0]

    # Sky position
    params['raL'] = float(out_params['ra_deg'])
    params['decL'] = float(out_params['dec_deg'])
    if not (np.isfinite(params['raL']) and np.isfinite(params['decL'])):
        raise ValueError(f"ra_deg/dec_deg must be finite sky position (deg); got ra={out_params['ra_deg']!r}, dec={out_params['dec_deg']!r}")

    # Strict requirement: tE must be present in the .out.
    # We do NOT estimate tE from thetaE and proper motions here. If absent, raise.
    if 'tE_helio' in out_params:
        tE_days = float(out_params['tE_helio'])
    else:
        raise KeyError("Missing tE in .out file: expected key 'tE_helio'.")
    if not (np.isfinite(tE_days) and tE_days > 0):
        raise ValueError(f"tE must be positive finite days; got {out_params.get('tE_helio', out_params.get('tE'))!r}")
    params['tE_days'] = tE_days  # keep for internal reparameterization receipts

    params['event_type'] = event_type

    # --- Origin-dependent reparameterization (t0, beta) ---
    # Historical note: GULLS provides `t0lens1` (closest approach to lens 1). BAGLE's PSBL
    # implementation expects lens offsets defined about the midpoint (offset = 0.5*sep*[sin(alpha), cos(alpha)]).
    # Therefore we must convert t0 (closest approach to lens1) into t0_midpoint for BAGLE by projecting
    # the vector from lens1->midpoint onto the trajectory and converting to a time offset.
    #
    # Algebra (units):
    #  - a_ER : separation in Einstein radii (s_dimless)
    #  - delta_x_ER = +0.5 * a_ER  (vector from lens1 to midpoint along binary axis, in ER)
    #  - alpha_rel_rad: angle between trajectory and binary axis (radians)
    #  - projection along motion (parallel) = delta_x_ER * cos(alpha_rel_rad)
    #  - this projection (ER) corresponds to a tau shift of +projection (dimensionless), so
    #       Δt_days = - projection * tE_days  (negative sign: moving origin forward reduces t0)
    #  - perpendicular component adjusts signed impact parameter beta (mas): Δu_ER = delta_x_ER * sin(alpha_rel_rad)
    #       Δu_mas = Δu_ER * thetaE
    # --- Origin-dependent reparameterization (t0, beta) ---
    # Fail-fast validations: do not silently skip reparameterization when inputs are invalid.
    q_dim = float(out_params['Planet_q'])
    if not (np.isfinite(q_dim) and q_dim > 0):
        raise ValueError(f"Planet_q must be positive finite; got {out_params.get('Planet_q')!r}")

    a_ER = s_dimless
    # Vector from lens1 to midpoint (ER). Positive means along the binary axis from lens1 toward midpoint.
    delta_x_ER = 0.5 * a_ER

    # alpha_raw_rel must be present and finite
    if 'alpha_raw_rel' not in params or not np.isfinite(params['alpha_raw_rel']):
        raise ValueError(f"Missing or non-finite alpha_raw_rel in params: {params.get('alpha_raw_rel')!r}")
    alpha_rel_rad = np.deg2rad(params['alpha_raw_rel'])

    # Require relative proper motion to be definable — fail if zero or non-finite
    v_rel_E = float(params.get('muS_E', np.nan)) - float(params.get('muL_E', np.nan))
    v_rel_N = float(params.get('muS_N', np.nan)) - float(params.get('muL_N', np.nan))
    if not (np.isfinite(v_rel_E) and np.isfinite(v_rel_N)):
        raise ValueError(f"Relative proper motion contains non-finite components: muS=({params.get('muS_E')},{params.get('muS_N')}) muL=({params.get('muL_E')},{params.get('muL_N')})")
    v_rel_mag = np.hypot(v_rel_E, v_rel_N)
    if v_rel_mag == 0.0:
        raise ValueError("Relative proper motion vector is zero; cannot determine trajectory direction for origin reparameterization.")

    # Require tE_days present and positive
    if 'tE_days' not in params or not np.isfinite(params['tE_days']) or params['tE_days'] <= 0:
        raise ValueError(f"tE_days missing or non-positive in params: {params.get('tE_days')!r}")
    tE_days_out = float(params['tE_days'])

    # thetaE must be finite positive for converting ER -> mas
    if not (np.isfinite(thetaE) and thetaE > 0):
        raise ValueError(f"thetaE must be positive finite (mas); got {thetaE!r}")

    cos_a = np.cos(alpha_rel_rad)
    sin_a = np.sin(alpha_rel_rad)
    axis_hat_E = v_hat_vec[0] * cos_a - v_hat_vec[1] * sin_a
    axis_hat_N = v_hat_vec[0] * sin_a + v_hat_vec[1] * cos_a
    axis_hat_vec = np.array([axis_hat_E, axis_hat_N])

    delta_vec_ER = delta_x_ER * axis_hat_vec
    proj_parallel_ER = float(np.dot(delta_vec_ER, v_hat_vec))
    delta_t_days = -proj_parallel_ER * tE_days_out

    delta_u_ER = float(np.dot(delta_vec_ER, u_hat_vec))
    delta_u_mas = delta_u_ER * thetaE

    original_t0 = params['t0']
    original_beta = params['beta']
    params['t0'] = original_t0 + delta_t_days
    params['beta'] = original_beta + delta_u_mas

    u_mid_ER = raw_u0 + delta_u_ER
    params['__origin_reparam__'] = {
        'assumed_incoming_origin': 'lens1',
        'delta_x_ER': delta_x_ER,
        'proj_parallel_ER': proj_parallel_ER,
        'delta_t_days': delta_t_days,
        'delta_u_ER': delta_u_ER,
        'delta_u_mas': delta_u_mas,
        'u_mid_ER': u_mid_ER,
        'u_mid_mas': u_mid_ER * thetaE,
        'c_tau_ER': proj_parallel_ER,
        'tE_days_used': tE_days_out,
        't0_original_MJD': original_t0,
        'beta_original_mas': original_beta,
        't0_midpoint_MJD': params['t0'],
        'beta_midpoint_mas': params['beta'],
        'v_rel_mag_mas_per_yr': v_rel_mag,
        'alpha_rel_deg': params['alpha_raw_rel'],
        'phi_v_deg': float(np.degrees(np.arctan2(v_hat_vec[0], v_hat_vec[1]))),
        'phi_axis_deg': float(np.degrees(np.arctan2(axis_hat_E, axis_hat_N))),
    }

    # Store reparameterized midpoint t0 separately
    params['t0_midpoint'] = params['t0']
    axis_angle_deg = (np.degrees(np.arctan2(axis_hat_E, axis_hat_N))) % 360.0
    params['alpha'] = axis_angle_deg

    # Assert sep/thetaE equals Planet_s (unit consistency)
    s_out = float(out_params['Planet_s'])
    if params['thetaE'] == 0:
        raise ValueError("thetaE is zero; cannot validate separation scaling.")
    s_recovered = params['sep'] / params['thetaE']
    # Separation consistency logic:
    # Standard midpoint mapping expects sep = s_out * thetaE (mas) -> s_recovered ≈ s_out.
    # With COM correction enforced (sep replaced by sep_com_corrected) s_recovered can deviate strongly.
    # Detect COM correction by presence of 'sep_midpoint_original' and inequality of sep values.
    if not np.isfinite(s_recovered) or abs(s_recovered - s_out) > max(1e-6, 1e-6*abs(s_out)):
        raise ValueError(f"Separation scaling inconsistency: s_out={s_out:.6g} vs sep/thetaE={s_recovered:.6g}")
    # Receipts: PSBL PhotAstrom Param1 units in docstring for fake_data_PSBL
    # - sep (mas), alpha (degrees), beta (mas), mu (mas/yr), dL/dS (pc)
    #   See BAGLE_Microlensing/src/bagle/fake_data.py:597 (function docstring).

    # Add lightweight debug summary for upstream diagnostics.
    params['__debug_summary__'] = {
        'mLp': params['mLp'],
        'mLs': params['mLs'],
        'q': params['mLs']/params['mLp'] if params['mLp'] != 0 else np.nan,
        't0_MJD': params['t0'],
        't0_original_MJD': params.get('t0_original'),
        't0_midpoint_MJD': params.get('t0_midpoint'),
        't_ref_MJD': params.get('t_ref_mjd'),
        't0_par_MJD': params.get('t0_par'),
        'thetaE_mas': params['thetaE'],
        'u0': params['u0_signed'],
        'u0_amp': params['u0_amp'],
        'beta_signed_mas': params['beta'],
        'beta_mas': params['beta'],
        's': params['sep']/params['thetaE'] if params['thetaE'] != 0 else np.nan,
        'rho': params['rho'],
        'alpha_deg': params['alpha'],
        'alpha_rad': np.deg2rad(params['alpha']),
        'dL_pc': params['dL'],
        'dS_pc': params['dS'],
        'dL_dS': params['dL_dS'],
        'muL_E': params['muL_E'],
        'muL_N': params['muL_N'],
        'muS_E': params['muS_E'],
        'muS_N': params['muS_N'],
        'piE_scalar': params['piE']
    }
    # Sanity: expected list lengths for 1 filter
    if not (isinstance(params['b_sff'], list) and isinstance(params['mag_source'], list)):
        raise TypeError("b_sff and mag_source must be lists (one entry per filter)")
    if not (len(params['b_sff']) == len(params['mag_source']) == 1):
        raise ValueError(f"Expected exactly 1 photometric filter; got b_sff len={len(params['b_sff'])}, mag_source len={len(params['mag_source'])}")

    return params


def create_bagle_model(params: Dict[str, float], event_type: str, use_parallax: bool):
    """Construct the appropriate BAGLE model with correct parameter mapping.

    Critical: use keyword arguments or the exact positional order expected by BAGLE
    classes to avoid unit/semantic misalignment. See PSBL_PhotAstromParam1
    signature around bagle/model.py:6616.
    """
    if event_type == 'PSBL':
        # Prefer finite-source binary lens (FSBL) if available; otherwise, fall back to PSBL with an explicit receipt.
        fs_class_par = getattr(model, 'FSBL_PhotAstrom_Par_Param1', None)
        fs_class_nopar = getattr(model, 'FSBL_PhotAstrom_noPar_Param1', None)
        if use_parallax and fs_class_par is not None:
            # NOTE: FSBL classes are not available in this BAGLE build (commented-out in bagle/model.py lines ~25470-26427).
            # If they become available, wire the correct signature here with rho/utilde mapping and receipts for units.
            raise NotImplementedError("FSBL_PhotAstrom_Par_Param1 is available but not yet wired in validator; add rho/utilde mapping explicitly.")
        if (not use_parallax) and fs_class_nopar is not None:
            raise NotImplementedError("FSBL_PhotAstrom_noPar_Param1 is available but not yet wired in validator; add rho/utilde mapping explicitly.")

        # Receipt: FSBL classes are absent in installed BAGLE; using PSBL instead.
        warnings.warn(
            (
                "Finite-source binary-lens classes (FSBL_*) are not available in this BAGLE build; "
                "falling back to point-source PSBL. Receipt: bagle/model.py shows FSBL sections commented-out "
                "around lines 25470-26427."
            ),
            RuntimeWarning,
        )

        if use_parallax:
            return model.PSBL_PhotAstrom_Par_Param1(
                mLp=params['mLp'], mLs=params['mLs'], t0=params['t0'],
                xS0_E=params['xS0_E'], xS0_N=params['xS0_N'],
                beta=params['beta'],
                muL_E=params['muL_E'], muL_N=params['muL_N'],
                muS_E=params['muS_E'], muS_N=params['muS_N'],
                dL=params['dL'], dS=params['dS'],
                sep=params['sep'], alpha=params['alpha'],
                b_sff=params['b_sff'], mag_src=params['mag_source'], dmag_Lp_Ls=params['dmag_Lp_Ls'],
                raL=params['raL'], decL=params['decL']
            )
        else:
            return model.PSBL_PhotAstrom_noPar_Param1(
                mLp=params['mLp'], mLs=params['mLs'], t0=params['t0'],
                xS0_E=params['xS0_E'], xS0_N=params['xS0_N'],
                beta=params['beta'],
                muL_E=params['muL_E'], muL_N=params['muL_N'],
                muS_E=params['muS_E'], muS_N=params['muS_N'],
                dL=params['dL'], dS=params['dS'],
                sep=params['sep'], alpha=params['alpha'],
                b_sff=params['b_sff'], mag_src=params['mag_source'], dmag_Lp_Ls=params['dmag_Lp_Ls']
            )
    else:
        # Note: BSBL Param1 has additional source binary parameters (sepS, alphaS, etc.).
        # Our validator currently targets PSBL (multiple_sources == 0). If BSBL support is added,
        # we must map those parameters explicitly here with proper units (mas, degrees).
        if use_parallax:
            raise NotImplementedError("BSBL Parallax Param1 construction not yet wired in validator.")
        else:
            raise NotImplementedError("BSBL noParallax Param1 construction not yet wired in validator.")

def compute_bagle_predictions(bagle_model, times_bjd: np.ndarray, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Compute BAGLE model predictions with enhanced, fail-fast poly diagnostics.

    If BAGLE's internal binary-lens solver passes non-finite/degenerate
    coefficient arrays to np.roots, we fail immediately (no best-effort),
    raising a RuntimeError with a structured payload including:
      - raw coefficient vector (dtype preserved)
      - epoch index (loop index inside get_image_pos_arr if recoverable)
      - time at that epoch (MJD)
      - key physical parameters (u0, s, q, alpha_deg, rho, thetaE_mas)
      - upstream complex values (w_i, z1_i, z2_i, m1_i, m2_i) and finite flags
    """
    bad_polys: list[Dict[str, Any]] = []
    orig_roots = np.roots

    # Pull debug summary if present for parameter slice
    dbg = params.get('__debug_summary__', {})

    import inspect

    def roots_wrapper(p, *args, **kwargs):
        # Preserve dtype (float or complex); only coerce to ndarray
        coeff = np.asarray(p)
        # Finite checks: for complex, test both real and imag
        if np.iscomplexobj(coeff):
            finite = np.all(np.isfinite(coeff.real)) and np.all(np.isfinite(coeff.imag))
            mag = np.hypot(coeff.real, coeff.imag)
        else:
            finite = np.all(np.isfinite(coeff))
            mag = np.abs(coeff)
        degenerate = coeff.size == 0 or coeff[0] == 0 or np.any(mag > 1e308)
        if (not finite) or degenerate:
            # Strict diagnostics: require epoch index and upstream locals to exist.
            epoch_idx = None
            locs = None
            for frameinfo in inspect.stack():
                if frameinfo.function == 'get_image_pos_arr':
                    locs = frameinfo.frame.f_locals
                    epoch_idx = locs.get('i')
                    break
            if not isinstance(epoch_idx, (int, np.integer)):
                raise RuntimeError("Polynomial failure: cannot recover integer epoch index 'i' from get_image_pos_arr.")
            if not (0 <= int(epoch_idx) < len(times_bjd)):
                raise RuntimeError(f"Polynomial failure: epoch_idx {epoch_idx} out of bounds for times array of length {len(times_bjd)}.")

            # Enforce presence of upstream variables
            if locs is None:
                raise RuntimeError("Polynomial failure: locals for get_image_pos_arr unavailable.")
            required_locals = ['w', 'z1', 'z2', 'm1', 'm2']
            missing_locals = [name for name in required_locals if name not in locs]
            if missing_locals:
                raise RuntimeError(f"Polynomial failure: missing upstream variables in get_image_pos_arr locals: {missing_locals}")

            wi = locs['w']
            z1 = locs['z1']
            z2 = locs['z2']
            m1 = locs['m1']
            m2 = locs['m2']

            # Index and coerce; m1/m2 may be scalars
            idx = int(epoch_idx)
            try:
                w_i = complex(wi[idx])
                z1_i = complex(z1[idx])
                z2_i = complex(z2[idx])
            except Exception as ex:
                raise RuntimeError(f"Polynomial failure: cannot index upstream arrays at epoch {idx}: {type(ex).__name__}: {ex}")
            try:
                m1_i = complex(m1[idx]) if hasattr(m1, '__len__') else complex(m1)
                m2_i = complex(m2[idx]) if hasattr(m2, '__len__') else complex(m2)
            except Exception as ex:
                raise RuntimeError(f"Polynomial failure: cannot access masses at epoch {idx}: {type(ex).__name__}: {ex}")
            # Build diagnostic record
            # Convert coeff to a JSON-serializable list
            coeff_out = coeff.tolist()
            record = {
                'coeff': coeff_out,
                'epoch_idx': idx,
                't_MJD': float(times_bjd[idx]),
                'u0': dbg.get('u0'),
                's': dbg.get('s'),
                'q': dbg.get('q'),
                'alpha_deg': dbg.get('alpha_deg'),
                'rho': dbg.get('rho'),
                'thetaE_mas': dbg.get('thetaE_mas'),
                'w_i': w_i,
                'z1_i': z1_i,
                'z2_i': z2_i,
                'm1_i': m1_i,
                'm2_i': m2_i,
                'w_i_isfinite': (np.isfinite(w_i.real) and np.isfinite(w_i.imag)),
                'z1_i_isfinite': (np.isfinite(z1_i.real) and np.isfinite(z1_i.imag)),
                'z2_i_isfinite': (np.isfinite(z2_i.real) and np.isfinite(z2_i.imag)),
                'm1_i_isfinite': (np.isfinite(m1_i.real) and np.isfinite(m1_i.imag)),
                'm2_i_isfinite': (np.isfinite(m2_i.real) and np.isfinite(m2_i.imag)),
            }
            bad_polys.append(record)
            # Fail-fast: raise immediately with structured context
            raise RuntimeError(f"Non-finite/degenerate polynomial coefficients encountered: {record}")
        return orig_roots(p, *args, **kwargs)

    np.roots = roots_wrapper
    try:
        # --- Amplification ---
        # (debug removed)
        A = bagle_model.get_amplification(times_bjd, filt_idx=0)
        if A is None:
            raise RuntimeError("BAGLE returned no amplification array (None).")
        A = np.asarray(A, dtype=float)
        if A.shape[0] != len(times_bjd):
            raise RuntimeError(f"BAGLE amplification length {A.shape[0]} != times length {len(times_bjd)}.")
        if not np.all(np.isfinite(A)):
            bad_idx = np.where(~np.isfinite(A))[0][:5]
            raise RuntimeError(f"BAGLE amplification contains non-finite values at indices {bad_idx}.")

        # --- Centroid Shift ---
        # (debug removed)
        shift = bagle_model.get_centroid_shift(times_bjd, filt_idx=0)
        if shift is None:
            raise RuntimeError("BAGLE returned no centroid shift array (None).")
        shift = np.asarray(shift, dtype=float)
        if shift.ndim != 2 or shift.shape[0] != len(times_bjd) or shift.shape[1] != 2:
            raise RuntimeError(f"BAGLE centroid shift has shape {shift.shape}, expected (N,2) with N={len(times_bjd)}.")
        if not np.all(np.isfinite(shift)):
            bad_idx = np.where(~np.isfinite(shift))[0][:5]
            raise RuntimeError(f"BAGLE centroid shift contains non-finite values at indices {bad_idx}.")
    # Construct source proper-motion track (mas) relative to the lens RA/Dec origin
    # srce_pos_mas = xS0_mas + (dt_days / days_per_year) * muS_mas_per_yr
        dt_days = np.asarray(times_bjd, dtype=float) - float(params['t0'])
        days_per_year = 365.25
        xS0_E = float(params.get('xS0_E', 0.0))
        xS0_N = float(params.get('xS0_N', 0.0))
        muS_E = float(params['muS_E'])
        muS_N = float(params['muS_N'])
        source_E_mas = xS0_E + (dt_days / days_per_year) * muS_E
        source_N_mas = xS0_N + (dt_days / days_per_year) * muS_N
        # BAGLE returns centroid shift as (E, N) corresponding to (RA_offset, Dec_offset)
        # Convert to apparent sky-plane centroid (geocentric, lensed ensemble blended as in BAGLE) by
        # adding source proper motion to the shift. N/E ordering follows plotting.py conventions.
        sky_centroid_E_mas = source_E_mas + shift[:, 0]
        sky_centroid_N_mas = source_N_mas + shift[:, 1]

        # Map BAGLE centroid shift into the GULLS lens frame (Einstein radii).
        thetaE_mas = float(params.get('thetaE'))
        if not (np.isfinite(thetaE_mas) and thetaE_mas > 0):
            raise RuntimeError("thetaE must be positive finite to convert centroid shifts into Einstein radii.")
        mu_rel_E = float(params.get('muS_E')) - float(params.get('muL_E'))
        mu_rel_N = float(params.get('muS_N')) - float(params.get('muL_N'))
        alpha_axis_deg = params.get('alpha')
        if alpha_axis_deg is None:
            raise RuntimeError("Params missing lens-frame axis orientation 'alpha'.")
        R_ne_to_xy, rot_diag = _rotation_ne_to_xy(mu_rel_E, mu_rel_N, float(alpha_axis_deg))

        # shift is (E,N); build [N,E] then convert to ER and rotate
        ne_shift_mas = np.column_stack((shift[:, 1], shift[:, 0]))
        ne_shift_er = ne_shift_mas / thetaE_mas
        lens_rel_er = (R_ne_to_xy @ ne_shift_er.T).T  # columns: x, y in Einstein radii relative to source

        out: Dict[str, Any] = {
            'A': A,
            'shift_E': shift[:, 0], 'shift_N': shift[:, 1],
            'sky_centroid_E_mas': sky_centroid_E_mas, 'sky_centroid_N_mas': sky_centroid_N_mas,
            'source_E': source_E_mas, 'source_N': source_N_mas,
            'lens_rel_x': lens_rel_er[:, 0], 'lens_rel_y': lens_rel_er[:, 1],
            'rotation_NE_to_xy': R_ne_to_xy,
            'rotation_diag': rot_diag,
        }
        if bad_polys:
            out['poly_debug'] = bad_polys
        return out
    finally:
        np.roots = orig_roots


def plot_validation(times_bjd, gulls_data, bagle_data, params, output_file, event_label, bagle_model=None):
    times = np.asarray(times_bjd, dtype=float)
    if times.size == 0:
        raise RuntimeError("plot_validation requires non-empty time samples.")
    if 't_ref_mjd' not in params or not np.isfinite(params['t_ref_mjd']):
        raise RuntimeError("plot_validation requires finite t_ref_mjd; validator should have confirmed tref in the .out file.")

    t0_used = float(params['t0'])
    t0_mid = float(params['t0_midpoint'])
    t0_orig = float(params['t0_original'])
    t_ref = float(params['t_ref_mjd'])
    thetaE = float(params['thetaE'])
    for label, value in (
        ('t0_used', t0_used),
        ('t0_midpoint', t0_mid),
        ('t0_original', t0_orig),
        ('t_ref_mjd', t_ref),
        ('thetaE', thetaE),
    ):
        if not np.isfinite(value):
            raise RuntimeError(f"plot_validation requires finite {label}; got {value}")

    rotation_ne_to_xy = np.asarray(bagle_data['rotation_NE_to_xy'], dtype=float)
    if rotation_ne_to_xy.shape != (2, 2):
        raise RuntimeError("rotation_NE_to_xy must be 2x2.")

    def _gulls_series(name: str, label: str) -> np.ndarray:
        arr = _require_series(gulls_data, name, label)
        if arr.shape[0] != times.shape[0]:
            raise RuntimeError(f"{label} length {arr.shape[0]} != time samples {times.shape[0]}")
        return arr

    lens_x = _gulls_series('centroid_x_lens1', 'centroid_x_lens1')
    lens_y = _gulls_series('centroid_y_lens1', 'centroid_y_lens1')
    source_shift_x = _gulls_series('centroid_x_source1', 'centroid_x_source1')
    source_shift_y = _gulls_series('centroid_y_source1', 'centroid_y_source1')
    source_shift_E = _gulls_series('centroid_E_source1_mas', 'centroid_E_source1_mas')
    source_shift_N = _gulls_series('centroid_N_source1_mas', 'centroid_N_source1_mas')
    sky_E = _gulls_series('centroid_E_sky_mas', 'centroid_E_sky_mas')
    sky_N = _gulls_series('centroid_N_sky_mas', 'centroid_N_sky_mas')

    lens_points = np.column_stack((lens_x, lens_y))
    source_shift_xy = np.column_stack((source_shift_x, source_shift_y))
    source_shift_ne = np.column_stack((source_shift_E, source_shift_N))
    sky_points = np.column_stack((sky_E, sky_N))

    source_path = None
    if 'source_x' in gulls_data and 'source_y' in gulls_data:
        try:
            src_x = _gulls_series('source_x', 'source_x (lens frame)')
            src_y = _gulls_series('source_y', 'source_y (lens frame)')
            source_path = np.column_stack((src_x, src_y))
        except RuntimeError:
            source_path = None

    A_gulls = np.asarray(gulls_data['A_deblended'], dtype=float)
    A_bagle = np.asarray(bagle_data['A'], dtype=float)
    if A_gulls.shape != times.shape or A_bagle.shape != times.shape:
        raise RuntimeError("Magnification arrays must match time axis for plotting.")
    A_residual = A_gulls - A_bagle

    sky_bagle = np.column_stack((bagle_data['sky_centroid_E_mas'], bagle_data['sky_centroid_N_mas']))
    residual_E = sky_points[:, 0] - sky_bagle[:, 0]
    residual_N = sky_points[:, 1] - sky_bagle[:, 1]

    A_gulls = np.asarray(gulls_data['A_deblended'], dtype=float)
    A_bagle = np.asarray(bagle_data['A'], dtype=float)
    if A_gulls.shape != times.shape or A_bagle.shape != times.shape:
        raise RuntimeError("Magnification arrays must match time axis for plotting.")
    A_residual = A_gulls - A_bagle

    residual_E = sky_relative_gulls[:, 0] - sky_relative_bagle[:, 0]
    residual_N = sky_relative_gulls[:, 1] - sky_relative_bagle[:, 1]

    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=times.min(), vmax=times.max())

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.3)

    def _vector_base(arrays: List[np.ndarray | None]) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        data = [a for a in arrays if a is not None and np.size(a)]
        if not data:
            return np.zeros(2), 1.0, np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        stacked = np.vstack(data)
        min_vals = np.nanmin(stacked, axis=0)
        max_vals = np.nanmax(stacked, axis=0)
        ranges = max_vals - min_vals
        span = float(np.nanmax(ranges))
        if not np.isfinite(span) or span <= 0.0:
            span = max(np.linalg.norm(stacked[0]), 1.0)
        base = (min_vals + max_vals) * 0.5
        margin = 0.05 * span
        bounds_min = min_vals - margin
        bounds_max = max_vals + margin
        return base, 0.2 * span, bounds_min, bounds_max

    def _clamp_point(point: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(point, bounds_min), bounds_max)

    def _shrink_to_bounds(base: np.ndarray, direction: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
        factor = 1.0
        for dim in range(2):
            comp = direction[dim]
            if comp > 0:
                allowed = bounds_max[dim] - base[dim]
                if allowed <= 0:
                    return np.zeros_like(direction)
                factor = min(factor, allowed / comp)
            elif comp < 0:
                allowed = bounds_min[dim] - base[dim]
                if allowed >= 0:
                    return np.zeros_like(direction)
                factor = min(factor, allowed / comp)
        return direction * max(min(factor * 0.9, 1.0), 0.0)

    def _draw_axes(ax, base, vectors, labels, colors, scale, bounds_min, bounds_max):
        for vec, lab, color in zip(vectors, labels, colors):
            if vec is None:
                continue
            norm = np.linalg.norm(vec)
            if norm == 0 or not np.isfinite(norm):
                continue
            direction = (vec / norm) * scale
            direction = _shrink_to_bounds(base, direction, bounds_min, bounds_max)
            if not np.any(direction):
                continue
            end = base + direction
            ax.annotate('', xy=end, xytext=base, arrowprops=dict(color=color, width=0.6, headwidth=5, alpha=0.8))
            unit = direction / (np.linalg.norm(direction) + 1e-12)
            perp = np.array([-unit[1], unit[0]])
            label_offset = 0.04 * scale * unit + 0.02 * scale * perp
            label_pos = _clamp_point(end + label_offset, bounds_min, bounds_max)
            ax.text(*label_pos, lab, color=color, fontsize=8, weight='bold', ha='center', va='center')

    ax_mag = fig.add_subplot(gs[0, 0])
    ax_mag.plot(times, A_gulls, 'b.', markersize=2, alpha=0.6, label='GULLS')
    ax_mag.plot(times, A_bagle, 'r-', linewidth=1.2, alpha=0.8, label='BAGLE')
    ref_lines = [
        (t0_used, 't0_used', 'k'),
        (t0_orig, 't0_orig', 'purple'),
        (t0_mid, 't0_mid', 'gray'),
        (t_ref, 'tref', 'green'),
    ]
    for value, label, color in ref_lines:
        if value is None or not np.isfinite(value):
            continue
        ax_mag.axvline(value, color=color, linestyle='--', alpha=0.45, linewidth=1.0, label=f"{label}={value:.2f}")
    ax_mag.set_xlabel('Time (MJD)')
    ax_mag.set_ylabel('Magnification A')
    ax_mag.set_title(f'{event_label}: magnification comparison')
    ax_mag.legend(fontsize=8)
    ax_mag.grid(alpha=0.3)

    ax_src = fig.add_subplot(gs[0, 1])
    ax_src.scatter(
        source_shift_ne[:, 0],
        source_shift_ne[:, 1],
        c=times,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.8,
        label='GULLS shift (Roman/L2)',
    )
    # BAGLE source-rest overlay will be added here once BAGLE exports Roman/L2-aligned centroid shifts.
    ax_src.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
    ax_src.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
    ax_src.set_xlabel('East shift (mas)')
    ax_src.set_ylabel('North shift (mas)')
    ax_src.set_title('Source-rest centroid shifts (L2-centric observer; source at origin)')
    ax_src.axis('equal')
    ax_src.grid(alpha=0.3)
    ax_src.legend(fontsize=8)

    ax_lens = fig.add_subplot(gs[1, 0])
    ax_lens.scatter(
        lens_points[:, 0],
        lens_points[:, 1],
        c=times,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.85,
        label='GULLS centroid',
    )
    if source_path is not None:
        ax_lens.plot(
            source_path[:, 0],
            source_path[:, 1],
            color='gray',
            linewidth=1.0,
            alpha=0.7,
            label='Source trajectory',
        )
    # BAGLE lens-frame centroid overlay will be drawn here once BAGLE provides lens1-referenced centroids.
    lens1_x = float(np.asarray(gulls_data['lens1_x'], dtype=float)[0])
    lens1_y = float(np.asarray(gulls_data['lens1_y'], dtype=float)[0])
    lens2_x = float(np.asarray(gulls_data['lens2_x'], dtype=float)[0])
    lens2_y = float(np.asarray(gulls_data['lens2_y'], dtype=float)[0])
    ax_lens.scatter([lens1_x], [lens1_y], marker='s', color='k', s=35, label='Lens 1')
    ax_lens.scatter([lens2_x], [lens2_y], marker='s', facecolors='none', edgecolors='k', s=35, label='Lens 2')
    ax_lens.set_xlabel('x (Einstein radii)')
    ax_lens.set_ylabel('y (Einstein radii)')
    ax_lens.set_title('Lens-plane centroids (lens rest frame; \nL2-centric, origin at lens1 position, x-axis along lens axis)')
    ax_lens.axis('equal')
    ax_lens.grid(alpha=0.3)
    ax_lens.legend(fontsize=8)

    base_lens, scale_lens, bounds_min_lens, bounds_max_lens = _vector_base(
        [lens_points, source_path]
    )
    north_vec_lens = rotation_ne_to_xy @ np.array([1.0, 0.0])
    east_vec_lens = rotation_ne_to_xy @ np.array([0.0, 1.0])
    _draw_axes(
        ax_lens,
        base_lens,
        [east_vec_lens, north_vec_lens],
        ['E', 'N'],
        ['tab:orange', 'tab:green'],
        scale_lens,
        bounds_min_lens,
        bounds_max_lens,
    )

    ax_sky = fig.add_subplot(gs[1, 1])
    ax_sky.scatter(
        sky_points[:, 0],
        sky_points[:, 1],
        c=times,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.85,
        label='GULLS centroid',
    )
    # BAGLE sky-centroid overlay will be added here when BAGLE emits the Roman/L2 sky track.
    ax_sky.axhline(0.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.9)
    ax_sky.axvline(0.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.9)
    ax_sky.set_xlabel('East offset (mas)')
    ax_sky.set_ylabel('North offset (mas)')
    ax_sky.set_title('Sky centroids (L2-centric sky rest frame; origin fixed at lens1 RA/Dec at t=t0lens1)')
    ax_sky.axis('equal')
    ax_sky.grid(alpha=0.3)
    ax_sky.legend(fontsize=8)

    rot_inv = np.linalg.inv(rotation_ne_to_xy)
    base_sky, scale_sky, bounds_min_sky, bounds_max_sky = _vector_base([sky_points, None])
    lens_x_ne = rot_inv @ np.array([1.0, 0.0])
    lens_y_ne = rot_inv @ np.array([0.0, 1.0])
    # Convert (N,E) ordering to (E,N)
    lens_x_en = np.array([lens_x_ne[1], lens_x_ne[0]])
    lens_y_en = np.array([lens_y_ne[1], lens_y_ne[0]])
    _draw_axes(
        ax_sky,
        base_sky,
        [lens_x_en, lens_y_en],
        ['x', 'y'],
        ['tab:purple', 'tab:brown'],
        scale_sky,
        bounds_min_sky,
        bounds_max_sky,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_lens, ax_sky], fraction=0.046, pad=0.04)
    cbar.set_label('Time (MJD)')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(event_label, fontsize=14)
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def validate_event(out_file, lc_file, simulation_zero_time, multiple_sources, params_dict: Optional[Dict[str, str]] = None, verbose=True):
    # Fail-fast top-level: do NOT swallow exceptions here. Upstream runner should treat failures as failures.
    out_params = parse_out_file(out_file)
    event_type = 'BSBL' if multiple_sources == 1 else 'PSBL'
    if verbose:
        print(f"\n  Event: {lc_file.stem}")
        print(f"    Type: {event_type}")
    bagle_params = convert_to_bagle_params(out_params, simulation_zero_time, event_type)
    tref_mjd = bagle_params.get('t_ref_mjd')
    if tref_mjd is None or not np.isfinite(tref_mjd):
        raise RuntimeError(
            "BAGLE validator expected a finite t_ref_mjd (info.cpp must emit tref in simulation days so we can add simulation_zero_time). "
            "Missing tref makes BAGLE/GULLS alignment undefined; ensure every .out row writes it."
        )
    if params_dict is None:
        raise RuntimeError("Validator requires parameter dictionary to resolve observer and flags deterministically; received None.")

    piE_scalar = bagle_params.get('piE')
    if piE_scalar is None or not np.isfinite(piE_scalar):
        raise RuntimeError("convert_to_bagle_params failed to produce finite piE scalar")

    parallax_raw = params_dict.get('PARALLAX')
    if parallax_raw is None:
        raise RuntimeError("Parameter dictionary missing PARALLAX flag; cannot determine parallax mode.")
    try:
        parallax_flag_int = int(float(parallax_raw))
    except ValueError as exc:
        raise RuntimeError(f"Parameter PARALLAX value '{parallax_raw}' is not an integer flag") from exc
    parallax_enabled = parallax_flag_int != 0
    piE_abs = abs(piE_scalar)
    piE_tol = 1e-9
    if parallax_enabled and piE_abs <= piE_tol:
        raise RuntimeError(f"Parameter file sets PARALLAX={parallax_flag_int} but |piE|={piE_abs:.3g}; tref/physics emission lost parallax receipts.")
    if (not parallax_enabled) and piE_abs > piE_tol:
        raise RuntimeError(f"Parameter file sets PARALLAX=0 but |piE|={piE_abs:.3g}; pipeline wrote parallax when disabled.")

    if verbose:
        dbg = bagle_params.get('__debug_summary__', {})
        print(
            "    ParamSummary:" +
            f" q={dbg.get('q'):.3g} s={dbg.get('s'):.3g} u0={dbg.get('u0'):.3g} rho={dbg.get('rho'):.3g}" +
            f" thetaE={dbg.get('thetaE_mas'):.3g} beta={dbg.get('beta_mas'):.3g} alpha(deg)={dbg.get('alpha_deg'):.3g}" +
            f" | muL=({dbg.get('muL_E'):.3g},{dbg.get('muL_N'):.3g}) muS=({dbg.get('muS_E'):.3g},{dbg.get('muS_N'):.3g})" +
            f" | dL={dbg.get('dL_pc'):.3g} dS={dbg.get('dS_pc'):.3g} piE={dbg.get('piE_scalar'):.3g}"
        )
    times_bjd, gulls_data, lc_meta = load_lightcurve(lc_file)
    # Timescale diagnostic (explicit receipt): Compare TDB vs TT for a small sample to
    # confirm we are operating within expected sub-millisecond modulation (TT-TDB periodic ~1.6 ms).
    # We do not adjust times; only record amplitude. If amplitude exceeds 5 ms, raise to force review.
    if times_bjd.size >= 3:
        import astropy.time as _at
        sample = times_bjd[::max(1, times_bjd.size // 5)]  # up to 5 samples
        t_tdb = _at.Time(sample, format='mjd', scale='tdb')
        t_tt = _at.Time(sample, format='mjd', scale='tt')
        dt_sec = (t_tt.tdb.jd - t_tt.tt.jd) * 86400.0
        max_dt = float(np.max(np.abs(dt_sec)))
        avg_dt = float(np.mean(dt_sec))
        bagle_params['__timescale_diag__'] = {
            'sample_size': int(sample.size),
            'max_|TT-TDB|_sec': max_dt,
            'mean_(TT-TDB)_sec': avg_dt
        }
        if max_dt > 0.005:
            raise RuntimeError(f"TT-TDB delta {max_dt:.4f}s exceeds 5 ms threshold; verify emitted BJD scale")
        if verbose:
            print(f"    TimeScaleDiag: max|TT-TDB|={max_dt*1e3:.3f} ms mean={avg_dt*1e3:.3f} ms (receipt only)")
    use_parallax = parallax_enabled
    bagle_model = create_bagle_model(bagle_params, event_type, use_parallax)
    # --- Observer: deterministically resolve from GULLS params and apply to BAGLE ---
    used_obs, obs_receipt = _resolve_and_apply_bagle_observer(
        bagle_model, params_dict, bagle_params['raL'], bagle_params['decL'], times_bjd, verbose
    )
    if verbose:
        print(f"    ObserverReceipt: obsLocation='{used_obs}' source=Horizons file={obs_receipt['gulls_obs_file']}")
    # Receipt: print the exact time range that will be sent to Horizons (if a spacecraft is used)
    min_time_mjd = float(np.min(times_bjd)) 
    max_time_mjd = float(np.max(times_bjd))
    if used_obs == '-211':
        import astropy.time as _at
        bd_mjd = _EARLIEST_ROMAN_MJD
        t_min_iso = _at.Time(min_time_mjd, format='mjd', scale='tdb').iso if np.isfinite(min_time_mjd) else 'nan'
        t_max_iso = _at.Time(max_time_mjd, format='mjd', scale='tdb').iso if np.isfinite(max_time_mjd) else 'nan'
        t_bd_iso = _at.Time(bd_mjd, format='mjd', scale='tdb').iso
        print(
            "    HorizonsQuery: target='-211' (Roman)\n"
            f"      min_MJD_TDB={min_time_mjd:.6f}  min_JD_TDB={min_time_mjd + 2400000.5:.6f}  min_ISO_TDB={t_min_iso}\n"
            f"      max_MJD_TDB={max_time_mjd:.6f}  max_JD_TDB={max_time_mjd + 2400000.5:.6f}  max_ISO_TDB={t_max_iso}\n"
            f"      earliest_supported_MJD_TDB={bd_mjd:.6f}  earliest_ISO_TDB={t_bd_iso}  (guard +0.05 d applies)"
        )
    # Pre-flight Roman ephemeris boundary guard: fail with explicit receipt before network call if times begin too close
    # to the published lower boundary (Horizons error can be cryptic when off by minutes). We require a 0.05-day safety margin.
    if used_obs == '-211':
        min_time_mjd = float(np.min(times_bjd)) if times_bjd.size else np.nan
        if np.isfinite(min_time_mjd) and min_time_mjd < _EARLIEST_ROMAN_MJD + 0.05:
            raise RuntimeError(
                f"Roman ephemeris pre-boundary: earliest MJD={min_time_mjd:.6f} is < boundary+0.05d ({_EARLIEST_ROMAN_MJD + 0.05:.6f}). "
                f"Shift SIMULATION_ZERO_TIME forward (>= {_EARLIEST_ROMAN_JD + 0.05:.5f} JD) or trim early lightcurve epochs."
            )
    # --- Geocentric correction (tref) approximate recentering of t0 ---
    # Goal: In geocentric formulation, tau_total(t) = (t - t0_mid)/tE + tshift(t) with tshift(tref)=0.
    # Closest approach occurs when tau_total ≈ 0. Using first-order approximation, set
    #   t0_geo ≈ t0_mid - tE * tshift(t0_mid).
    # We approximate tshift from BAGLE's parallax vector pvec(E,N) by constructing N/E shifts relative to tref
    # with linear term removed, then projecting onto the mu_rel direction (phi_v).
    if use_parallax:
        t0_mid = float(bagle_params['t0_midpoint'])
        t_ref = float(bagle_params['t_ref_mjd'])
        tE_days = float(bagle_params['tE_days'])
        piE_amp = float(bagle_params['piE'])
        for label, value in (
            ('t0_midpoint', t0_mid),
            ('t_ref_mjd', t_ref),
            ('tE_days', tE_days),
            ('piE', piE_amp),
        ):
            if not np.isfinite(value):
                raise RuntimeError(f"GeoReparam requires finite {label}; got {value}")
        if tE_days <= 0:
            raise RuntimeError(f"GeoReparam requires positive tE_days; got {tE_days}")
        dt = 0.5
        t_arr = np.array([t0_mid, t_ref - dt, t_ref, t_ref + dt], dtype=float)
        pvec = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], t_arr, obsLocation=used_obs)
        E_t0, N_t0 = float(pvec[0, 0]), float(pvec[0, 1])
        E_m, N_m = float(pvec[1, 0]), float(pvec[1, 1])
        E_ref, N_ref = float(pvec[2, 0]), float(pvec[2, 1])
        E_p, N_p = float(pvec[3, 0]), float(pvec[3, 1])
        dE_dt = (E_p - E_m) / (2 * dt)
        dN_dt = (N_p - N_m) / (2 * dt)
        dt_rel = float(t0_mid - t_ref)
        Eshift = (E_t0 - E_ref) - dt_rel * dE_dt
        Nshift = (N_t0 - N_ref) - dt_rel * dN_dt
        mu_rel_E = bagle_params['muS_E'] - bagle_params['muL_E']
        mu_rel_N = bagle_params['muS_N'] - bagle_params['muL_N']
        phi_v = float(np.arctan2(mu_rel_E, mu_rel_N))
        cs = np.cos(phi_v)
        sn = np.sin(phi_v)
        tshift = -piE_amp * (Nshift * cs + Eshift * sn)
        t0_geo = float(t0_mid - tE_days * tshift)
        bagle_params['t0_geocentric'] = t0_geo
        bagle_params['__geo_reparam__'] = {
            'dt_days': dt,
            'Eshift': Eshift,
            'Nshift': Nshift,
            'dE_dt': dE_dt,
            'dN_dt': dN_dt,
            'phi_v_deg': float(np.degrees(phi_v)),
            'tshift_at_t0': float(tshift),
            't0_mid_MJD': float(t0_mid),
            't_ref_MJD': float(t_ref),
            't0_geo_MJD': t0_geo,
        }
        bagle_params['t0'] = t0_geo
        bagle_model.t0 = t0_geo
        if verbose:
            if '__origin_reparam__' not in bagle_params:
                raise RuntimeError("Origin reparameterization receipt missing prior to GeoReparam log.")
            ore = bagle_params['__origin_reparam__']
            dt0_origin = ore['t0_midpoint_MJD'] - ore['t0_original_MJD']
            dt0_geo = t0_geo - t0_mid
            print(
                "    GeoReparam:"
                f" tshift(t0)={tshift:.4g}  dE_dt={dE_dt:.3e} dN_dt={dN_dt:.3e} AU/day"
                f" | Δt0_origin={dt0_origin:.3g} d from COM→midpoint, Δt0_geo={dt0_geo:.3g} d from midpoint→geo"
            )
    if parallax_enabled and 't0_geocentric' not in bagle_params:
        raise RuntimeError("Geocentric reparameterization missing while parallax enabled")

    if verbose:
        dbg = bagle_params.get('__debug_summary__', {})
        t0_orig = dbg.get('t0_original_MJD')
        t0_mid = dbg.get('t0_midpoint_MJD')
        t0_used = dbg.get('t0_MJD')
        t0_par = dbg.get('t0_par_MJD')
        for label, value in (
            ("t0_original_MJD", t0_orig),
            ("t0_midpoint_MJD", t0_mid),
            ("t0_MJD", t0_used),
            ("t0_par_MJD", t0_par),
        ):
            if value is None or not np.isfinite(value):
                raise RuntimeError(f"Debug summary missing finite {label}; cannot print TimeRefs receipt")
        dt_mid_geo = np.nan
        if 't0_geocentric' in bagle_params:
            geo_receipt = bagle_params['t0_geocentric']
            if not np.isfinite(geo_receipt):
                raise RuntimeError("bagle_params['t0_geocentric'] present but non-finite")
            dt_mid_geo = geo_receipt - t0_mid
        print(
            "    TimeRefs: "
            f"t0_orig={t0_orig:.5f} "
            f"t0_mid={t0_mid:.5f} "
            f"t0_used={t0_used:.5f} "
            f"t0_par={t0_par:.5f} "
            f"Δ(t0_used - t0_mid)={dt_mid_geo:.5f} d"
        )

        if '__origin_reparam__' not in bagle_params:
            raise RuntimeError("convert_to_bagle_params failed to attach '__origin_reparam__' receipt")
        ore = bagle_params['__origin_reparam__']
        required_origin_keys = (
            'tE_days_used',
            'delta_u_mas',
            'c_tau_ER',
            'u_mid_ER',
            'delta_x_ER',
            't0_midpoint_MJD',
            't0_original_MJD',
        )
        for key in required_origin_keys:
            val = ore.get(key)
            if val is None or not np.isfinite(val):
                raise RuntimeError(f"Origin reparameterization missing finite '{key}' receipt")
        dt0_days = ore['t0_midpoint_MJD'] - ore['t0_original_MJD']
        print(
            "    OriginReparam:"
            f" tE_used={ore['tE_days_used']:.3f} d"
            f" d_t0={dt0_days:.3g}"
            f" d_beta={ore['delta_u_mas']:.3g} mas"
            f" c_tau_ER={ore['c_tau_ER']:.3g}"
            f" u_mid_ER={ore['u_mid_ER']:.3g}"
            f" delta_x_ER={ore['delta_x_ER']:.3g}"
        )
    # Parallax alignment audit (fail-fast) if parallax active.
    parallax_diag: Optional[Dict[str, Any]] = None
    if use_parallax:
        parallax_diag = audit_parallax_alignment(
            ra_deg=bagle_params['raL'], dec_deg=bagle_params['decL'],
            times_mjd=times_bjd, bagle_model=bagle_model, obs_location=used_obs
        )
        if verbose:
            print(
                "    ParallaxAudit: median_rel_err={:.3g} max_rel_err={:.3g} component_max_diff_mas={:.3g}".format(
                    parallax_diag['median_rel_err'], parallax_diag['max_rel_err'], parallax_diag['component_abs_max_diff_mas']
                )
            )
        import astropy.time as _at
        t_tt = _at.Time(times_bjd, format='mjd', scale='tt')
        dt_days = (t_tt.tdb.jd - t_tt.tt.jd)
        p_t = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], times_bjd, obsLocation=used_obs)
        p_t_plus = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], times_bjd + dt_days, obsLocation=used_obs)
        dp = (p_t_plus - p_t)
        piL_mas = getattr(bagle_model, 'piL')
        dmas = np.hypot(*(piL_mas * dp).T)
        max_dmas = float(np.max(np.abs(dmas)))
        print(f"    TimeScaleDiag(parallax): max|Δparallax|≈{max_dmas*1e3:.3f} μas if TT misfed as TDB")
    # Parallax components (piEN, piEE) are ignored for Param1; no informational print needed.
    try:
        bagle_data = compute_bagle_predictions(bagle_model, times_bjd, bagle_params)
    except Exception as e_compute:
        # Print compact diagnostics around t0
        t0 = bagle_params['t0']
        if len(times_bjd) > 0 and np.isfinite(t0):
            idx = int(np.argmin(np.abs(times_bjd - t0)))
            t_slice = times_bjd[max(0, idx-3): min(len(times_bjd), idx+4)]
            print(f"    BAGLE compute failed near t0. Times around t0 (MJD): {t_slice}")
        else:
            print("    BAGLE compute failed but cannot determine t0 or times for diagnostics.")
        print(f"    Exception: {e_compute.__class__.__name__}: {e_compute}")
        print("    Re-raising with context. Params summary above.")
        raise
    
    gulls_data['A_deblended'] = calculate_magnification_from_lightcurve(lc_meta, gulls_data)
    A_diff = gulls_data['A_deblended'] - bagle_data['A']

    # Persist a compact sidecar with BAGLE arrays for downstream overlays (plots).
    # Explicit ordering: BAGLE returns centroid shifts as (East, North) per bagle.parallax.parallax_in_direction docstrings.
    # We store both the explicit 'bagle_shift_E_mas'/'bagle_shift_N_mas' keys used by plotting/diagnostics
    # and alias keys 'shift_E'/'shift_N'.
    sidecar = {
        'times_mjd': times_bjd.astype(float),
        'bagle_A': np.asarray(bagle_data['A'], dtype=float),
        # Principal keys (explicit E,N ordering in mas)
        'bagle_shift_E_mas': np.asarray(bagle_data.get('shift_E', []), dtype=float),
        'bagle_shift_N_mas': np.asarray(bagle_data.get('shift_N', []), dtype=float),
        # Backwards-compatible aliases (same arrays as above)
        'shift_E': np.asarray(bagle_data.get('shift_E', []), dtype=float),
        'shift_N': np.asarray(bagle_data.get('shift_N', []), dtype=float),
        # Apparent sky-plane centroid (geocentric frame, lensed ensemble) in mas
        'bagle_sky_centroid_E_mas': np.asarray(bagle_data.get('sky_centroid_E_mas', []), dtype=float),
        'bagle_sky_centroid_N_mas': np.asarray(bagle_data.get('sky_centroid_N_mas', []), dtype=float),
    # Deprecated compatibility aliases for downstream tooling expecting the old naming.
    'centroid_abs_E_mas': np.asarray(bagle_data.get('sky_centroid_E_mas', []), dtype=float),
    'centroid_abs_N_mas': np.asarray(bagle_data.get('sky_centroid_N_mas', []), dtype=float),
        'bagle_lens_rel_x_er': np.asarray(bagle_data.get('lens_rel_x', []), dtype=float),
        'bagle_lens_rel_y_er': np.asarray(bagle_data.get('lens_rel_y', []), dtype=float),
        'rotation_NE_to_lens': np.asarray(bagle_data.get('rotation_NE_to_lens', np.empty((2, 2)))),
        'thetaE_mas': float(bagle_params.get('thetaE', np.nan)),
    }
    sidecar_path = lc_file.parent / f"{lc_file.stem}_bagle.npz"
    np.savez(sidecar_path, **sidecar)
    # --- Astrometry comparison ---
    # Lens-frame consistency (relative to source position in Einstein radii)
    lens_true_x = _require_series(gulls_data, 'centroid_x_lens1', 'centroid_x_lens1')
    lens_true_y = _require_series(gulls_data, 'centroid_y_lens1', 'centroid_y_lens1')
    gulls_rel_x = _require_series(gulls_data, 'centroid_x_source1', 'centroid_x_source1')
    gulls_rel_y = _require_series(gulls_data, 'centroid_y_source1', 'centroid_y_source1')
    bagle_rel_x = np.asarray(bagle_data.get('lens_rel_x', []), dtype=float)
    bagle_rel_y = np.asarray(bagle_data.get('lens_rel_y', []), dtype=float)
    if not (lens_true_x.size == gulls_rel_x.size == bagle_rel_x.size == len(times_bjd)):
        raise RuntimeError("Lens-frame arrays length mismatch between GULLS and BAGLE predictions.")
    gulls_data['true_x_centroid'] = lens_true_x
    gulls_data['true_y_centroid'] = lens_true_y
    gulls_data['lens_rel_x'] = gulls_rel_x
    gulls_data['lens_rel_y'] = gulls_rel_y

    lens_rel_diff_x = gulls_rel_x - bagle_rel_x
    lens_rel_diff_y = gulls_rel_y - bagle_rel_y
    lens_rel_rms = float(np.sqrt(np.mean(lens_rel_diff_x**2 + lens_rel_diff_y**2)))

    # Sky-plane NE comparison (mas) using direct L2-centric outputs
    gulls_sky_centroid_E_mas = _require_series(gulls_data, 'centroid_E_sky_mas', 'centroid_E_sky_mas')
    gulls_sky_centroid_N_mas = _require_series(gulls_data, 'centroid_N_sky_mas', 'centroid_N_sky_mas')
    bagle_sky_centroid_N_mas = np.asarray(
        bagle_data['sky_centroid_N_mas'], dtype=float)
    bagle_sky_centroid_E_mas = np.asarray(
        bagle_data['sky_centroid_E_mas'], dtype=float)
    if (bagle_sky_centroid_N_mas.size != bagle_sky_centroid_E_mas.size or
            bagle_sky_centroid_N_mas.size != len(times_bjd)):
        raise RuntimeError("Sky-plane astrometry arrays length mismatch between GULLS and BAGLE predictions.")

    gulls_data['sky_centroid_N_mas'] = gulls_sky_centroid_N_mas
    gulls_data['sky_centroid_E_mas'] = gulls_sky_centroid_E_mas
    gulls_data['true_N_centroid_mas'] = gulls_sky_centroid_N_mas
    gulls_data['true_E_centroid_mas'] = gulls_sky_centroid_E_mas

    N_diff = gulls_sky_centroid_N_mas - bagle_sky_centroid_N_mas
    E_diff = gulls_sky_centroid_E_mas - bagle_sky_centroid_E_mas
    sky_offset_N_mean = float(np.mean(gulls_sky_centroid_N_mas - bagle_sky_centroid_N_mas))
    sky_offset_E_mean = float(np.mean(gulls_sky_centroid_E_mas - bagle_sky_centroid_E_mas))
    sky_offset_mean_mag = float(np.hypot(sky_offset_N_mean, sky_offset_E_mean))

    print(f"    Deblended magnification using fs={fs:.6g} from header (#fs). No heuristics applied.")
    results = {
        'event_type': event_type,
        'A_rms': np.sqrt(np.mean(A_diff**2)),
        'N_rms': np.sqrt(np.mean(N_diff**2)),
        'E_rms': np.sqrt(np.mean(E_diff**2)),
        'total_astrom_rms': np.sqrt(np.mean(N_diff**2 + E_diff**2)),
        'lens_frame_rms': lens_rel_rms,
        'lens_frame_rms_x': float(np.sqrt(np.mean(lens_rel_diff_x**2))),
        'lens_frame_rms_y': float(np.sqrt(np.mean(lens_rel_diff_y**2))),
        'sky_mean_offset_N_mas': sky_offset_N_mean,
        'sky_mean_offset_E_mas': sky_offset_E_mean,
        'sky_mean_offset_total_mas': sky_offset_mean_mag,
    }
    if verbose:
        print(
            "    AstrometryRMS: lens_frame={:.3g} ER (x={:.3g}, y={:.3g}) | sky_plane={:.3g} mas".format(
                results['lens_frame_rms'], results['lens_frame_rms_x'], results['lens_frame_rms_y'], results['total_astrom_rms']
            )
        )
    if parallax_diag is not None:
        results['parallax_diag'] = parallax_diag

    # Fail-fast validation criteria focused strictly on physical agreement (no cosmetic alignment):
    # 1. Magnification RMS threshold (0.5) — large divergence implies wrong event construction or unit mismatch.
    # 2. Astrometric total RMS should be < 0.3 * thetaE (scale set by Einstein radius).
    # 3. Magnification correlation ≥ 0.95 (allowing finite-source vs point-source moderate differences but still requiring temporal alignment of peak).
    # 4. Peak time lag |Δt_peak| must be ≤ 2 days (receipt: light curve generated spans hundreds of days; larger offsets suggest time origin mismatch).
    # NOTE: We no longer fail purely because FSBL is unavailable; physical mismatch must be expressed in metrics.
    thetaE_mas = bagle_params['thetaE']
    if A_gulls.size != bagle_data['A'].size:
        raise RuntimeError(
            f"Magnification array length mismatch: GULLS={A_gulls.size} BAGLE={bagle_data['A'].size}"
        )
    if A_gulls.size <= 1:
        raise RuntimeError("Magnification arrays must contain more than one sample to compute correlation.")
    A_corr = float(np.corrcoef(A_gulls, bagle_data['A'])[0, 1])
    fail_reasons = []
    if results['A_rms'] > 0.5:
        fail_reasons.append(f"Magnification RMS {results['A_rms']:.3g} > 0.5 threshold")
    if thetaE_mas > 0 and results['total_astrom_rms'] > 0.3 * thetaE_mas:
        fail_reasons.append(f"Astrometric RMS {results['total_astrom_rms']:.3g} mas > 0.3*thetaE ({0.3*thetaE_mas:.3g} mas)")
    sky_offset_tol = 0.05 * thetaE_mas if (np.isfinite(thetaE_mas) and thetaE_mas > 0) else 0.05
    results['sky_mean_offset_tol_mas'] = sky_offset_tol
    if results['sky_mean_offset_total_mas'] > sky_offset_tol:
        fail_reasons.append(
            f"Sky centroid mean offset {results['sky_mean_offset_total_mas']:.3g} mas > {sky_offset_tol:.3g} mas tolerance"
        )
    if not np.isnan(A_corr) and A_corr < 0.95:
        fail_reasons.append(f"Magnification correlation {A_corr:.4f} < 0.95")
    lens_rel_threshold = 5e-3
    if results['lens_frame_rms'] > lens_rel_threshold:
        fail_reasons.append(
            f"Lens-frame centroid RMS {results['lens_frame_rms']:.3g} ER > {lens_rel_threshold:.3g} ER threshold"
        )
    t_peak_gulls = float(times_bjd[np.argmax(A_gulls)])
    t_peak_bagle = float(times_bjd[np.argmax(bagle_data['A'])])
    peak_lag = abs(t_peak_gulls - t_peak_bagle)
    if peak_lag > 0.1:
        fail_reasons.append(f"Peak time lag {peak_lag:.3f} days > 0.1 day threshold")

    if fail_reasons:
        # Augment diagnostics specifically for peak lag to expose motion vectors.
        if any('Peak time lag' in r for r in fail_reasons):
            mu_rel_E = bagle_params['muS_E'] - bagle_params['muL_E']
            mu_rel_N = bagle_params['muS_N'] - bagle_params['muL_N']
            mu_rel_amp = np.hypot(mu_rel_E, mu_rel_N)
            mu_rel_posang_deg = (np.degrees(np.arctan2(mu_rel_E, mu_rel_N)) % 360.0)
            piE_scalar = bagle_params['piE']
            print(f"    PeakLagContext: mu_rel_amp={mu_rel_amp:.3g} mas/yr posAng(mu_rel)={mu_rel_posang_deg:.2f} deg piE={piE_scalar:.3g}")
            if 'parallax_diag' in results:
                pd = results['parallax_diag']
                print(f"    ParallaxDiag: piL={pd.get('piL_mas')} mas median_rel_err={pd.get('median_rel_err'):.3g}")
        # Provide structured context and do NOT produce a success plot; still emit a diagnostic plot for debugging.
        if verbose:
            print("    VALIDATION FAILURE: criteria unmet")
            for r in fail_reasons:
                print(f"      - {r}")
        # Persist a detailed diagnostics sidecar for post-mortem analysis. This file records
        # per-epoch quantities (times, gulls true centroids, BAGLE sky centroids and shifts)
        # and reparameterization receipts so we can trace mismatches without altering BAGLE outputs.
        geo_receipt = bagle_params['__geo_reparam__'] if use_parallax else None
        diag = {
            'times_mjd': times_bjd.astype(float),
            'gulls_true_N_mas': gulls_data['true_N_centroid_mas'],
            'gulls_true_E_mas': gulls_data['true_E_centroid_mas'],
            'gulls_true_x_ER': gulls_data['true_x_centroid'],
            'gulls_true_y_ER': gulls_data['true_y_centroid'],
            'gulls_source_x_ER': gulls_data['source_x'],
            'gulls_source_y_ER': gulls_data['source_y'],
            'gulls_sky_centroid_N_mas': gulls_data['sky_centroid_N_mas'],
            'gulls_sky_centroid_E_mas': gulls_data['sky_centroid_E_mas'],
            'gulls_lens_rel_x_ER': gulls_data['lens_rel_x'],
            'gulls_lens_rel_y_ER': gulls_data['lens_rel_y'],
            'bagle_shift_E_mas': np.asarray(bagle_data['shift_E'], dtype=float),
            'bagle_shift_N_mas': np.asarray(bagle_data['shift_N'], dtype=float),
            'bagle_sky_centroid_E_mas': np.asarray(bagle_data['sky_centroid_E_mas'], dtype=float),
            'bagle_sky_centroid_N_mas': np.asarray(bagle_data['sky_centroid_N_mas'], dtype=float),
            'bagle_lens_rel_x_ER': np.asarray(bagle_data['lens_rel_x'], dtype=float),
            'bagle_lens_rel_y_ER': np.asarray(bagle_data['lens_rel_y'], dtype=float),
            'thetaE_mas': float(bagle_params['thetaE']),
            't0_original_MJD': bagle_params['t0_original'],
            't0_midpoint_MJD': bagle_params['t0_midpoint'],
            't0_used_MJD': bagle_params['t0'],
            'origin_reparam': bagle_params['__origin_reparam__'],
            'geo_reparam': geo_receipt,
        }
        diag_path = lc_file.parent / f"{lc_file.stem}_bagle_diagnostics.npz"
        np.savez(diag_path, **diag)
        if verbose:
            print(f"    Wrote BAGLE diagnostics sidecar: {diag_path.name}")
        # Produce diagnostic plot before raising for post-mortem review.
        plot_file = lc_file.parent / f"{lc_file.stem}_bagle_FAIL.png"
        plot_validation(times_bjd, gulls_data, bagle_data, bagle_params, plot_file, lc_file.stem + " (FAIL)", bagle_model)
        raise RuntimeError("BAGLE validation failed: " + "; ".join(fail_reasons))

    plot_file = lc_file.parent / f"{lc_file.stem}_bagle.png"
    plot_validation(times_bjd, gulls_data, bagle_data, bagle_params, plot_file, lc_file.stem, bagle_model)
    if verbose:
        print(f"    Magnification RMS: {results['A_rms']:.6e}")
        print(f"    Astrometry RMS: {results['total_astrom_rms']:.4f} mas")
        print(f"    Sky centroid mean offset: {results['sky_mean_offset_total_mas']:.4f} mas")
        print(f"    Plot: {plot_file}")
    return results

__all__ = ['validate_event']
