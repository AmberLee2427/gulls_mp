"""BAGLE physics validation for gulls binary lens astrometry."""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Run-or-die: require BAGLE to be importable via the Python environment.
from bagle import model
from bagle import parallax as _bagle_parallax  # For direct parallax vector computation receipts.
from .constants import REPO_ROOT

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
    # (debug removed)
        astrometry_no_par = bagle_model.get_lens_origin_astrometry(times_mjd, filt_idx=0)
    finally:
        bagle_model.parallaxFlag = True
    # (debug removed)
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


def _read_gulls_observatory_settings(params: Dict[str, str]) -> Dict[str, Any]:
    """Deterministically resolve GULLS observatory settings from params.

    Uses OBSERVATORY_DIR and OBSERVATORY_LIST to locate the observatory file(s),
    reads the first listed file, and extracts SPACE/ORBIT (and NAME when present).

    Returns a dict with keys: { 'file': Path, 'SPACE': int|None, 'ORBIT': int|None, 'NAME': str|None }.
    Raises on any missing paths or unreadable files.
    """
    obs_dir = params.get('OBSERVATORY_DIR')
    obs_list = params.get('OBSERVATORY_LIST')
    if not obs_dir or not obs_list:
        raise RuntimeError("Parameter file must define OBSERVATORY_DIR and OBSERVATORY_LIST to resolve observer.")

    list_path = (REPO_ROOT / obs_dir / obs_list).resolve()
    if not list_path.is_file():
        raise RuntimeError(f"Observatory list not found: {list_path}")

    entries: List[str] = [
        ln.strip() for ln in list_path.read_text(encoding='utf-8').splitlines()
        if ln.strip() and not ln.strip().startswith('#')
    ]
    if not entries:
        raise RuntimeError(f"Observatory list file is empty: {list_path}")

    first = entries[0]
    obs_file = (REPO_ROOT / obs_dir / first).resolve()
    if not obs_file.is_file():
        raise RuntimeError(f"Observatory file not found from list: {obs_file}")

    # Parse key/value style where keys and values may be whitespace or tab separated
    kv: Dict[str, str] = {}
    for raw in obs_file.read_text(encoding='utf-8').splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.replace('\t', ' ').split()
        if len(parts) >= 2:
            key = parts[0].upper()
            val = parts[1]
            kv[key] = val

    out: Dict[str, Any] = {
        'file': obs_file,
        'SPACE': int(kv['SPACE']) if 'SPACE' in kv and kv['SPACE'].isdigit() else (1 if kv.get('SPACE') in ('true', 'True') else None),
        'ORBIT': int(kv['ORBIT']) if 'ORBIT' in kv and kv['ORBIT'].isdigit() else None,
        'NAME': kv.get('NAME')
    }
    return out

"""Strict mode: offline L2 approximation has been removed. Horizons must resolve '-211'."""

def _resolve_and_apply_bagle_observer(bagle_model, params: Dict[str, str], ra_deg: float, dec_deg: float, times_mjd: np.ndarray, verbose: bool) -> Tuple[str, Dict[str, Any]]:
    """Resolve GULLS observer and set bagle_model.obsLocation accordingly.

    Returns (used_obs_location, receipt_dict). Raises on unresolvable/unsupported settings.
    """
    settings = _read_gulls_observatory_settings(params)
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
    try:
        bagle_model.obsLocation = [used]
    except Exception as ex:
        raise RuntimeError(f"Failed to apply obsLocation '{used}' to BAGLE model: {type(ex).__name__}: {ex}")
    # Minimal observer debug: echo only the resolved/used observer string
    # (debug removed)

    receipt['bagle_obs_applied'] = used
    # Quick sanity probe via audit function later; return
    return used, receipt

def parse_out_file(out_file: Path) -> Dict[str, Any]:
    with open(out_file, 'r') as f:
        columns = f.readline().strip().split()
        values = f.readline().strip().split()
    data = {col: val for col, val in zip(columns, values)}
    return data

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
        'ra_deg', 'dec_deg',
        # Require explicit source/lens galactic + equatorial for PM transform
        'Lens_l', 'Lens_b', 'Lens_RA2000.0', 'Lens_DEC2000.0',
        'Source_l', 'Source_b', 'Source_RA2000.0', 'Source_DEC2000.0'
    ]
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
    params['t0'] = t0_bjd - 2400000.5  # MJD

    thetaE = float(out_params['thetaE'])
    if not np.isfinite(thetaE) or thetaE <= 0:
        raise ValueError(f"thetaE must be positive finite (mas); got {out_params['thetaE']!r}")
    params['thetaE'] = thetaE
    # u0 sign convention reconciliation:
    # GULLS .out provides u0lens1 which can be negative (e.g., -0.787... in sample). BAGLE allows beta (closest approach)
    # to be signed; sign encodes orientation (beta > 0 when source East of lens per BAGLE PSBL_PhotAstromParam1 docstring).
    # We map beta_signed = u0lens1 * thetaE directly, preserving sign, and record u0_amp = abs(u0lens1).
    raw_u0 = float(out_params['u0lens1'])
    if not np.isfinite(raw_u0):
        raise ValueError(f"u0lens1 must be finite (Einstein radii); got {out_params['u0lens1']!r}")
    # UNIT NOTE: beta is an angular closest approach. We construct it in mas via beta = u0 * thetaE (mas).
    # In BAGLE's PSBL_PhotAstrom pipeline, Einstein-radii vs. arcsec unit handling is mixed:
    # - source trajectory uses Einstein radii (u),
    # - lens astrometry uses arcsec and converts mas->arcsec with 1e-3 (see model.py ~6160: xL += (piL * parallax)*1e-3).
    # If BAGLE later normalizes by thetaE internally to compute u, a mas vs arcsec mismatch would scale timescales by ~1e3.
    params['beta'] = raw_u0 * thetaE  # signed beta (mas)
    params['u0_amp'] = abs(raw_u0)    # magnitude in Einstein radii
    params['u0_signed'] = raw_u0      # retain original sign for receipts
    s_dimless = float(out_params['Planet_s'])
    if not np.isfinite(s_dimless) or s_dimless <= 0:
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
    params['sep'] = s_dimless * thetaE
    # Alpha handling:
    # - GULLS generation sets Event->alpha in degrees in buildEvent.cpp:
    #   Event->alpha = 360.0 * ran2(idum);  (uniform in [0,360) deg)
    # - BAGLE PSBL PhotAstrom Param1 expects alpha in DEGREES as documented in
    #   BAGLE_Microlensing/src/bagle/fake_data.py (see docstring near the call to
    #   model.PSBL_PhotAstrom_Par_Param1 where "alpha : float (degrees)" is specified).
    # Therefore: keep alpha in degrees when passing into BAGLE.
    alpha_raw = float(out_params['alpha'])  # degrees from GULLS (angle between trajectory and binary axis)
    if not np.isfinite(alpha_raw) or not (0.0 <= alpha_raw < 360.0 + 1e-9):
        raise ValueError(
            "alpha must be finite degrees in [0, 360). "
            "Receipt: GULLS generates alpha in degrees (uniform 0..360) and BAGLE expects degrees: "
            "BAGLE_Microlensing/src/bagle/fake_data.py:597 (fake_data_PSBL docstring lists 'alpha : float (degrees)').")
    # Defer mapping to BAGLE alpha until after we compute the sky trajectory angle from proper motions.
    params['alpha_raw_rel'] = alpha_raw    # store relative angle (traj vs axis) for reparameterization
    rho = float(out_params['rho'])
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"rho must be positive finite (dimensionless); got {out_params['rho']!r}")
    params['rho'] = rho

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

    # Astrometric reference – keep explicit zeros for origin definition
    params['xS0_E'] = 0.0
    params['xS0_N'] = 0.0

    # Proper motions: gulls outputs are Galactic (l,b) mas/yr; BAGLE wants RA/Dec (E,N) mas/yr.
    # Transform using astropy.
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # Build Galactic coords with proper motions; gulls columns are in degrees.
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
    # NOTE: Relative proper motion sets trajectory speed. If BAGLE normalizes w.r.t. thetaE in arcsec while thetaE is given in mas,
    # tE could be off by ~1e3, broadening the light curve and shifting the peak. Track this if peak lag persists.
    if not (np.isfinite(params['muS_E']) and np.isfinite(params['muS_N'])):
        raise ValueError("Source proper motions transformed to ICRS are non-finite (mas/yr). Check input mul/mub.")

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

    # Map GULLS alpha (relative angle) to BAGLE alpha (axis orientation on sky):
    #   Let φ_v be the sky angle of relative proper motion v_rel (mas/yr), with components (E,N) matching [sin, cos] convention.
    #   If α_gulls is the angle between trajectory direction and the binary axis, then the binary axis sky angle is:
    #       φ_axis = φ_v - α_gulls.
    #   BAGLE expects alpha as the binary axis orientation on the sky used in lens offsets (offset ~ [sin α, cos α]).
    v_rel_E = params['muS_E'] - params['muL_E']
    v_rel_N = params['muS_N'] - params['muL_N']
    phi_v = float(np.degrees(np.arctan2(v_rel_E, v_rel_N)))  # degrees, consistent with [sin, cos]
    # Convention check: GULLS alpha is documented as the angle between trajectory and binary axis.
    # If that angle is measured from trajectory to axis (i.e., rotate trajectory by +alpha to align with axis),
    # then the sky orientation of the axis is φ_axis = φ_v + α_gulls. Try '+' convention here.
    alpha_axis_deg = (phi_v + params['alpha_raw_rel']) % 360.0
    params['alpha'] = alpha_axis_deg

    # Parallax – required (scalar amplitude only for Param1). Directional components (piEN, piEE) are intentionally ignored:
    # Receipt: PSBL_PhotAstrom_Par_Param1 signature includes only |piE|. Relative proper motion provides directional info.
    params['piE'] = float(out_params['piE'])
    if not np.isfinite(params['piE']):
        raise ValueError(f"piE must be finite; got {out_params['piE']!r}")

    # Photometric scaling – required for BAGLE model construction
    params['b_sff'] = [1.0]
    mag_src_val = float(out_params['Source_W146'])
    if not np.isfinite(mag_src_val):
        raise ValueError(f"Source_W146 must be finite magnitude; got {out_params['Source_W146']!r}")
    params['mag_src'] = [mag_src_val]
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

    # Not used by BAGLE construction below; omit to avoid implying certainty
    # params['tE'] = float(out_params.get('tE_helio', 100.0))

    params['event_type'] = event_type

    # --- Origin-dependent reparameterization (t0, beta) REVISED ---
    # COM lens positions (ER): x1_COM = -q/(1+q)*a, x2_COM = +1/(1+q)*a, a = s_dimless.
    # Midpoint in COM coords: x_midpoint_COM = ((1 - q)/(2(1+q))) * a.
    # Translation vector from COM to midpoint frame applied to all positions: Δr = -x_midpoint_COM * thetaE * axis_hat (mas)
    # axis_hat (E,N) = (sin alpha, cos alpha) per BAGLE lens offset convention.
    # Velocity basis: v_rel = muS - muL (mas/yr); v_hat = v_rel / |v_rel|; n_hat = (-v_hat_N, v_hat_E).
    # Project translation:
    #   Δtau_mas = Δr · v_hat,  Δu_mas = Δr · n_hat.
    # Convert to ER: divide by thetaE. Einstein timescale (days, straight-line) tE = (thetaE / |v_rel|) * 365.25.
    # New parameters:
    #   t0_midpoint = t0_COM - (Δtau_ER * tE_days)
    #   beta_midpoint = beta_COM + Δu_mas
    # This decreases t0 if origin shifts forward along motion direction.
    q_dim = float(out_params['Planet_q'])
    if np.isfinite(q_dim) and q_dim > 0:
        a_ER = s_dimless
        x_midpoint_COM_ER = ((1.0 - q_dim) / (2.0 * (1.0 + q_dim))) * a_ER
        alpha_rel_rad = np.deg2rad(params['alpha_raw_rel'])
        v_rel_E = params['muS_E'] - params['muL_E']
        v_rel_N = params['muS_N'] - params['muL_N']
        v_rel_mag = np.hypot(v_rel_E, v_rel_N)
        if v_rel_mag > 0 and np.isfinite(v_rel_mag):
            # Translation along binary axis applied to lens positions when switching COM->midpoint frame.
            # We shift the coordinate origin by Δx_ER = -x_midpoint_COM_ER (Einstein radii) along the axis.
            # Project this into (tau,u) coordinates tied to trajectory using relative angle alpha_rel.
            delta_x_ER = -x_midpoint_COM_ER
            delta_tau_ER = delta_x_ER * np.cos(alpha_rel_rad)  # ER
            u_mid_ER = params['u0_signed'] + delta_x_ER * np.sin(alpha_rel_rad)  # ER
            delta_u_mas = (u_mid_ER - params['u0_signed']) * thetaE  # mas
            tE_days = (thetaE / v_rel_mag) * 365.25  # days (straight-line)
            original_t0 = params['t0']
            original_beta = params['beta']
            params['t0'] = original_t0 - delta_tau_ER * tE_days
            params['beta'] = original_beta + delta_u_mas
            params['__origin_reparam__'] = {
                'x_midpoint_COM_ER': x_midpoint_COM_ER,
                'delta_x_ER': delta_x_ER,
                'delta_tau_ER': delta_tau_ER,
                'u_mid_ER': u_mid_ER,
                'delta_u_mas': delta_u_mas,
                'tE_days': tE_days,
                't0_original_MJD': original_t0,
                'beta_original_mas': original_beta,
                't0_midpoint_MJD': params['t0'],
                'beta_midpoint_mas': params['beta'],
                'v_rel_mag_mas_per_yr': v_rel_mag,
                'alpha_rel_deg': params['alpha_raw_rel']
            }

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
    if not (isinstance(params['b_sff'], list) and isinstance(params['mag_src'], list)):
        raise TypeError("b_sff and mag_src must be lists (one entry per filter)")
    if not (len(params['b_sff']) == len(params['mag_src']) == 1):
        raise ValueError(f"Expected exactly 1 photometric filter; got b_sff len={len(params['b_sff'])}, mag_src len={len(params['mag_src'])}")

    return params

def load_lightcurve(lc_file: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Read a GULLS .lc using shared helper used by smoke test plotting.

        Returns:
            - times_mjd: numpy array of MJD (assumed TDB) derived from BJD in the file.
                Receipts:
                - MJD = JD - 2400000.5
                - GULLS BJD column is written by C++ in gulls_mp/src/outputLightcurve.cpp,
                    header defines "BJD" and rows emit Event->pllx[obsidx].epochs[shiftedidx].
                    This is the absolute time stamp used by the generator (assumed BJD_TDB).
                - BAGLE requires MJD inputs and internally uses TDB for parallax
                    (BAGLE_Microlensing/src/bagle/model.py “Other notes”; and
                     BAGLE_Microlensing/src/bagle/parallax.py builds Time(..., scale='tdb')).
                - Assumes GULLS BJD column is BJD_TDB. No UTC/TT conversion is attempted.
      - data: dict of numpy arrays for all numeric columns
      - meta: header metadata parsed (#Planet, #Event, #fs)
    """
    from .lightcurve_io import read_gulls_lightcurve

    df, meta = read_gulls_lightcurve(lc_file)
    if 'BJD' not in df.columns:
        raise ValueError(f"Lightcurve missing BJD column: {lc_file}")
    times_mjd = df['BJD'].to_numpy(dtype=float, copy=False) - 2400000.5
    # Time scale diagnostic (receipt only, no mutation): compute median spacing and check for
    # sub-second irregularities that might hint at UTC leap second handling instead of TDB.
    # If suspicious patterns found in future, escalate rather than silently adjust.
    if len(times_mjd) > 2:
        dt = np.diff(np.sort(times_mjd))
        med_dt = float(np.median(dt))
        # If median cadence < 1e-4 days (~8.64 s) we still accept; we only warn on pathological negative or zero.
        if med_dt <= 0:
            raise ValueError(f"Non-positive median cadence detected (med_dt={med_dt}); investigate time stamps before validation.")
    data = {col: df[col].to_numpy(dtype=float, copy=False) for col in df.columns}
    return times_mjd, data, meta

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
                b_sff=params['b_sff'], mag_src=params['mag_src'], dmag_Lp_Ls=params['dmag_Lp_Ls'],
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
                b_sff=params['b_sff'], mag_src=params['mag_src'], dmag_Lp_Ls=params['dmag_Lp_Ls']
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

        out: Dict[str, Any] = {'A': A, 'shift_N': shift[:, 0], 'shift_E': shift[:, 1]}
        if bad_polys:
            out['poly_debug'] = bad_polys
        return out
    finally:
        np.roots = orig_roots

def plot_validation(times_bjd, gulls_data, bagle_data, params, output_file, event_label):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    t0_bjd = params['t0']
    # Use physics-based deblended magnification (requires fs)
    A_gulls = gulls_data['A_deblended']
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_bjd, A_gulls, 'b.', label='gulls', alpha=0.5, markersize=2)
    ax1.plot(times_bjd, bagle_data['A'], 'r-', label='BAGLE', alpha=0.7)
    ax1.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3, label=f't0={t0_bjd:.1f}')
    ax1.set_xlabel('Time (MJD)'); ax1.set_ylabel('Magnification A'); ax1.set_title(f'{event_label}: Magnification')
    ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    A_residual = A_gulls - bagle_data['A']
    ax2.plot(times_bjd, A_residual, 'k.', markersize=2)
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5); ax2.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (MJD)'); ax2.set_ylabel('Magnification Residual\n(gulls - BAGLE)'); ax2.set_title('Residuals')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    A_rms = np.sqrt(np.mean(A_residual**2))
    ax3.hist(A_residual, bins=50, edgecolor='black', alpha=0.7); ax3.axvline(0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Magnification Residual'); ax3.set_ylabel('Count'); ax3.set_title(f'RMS={A_rms:.4e}')
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(times_bjd, gulls_data['true_N_centroid_mas'], 'b.', label='gulls', alpha=0.5, markersize=2)
    ax4.plot(times_bjd, bagle_data['shift_N'], 'r-', label='BAGLE', alpha=0.7)
    ax4.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3); ax4.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax4.set_xlabel('Time (MJD)'); ax4.set_ylabel('North Shift (mas)'); ax4.set_title('Astrometry: North')
    ax4.legend(); ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times_bjd, gulls_data['true_E_centroid_mas'], 'b.', label='gulls', alpha=0.5, markersize=2)
    ax5.plot(times_bjd, bagle_data['shift_E'], 'r-', label='BAGLE', alpha=0.7)
    ax5.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3); ax5.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax5.set_xlabel('Time (MJD)'); ax5.set_ylabel('East Shift (mas)'); ax5.set_title('Astrometry: East')
    ax5.legend(); ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(gulls_data['true_E_centroid_mas'], gulls_data['true_N_centroid_mas'],
                          c=times_bjd, cmap='viridis', s=20, alpha=0.6, label='gulls')
    ax6.plot(bagle_data['shift_E'], bagle_data['shift_N'], 'r-', alpha=0.5, linewidth=1, label='BAGLE')
    ax6.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='Unlensed')
    ax6.set_xlabel('East (mas)'); ax6.set_ylabel('North (mas)'); ax6.set_title('Sky (N=up, E=right)')
    ax6.legend(); ax6.grid(alpha=0.3); ax6.axis('equal')
    cbar = plt.colorbar(scatter, ax=ax6); cbar.set_label('MJD')
    
    ax7 = fig.add_subplot(gs[2, 0])
    N_residual = gulls_data['true_N_centroid_mas'] - bagle_data['shift_N']
    ax7.plot(times_bjd, N_residual, 'k.', markersize=2)
    ax7.axhline(0, color='r', linestyle='--', alpha=0.5); ax7.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Time (MJD)'); ax7.set_ylabel('North Residual (mas)'); ax7.set_title('North Residuals')
    ax7.grid(alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    E_residual = gulls_data['true_E_centroid_mas'] - bagle_data['shift_E']
    ax8.plot(times_bjd, E_residual, 'k.', markersize=2)
    ax8.axhline(0, color='r', linestyle='--', alpha=0.5); ax8.axvline(t0_bjd, color='k', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Time (MJD)'); ax8.set_ylabel('East Residual (mas)'); ax8.set_title('East Residuals')
    ax8.grid(alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    total_residual = np.sqrt(N_residual**2 + E_residual**2)
    ax9.hist(total_residual, bins=50, edgecolor='black', alpha=0.7)
    rms_total = np.sqrt(np.mean(total_residual**2))
    ax9.axvline(rms_total, color='r', linestyle='--', linewidth=2, label=f'RMS={rms_total:.3f} mas')
    ax9.set_xlabel('Total Astrometric Residual (mas)'); ax9.set_ylabel('Count'); ax9.set_title('Residual Distribution')
    ax9.legend(); ax9.grid(alpha=0.3)
    
    plt.suptitle(f'{event_label} | {params["event_type"]} | t0={t0_bjd:.2f} MJD | ' +
                 f'thetaE={params["thetaE"]:.3f} mas | q={params["mLs"]/params["mLp"]:.4f} | ' +
                 f's={params["sep"]/params["thetaE"]:.3f} thetaE | rho={params["rho"]:.4f}', fontsize=12)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def validate_event(out_file, lc_file, simulation_zero_time, multiple_sources, params_dict: Optional[Dict[str, str]] = None, verbose=True):
    # Fail-fast top-level: do NOT swallow exceptions here. Upstream runner should treat failures as failures.
    out_params = parse_out_file(out_file)
    event_type = 'BSBL' if multiple_sources == 1 else 'PSBL'
    if verbose:
        print(f"\n  Event: {lc_file.stem}")
        print(f"    Type: {event_type}")
    bagle_params = convert_to_bagle_params(out_params, simulation_zero_time, event_type)
    if verbose:
        dbg = bagle_params.get('__debug_summary__', {})
        print("    ParamSummary:" + 
        f" q={dbg.get('q'):.3g} s={dbg.get('s'):.3g} u0={dbg.get('u0'):.3g} rho={dbg.get('rho'):.3g}" +
        f" thetaE={dbg.get('thetaE_mas'):.3g} beta={dbg.get('beta_mas'):.3g} alpha(deg)={dbg.get('alpha_deg'):.3g}" +
        f" | muL=({dbg.get('muL_E'):.3g},{dbg.get('muL_N'):.3g}) muS=({dbg.get('muS_E'):.3g},{dbg.get('muS_N'):.3g})" +
        f" | dL={dbg.get('dL_pc'):.3g} dS={dbg.get('dS_pc'):.3g} piE={dbg.get('piE_scalar'):.3g}")
    times_bjd, gulls_data, lc_meta = load_lightcurve(lc_file)
    # Timescale diagnostic (explicit receipt): Compare TDB vs TT for a small sample to
    # confirm we are operating within expected sub-millisecond modulation (TT-TDB periodic ~1.6 ms).
    # We do not adjust times; only record amplitude. If amplitude exceeds 5 ms, raise to force review.
    try:
        if times_bjd.size >= 3:
            import astropy.time as _at
            sample = times_bjd[::max(1, times_bjd.size // 5)]  # up to 5 samples
            t_tdb = _at.Time(sample, format='mjd', scale='tdb')
            t_tt  = _at.Time(sample, format='mjd', scale='tt')
            # Difference in seconds
            dt_sec = (t_tt.tdb.jd - t_tt.tt.jd) * 86400.0  # Using tt object for consistent base
            max_dt = float(np.max(np.abs(dt_sec)))
            avg_dt = float(np.mean(dt_sec))
            bagle_params['__timescale_diag__'] = {
                'sample_size': int(sample.size),
                'max_|TT-TDB|_sec': max_dt,
                'mean_(TT-TDB)_sec': avg_dt
            }
            if max_dt > 0.005:  # >5 ms threshold — larger than expected astrophysical correction amplitude
                raise RuntimeError(f"TT-TDB delta {max_dt:.4f}s exceeds 5 ms threshold; verify emitted BJD scale")
            if verbose:
                print(f"    TimeScaleDiag: max|TT-TDB|={max_dt*1e3:.3f} ms mean={avg_dt*1e3:.3f} ms (receipt only)")
    except Exception as _ts_ex:
        # Fail-fast philosophy: escalate if we cannot perform the diagnostic
        raise RuntimeError(f"Timescale diagnostic failed: {_ts_ex}") from _ts_ex
    use_parallax = abs(bagle_params['piE']) > 1e-6
    bagle_model = create_bagle_model(bagle_params, event_type, use_parallax)
    # --- Observer: deterministically resolve from GULLS params and apply to BAGLE ---
    if params_dict is None:
        raise RuntimeError("Validator requires parameter dictionary to resolve observer deterministically; received None.")
    used_obs, obs_receipt = _resolve_and_apply_bagle_observer(
        bagle_model, params_dict, bagle_params['raL'], bagle_params['decL'], times_bjd, verbose
    )
    if verbose:
        print(f"    ObserverReceipt: obsLocation='{used_obs}' source=Horizons file={obs_receipt['gulls_obs_file']}")
    # Receipt: print the exact time range that will be sent to Horizons (if a spacecraft is used)
    try:
        import astropy.time as _at
        min_time_mjd = float(np.min(times_bjd)) if times_bjd.size else float('nan')
        max_time_mjd = float(np.max(times_bjd)) if times_bjd.size else float('nan')
        if used_obs == '-211':
            bd_mjd = _EARLIEST_ROMAN_MJD
            t_min_iso = _at.Time(min_time_mjd, format='mjd', scale='tdb').iso if np.isfinite(min_time_mjd) else 'nan'
            t_max_iso = _at.Time(max_time_mjd, format='mjd', scale='tdb').iso if np.isfinite(max_time_mjd) else 'nan'
            t_bd_iso  = _at.Time(bd_mjd,      format='mjd', scale='tdb').iso
            print(
                "    HorizonsQuery: target='-211' (Roman)\n"
                f"      min_MJD_TDB={min_time_mjd:.6f}  min_JD_TDB={min_time_mjd + 2400000.5:.6f}  min_ISO_TDB={t_min_iso}\n"
                f"      max_MJD_TDB={max_time_mjd:.6f}  max_JD_TDB={max_time_mjd + 2400000.5:.6f}  max_ISO_TDB={t_max_iso}\n"
                f"      earliest_supported_MJD_TDB={bd_mjd:.6f}  earliest_ISO_TDB={t_bd_iso}  (guard +0.05 d applies)"
            )
    except Exception as _print_ex:
        if verbose:
            print(f"    HorizonsQuery: failed to format time receipts: {_print_ex}")
    # Pre-flight Roman ephemeris boundary guard: fail with explicit receipt before network call if times begin too close
    # to the published lower boundary (Horizons error can be cryptic when off by minutes). We require a 0.05-day safety margin.
    if used_obs == '-211':
        min_time_mjd = float(np.min(times_bjd)) if times_bjd.size else np.nan
        if np.isfinite(min_time_mjd) and min_time_mjd < _EARLIEST_ROMAN_MJD + 0.05:
            raise RuntimeError(
                f"Roman ephemeris pre-boundary: earliest MJD={min_time_mjd:.6f} is < boundary+0.05d ({_EARLIEST_ROMAN_MJD + 0.05:.6f}). "
                f"Shift SIMULATION_ZERO_TIME forward (>= {_EARLIEST_ROMAN_JD + 0.05:.5f} JD) or trim early lightcurve epochs."
            )
    # Parallax alignment audit (fail-fast) if parallax active.
    parallax_diag: Optional[Dict[str, Any]] = None
    if use_parallax:
        try:
            parallax_diag = audit_parallax_alignment(
                ra_deg=bagle_params['raL'], dec_deg=bagle_params['decL'],
                times_mjd=times_bjd, bagle_model=bagle_model, obs_location=used_obs
            )
            if verbose:
                print("    ParallaxAudit: median_rel_err={:.3g} max_rel_err={:.3g} component_max_diff_mas={:.3g}".format(
                    parallax_diag['median_rel_err'], parallax_diag['max_rel_err'], parallax_diag['component_abs_max_diff_mas']))
            # Additional timescale sensitivity (optional receipt): quantify effect on parallax if TT were fed as TDB.
            try:
                import astropy.time as _at
                # Sample a few points for speed
                sample_idx = np.linspace(0, times_bjd.size - 1, num=min(8, times_bjd.size), dtype=int)
                t_samp = times_bjd[sample_idx]
                t_tt = _at.Time(t_samp, format='mjd', scale='tt')
                dt_days = (t_tt.tdb.jd - t_tt.tt.jd)  # TDB - TT in days
                # Minimal observer debug (requested): just echo the observer string actually being passed.
                # (debug removed)
                p_t = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], t_samp, obsLocation=used_obs)
                # (debug removed)
                p_t_plus = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], t_samp + dt_days, obsLocation=used_obs)
                dp = (p_t_plus - p_t)
                # Determine piL (mas)
                piL_mas = getattr(bagle_model, 'piL', None)
                if (piL_mas is None) and hasattr(bagle_model, 'dL') and np.isfinite(bagle_model.dL) and bagle_model.dL > 0:
                    piL_mas = 1000.0 / float(bagle_model.dL)
                if piL_mas is not None and np.isfinite(piL_mas):
                    dmas = np.hypot(*(piL_mas * dp).T)  # mas
                    max_dmas = float(np.max(np.abs(dmas))) if dmas.size else 0.0
                    if verbose:
                        print(f"    TimeScaleDiag(parallax): max|Δparallax|≈{max_dmas*1e3:.3f} μas if TT misfed as TDB")
            except Exception as _pdiag_ex:
                if verbose:
                    print(f"    TimeScaleDiag(parallax) skipped: {_pdiag_ex}")
        except Exception as e_audit:
            # Fail-fast with context; produce audit failure receipt.
            if verbose:
                print(f"    Parallax audit failed: {type(e_audit).__name__}: {e_audit}")
            raise
    # Parallax components (piEN, piEE) are ignored for Param1; no informational print needed.
    try:
        bagle_data = compute_bagle_predictions(bagle_model, times_bjd, bagle_params)
    except Exception as e_compute:
        if verbose:
            # Print compact diagnostics around t0
            t0 = bagle_params['t0']
            if len(times_bjd) > 0 and np.isfinite(t0):
                idx = int(np.argmin(np.abs(times_bjd - t0)))
                t_slice = times_bjd[max(0, idx-3): min(len(times_bjd), idx+4)]
                print(f"    BAGLE compute failed near t0. Times around t0 (MJD): {t_slice}")
            print("    Re-raising with context. Params summary above.")
        raise
    # --- Magnification deblending ---
    # Compute deblended magnification from header fs if not already present.
    if 'fs' not in lc_meta or not np.isfinite(lc_meta['fs']):
        raise RuntimeError("Lightcurve header missing finite fs; cannot deblend magnification.")
    fs = float(lc_meta['fs'])
    if fs <= 0 or fs > 1:
        raise RuntimeError(f"Invalid fs in header: {fs}. Expected 0 < fs ≤ 1.")
    F_total = gulls_data['true_relative_flux']
    F_blend = 1.0 - fs
    A_gulls = (F_total - F_blend) / fs
    gulls_data['A_deblended'] = A_gulls
    A_diff = A_gulls - bagle_data['A']
    # --- Astrometry comparison ---
    # User explicitly rejected centroid baseline detrending. Compare raw centroid shifts directly.
    N_diff = gulls_data['true_N_centroid_mas'] - bagle_data['shift_N']
    E_diff = gulls_data['true_E_centroid_mas'] - bagle_data['shift_E']
    if verbose:
        print(f"    Deblended magnification using fs={fs:.6g} from header (#fs). No heuristics applied.")
    results = {
        'event_type': event_type,
        'A_rms': np.sqrt(np.mean(A_diff**2)),
        'N_rms': np.sqrt(np.mean(N_diff**2)),
        'E_rms': np.sqrt(np.mean(E_diff**2)),
        'total_astrom_rms': np.sqrt(np.mean(N_diff**2 + E_diff**2))
    }
    if parallax_diag is not None:
        results['parallax_diag'] = parallax_diag

    # Fail-fast validation criteria focused strictly on physical agreement (no cosmetic alignment):
    # 1. Magnification RMS threshold (0.5) — large divergence implies wrong event construction or unit mismatch.
    # 2. Astrometric total RMS should be < 0.3 * thetaE (scale set by Einstein radius).
    # 3. Magnification correlation ≥ 0.95 (allowing finite-source vs point-source moderate differences but still requiring temporal alignment of peak).
    # 4. Peak time lag |Δt_peak| must be ≤ 2 days (receipt: light curve generated spans hundreds of days; larger offsets suggest time origin mismatch).
    # NOTE: We no longer fail purely because FSBL is unavailable; physical mismatch must be expressed in metrics.
    thetaE_mas = bagle_params['thetaE']
    A_corr = np.nan
    if A_gulls.size == bagle_data['A'].size and A_gulls.size > 1:
        try:
            A_corr = float(np.corrcoef(A_gulls, bagle_data['A'])[0, 1])
        except Exception:
            A_corr = np.nan
    fail_reasons = []
    if results['A_rms'] > 0.5:
        fail_reasons.append(f"Magnification RMS {results['A_rms']:.3g} > 0.5 threshold")
    if thetaE_mas > 0 and results['total_astrom_rms'] > 0.3 * thetaE_mas:
        fail_reasons.append(f"Astrometric RMS {results['total_astrom_rms']:.3g} mas > 0.3*thetaE ({0.3*thetaE_mas:.3g} mas)")
    if not np.isnan(A_corr) and A_corr < 0.95:
        fail_reasons.append(f"Magnification correlation {A_corr:.4f} < 0.95")
    # Peak time lag check
    try:
        t_peak_gulls = float(times_bjd[np.argmax(A_gulls)])
        t_peak_bagle = float(times_bjd[np.argmax(bagle_data['A'])])
        peak_lag = abs(t_peak_gulls - t_peak_bagle)
        if peak_lag > 0.1:
            fail_reasons.append(f"Peak time lag {peak_lag:.3f} days > 0.1 day threshold")
    except Exception as _ex:
        fail_reasons.append("Peak time lag computation failed")

    if fail_reasons:
        # Augment diagnostics specifically for peak lag to expose motion vectors.
        if any('Peak time lag' in r for r in fail_reasons):
            try:
                # Compute relative proper motion amplitude and direction receipts.
                mu_rel_E = bagle_params['muS_E'] - bagle_params['muL_E']
                mu_rel_N = bagle_params['muS_N'] - bagle_params['muL_N']
                mu_rel_amp = np.hypot(mu_rel_E, mu_rel_N)
                mu_rel_posang_deg = (np.degrees(np.arctan2(mu_rel_E, mu_rel_N)) % 360.0)
                # Parallax scalar
                piE_scalar = bagle_params.get('piE')
                # Record context
                print(f"    PeakLagContext: mu_rel_amp={mu_rel_amp:.3g} mas/yr posAng(mu_rel)={mu_rel_posang_deg:.2f} deg piE={piE_scalar:.3g}")
                if 'parallax_diag' in results:
                    pd = results['parallax_diag']
                    print(f"    ParallaxDiag: piL={pd.get('piL_mas')} mas median_rel_err={pd.get('median_rel_err'):.3g}")
            except Exception as _ctx_ex:
                print(f"    PeakLagContext: failed to compute extra diagnostics: {type(_ctx_ex).__name__}: {_ctx_ex}")
        # Provide structured context and do NOT produce a success plot; still emit a diagnostic plot for debugging.
        if verbose:
            print("    VALIDATION FAILURE: criteria unmet")
            for r in fail_reasons:
                print(f"      - {r}")
        # Produce diagnostic plot before raising for post-mortem review.
        plot_file = lc_file.parent / f"{lc_file.stem}_bagle_FAIL.png"
        plot_validation(times_bjd, gulls_data, bagle_data, bagle_params, plot_file, lc_file.stem + " (FAIL)")
        raise RuntimeError("BAGLE validation failed: " + "; ".join(fail_reasons))

    plot_file = lc_file.parent / f"{lc_file.stem}_bagle.png"
    plot_validation(times_bjd, gulls_data, bagle_data, bagle_params, plot_file, lc_file.stem)
    if verbose:
        print(f"    Magnification RMS: {results['A_rms']:.6e}")
        print(f"    Astrometry RMS: {results['total_astrom_rms']:.4f} mas")
        print(f"    Plot: {plot_file}")
    return results

__all__ = ['validate_event']
