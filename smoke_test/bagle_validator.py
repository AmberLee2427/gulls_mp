"""BAGLE physics validation for gulls binary lens astrometry."""
from __future__ import annotations
import math
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


def _rotation_ne_to_lens(mu_rel_E: float, mu_rel_N: float, alpha_deg: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the rotation matrix mapping (North, East) offsets to the lens-frame (x, y).

    The construction mirrors VBMicrolensing's BinaryAstroLightCurve definitions (see plotting.py).
    Raises RuntimeError if the supplied vectors do not define a proper rotation.
    """
    if not (np.isfinite(mu_rel_E) and np.isfinite(mu_rel_N)):
        raise RuntimeError("Relative proper motion components must be finite for rotation diagnostic.")
    if mu_rel_E == 0.0 and mu_rel_N == 0.0:
        raise RuntimeError("Relative proper motion vector is zero; cannot define along-track axis.")
    if not np.isfinite(alpha_deg):
        raise RuntimeError("alpha_deg must be finite to construct lens-frame rotation.")

    phi_mu = math.atan2(mu_rel_E, mu_rel_N)  # radians; aligns with plotting.py convention
    ct = math.cos(phi_mu)
    st = math.sin(phi_mu)
    R_NE2TN_U = np.array([[ct, st], [-st, ct]], dtype=float)

    ca = math.cos(math.radians(alpha_deg))
    sa = math.sin(math.radians(alpha_deg))
    A_alpha = np.array([[-ca, sa], [-sa, -ca]], dtype=float)

    R = A_alpha @ R_NE2TN_U
    c_est = float(R[0, 0])
    s_est = float(R[0, 1])
    if not (
        np.allclose(R[1, 0], -s_est, atol=1e-6) and
        np.allclose(R[1, 1], c_est, atol=1e-6)
    ):
        raise RuntimeError("Derived NE→lens rotation is not orthonormal within 1e-6; check inputs.")

    phi_est = math.atan2(s_est, c_est)
    diag = {
        'phi_mu_deg': math.degrees(phi_mu),
        'alpha_deg': float(alpha_deg),
        'phi_est_deg': math.degrees(phi_est),
    }
    return R, diag


def _derive_gulls_sky_centroid_ne(gulls_data: Dict[str, np.ndarray], params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return apparent sky-plane centroid offsets (mas) in the geocentric N/E frame used by GULLS.

    The centroid corresponds to the lensed source image ensemble, already blended to match the light
    curve definition. RA/Dec from the light curve header define the observer frame (geocentric) and
    the origin at the reported lens position.
    """
    for key in ("true_centroid_ra_deg", "true_centroid_dec_deg"):
        if key not in gulls_data:
            raise RuntimeError(f"Lightcurve missing required column '{key}' for sky-plane astrometry reconstruction.")

    ra_series = np.asarray(gulls_data['true_centroid_ra_deg'], dtype=float)
    dec_series = np.asarray(gulls_data['true_centroid_dec_deg'], dtype=float)
    if ra_series.size == 0:
        raise RuntimeError("Cannot derive sky-plane astrometry from empty RA/Dec series.")
    if not (np.all(np.isfinite(ra_series)) and np.all(np.isfinite(dec_series))):
        raise RuntimeError("RA/Dec series contain non-finite values; cannot derive sky-plane NE offsets.")

    ra0 = float(params.get('raL'))
    dec0 = float(params.get('decL'))
    if not (np.isfinite(ra0) and np.isfinite(dec0)):
        raise RuntimeError("Params must provide finite raL/decL for sky-plane astrometry reconstruction.")

    ra_rad = np.deg2rad(ra_series)
    dec_rad = np.deg2rad(dec_series)
    ra0_rad = math.radians(ra0)
    dec0_rad = math.radians(dec0)

    delta_ra_rad = np.unwrap(ra_rad - ra0_rad)
    delta_dec_rad = dec_rad - dec0_rad

    cos_dec0 = math.cos(dec0_rad)
    if abs(cos_dec0) < 1e-9:
        raise RuntimeError("Reference declination too close to ±90°, cannot project RA offsets without blow-up.")

    rad_to_mas = (180.0 / math.pi) * 3600.0 * 1000.0
    east_mas = delta_ra_rad * cos_dec0 * rad_to_mas
    north_mas = delta_dec_rad * rad_to_mas
    return north_mas, east_mas


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
    params['t0'] = t0_mjd  # reassignable later after origin reparam
    params['t0_original'] = t0_mjd  # retain original (COM frame) for plotting/diagnostics

    # Reference time (tref) in .out used by GULLS parallax setup (geocentric transform anchor)
    tref_rel = float(out_params['tref'])
    if not np.isfinite(tref_rel):
        raise ValueError(f"tref must be finite (BJD-relative days); got {out_params['tref']!r}")
    tref_bjd = tref_rel + simulation_zero_time
    tref_mjd = tref_bjd - 2400000.5
    params['t_ref_mjd'] = tref_mjd
    # Expose a conceptual t0_par equal to tref (geocentric reference) for diagnostics
    params['t0_par'] = tref_mjd

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
    # this is wrong!!! see /Users/malpas.1/Code/BAGLE_Microlensing/model_fit_tutorial.txt
    params['xS0_E'] = 0.0  # source position at t0
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
    #   Let φ_v be the sky angle of the relative proper motion v_rel (mas/yr), with components (E,N) matching [sin, cos].
    #   GULLS stores alpha_rel = angle between trajectory direction and the binary axis (same handedness).
    #   Therefore the binary axis sky angle is φ_axis = φ_v − alpha_rel (mod 360°).
    #   BAGLE expects alpha as the binary axis orientation on the sky used in lens offsets (offset ~ [sin α, cos α]).
    v_rel_E = params['muS_E'] - params['muL_E']
    v_rel_N = params['muS_N'] - params['muL_N']
    phi_v = float(np.degrees(np.arctan2(v_rel_E, v_rel_N)))  # degrees, consistent with [sin, cos]
    # BAGLE uses alpha such that lens offset vector = 0.5*sep*[sin(alpha), cos(alpha)].
    # We want axis orientation on sky (alpha_bagle) consistent with the relative PM direction and GULLS alpha_rel:
    # Required: alpha_bagle = (phi_v − alpha_rel) mod 360°.
    alpha_axis_deg = (phi_v - params['alpha_raw_rel']) % 360.0
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

    # Compute actual motion angle phi_v and binary-axis absolute angle phi_axis on sky (radians)
    phi_v = float(np.arctan2(v_rel_E, v_rel_N))
    phi_axis = phi_v - alpha_rel_rad

    # Axis unit vector in sky (E, N)
    axis_hat_E = float(np.sin(phi_axis))
    axis_hat_N = float(np.cos(phi_axis))
    # Motion unit vector in sky (E, N)
    v_hat_E = v_rel_E / v_rel_mag
    v_hat_N = v_rel_N / v_rel_mag

    # Projection of delta (lens1->midpoint) onto motion direction (dimensionless, ER)
    proj_parallel_ER = delta_x_ER * (axis_hat_E * v_hat_E + axis_hat_N * v_hat_N)

    # Time offset in days: Δt = - proj_parallel_ER * tE_days (negative when shifting origin forward along motion)
    delta_t_days = -proj_parallel_ER * tE_days_out

    # Perpendicular component adjusts signed impact parameter beta (mas)
    delta_u_ER = delta_x_ER * np.sin(alpha_rel_rad)
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
        'phi_v_deg': float(np.degrees(phi_v)),
        'phi_axis_deg': float(np.degrees(phi_axis)),
    }

    # Store reparameterized midpoint t0 separately
    params['t0_midpoint'] = params['t0']

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
    # Construct source proper-motion track (mas) relative to the lens RA/Dec origin
    # srce_pos_mas = xS0_mas + (dt_days / days_per_year) * muS_mas_per_yr
        dt_days = np.asarray(times_bjd, dtype=float) - float(params['t0'])
        days_per_year = 365.25
        xS0_E = float(params.get('xS0_E', 0.0))
        xS0_N = float(params.get('xS0_N', 0.0))
        muS_E = float(params['muS_E'])
        muS_N = float(params['muS_N'])
        srce_E_mas = xS0_E + (dt_days / days_per_year) * muS_E
        srce_N_mas = xS0_N + (dt_days / days_per_year) * muS_N
        # BAGLE returns centroid shift as (E, N) corresponding to (RA_offset, Dec_offset)
        # Convert to apparent sky-plane centroid (geocentric, lensed ensemble blended as in BAGLE) by
        # adding source proper motion to the shift. N/E ordering follows plotting.py conventions.
        sky_centroid_E_mas = srce_E_mas + shift[:, 0]
        sky_centroid_N_mas = srce_N_mas + shift[:, 1]

        # Map BAGLE centroid shift into the GULLS lens frame (Einstein radii).
        thetaE_mas = float(params.get('thetaE'))
        if not (np.isfinite(thetaE_mas) and thetaE_mas > 0):
            raise RuntimeError("thetaE must be positive finite to convert centroid shifts into Einstein radii.")
        mu_rel_E = float(params.get('muS_E')) - float(params.get('muL_E'))
        mu_rel_N = float(params.get('muS_N')) - float(params.get('muL_N'))
        alpha_axis_deg = params.get('alpha')
        if alpha_axis_deg is None:
            raise RuntimeError("Params missing lens-frame axis orientation 'alpha'.")
        R_ne_to_lens, rot_diag = _rotation_ne_to_lens(mu_rel_E, mu_rel_N, float(alpha_axis_deg))

        # shift is (E,N); build [N,E] then convert to ER and rotate
        ne_shift_mas = np.column_stack((shift[:, 1], shift[:, 0]))
        ne_shift_er = ne_shift_mas / thetaE_mas
        lens_rel_er = (R_ne_to_lens @ ne_shift_er.T).T  # columns: x, y in Einstein radii relative to source

        out: Dict[str, Any] = {
            'A': A,
            'shift_E': shift[:, 0], 'shift_N': shift[:, 1],
            'sky_centroid_E_mas': sky_centroid_E_mas, 'sky_centroid_N_mas': sky_centroid_N_mas,
            'srce_E': srce_E_mas, 'srce_N': srce_N_mas,
            'lens_rel_x': lens_rel_er[:, 0], 'lens_rel_y': lens_rel_er[:, 1],
            'rotation_NE_to_lens': R_ne_to_lens,
            'rotation_diag': rot_diag,
        }
        if bad_polys:
            out['poly_debug'] = bad_polys
        return out
    finally:
        np.roots = orig_roots

def plot_validation(times_bjd, gulls_data, bagle_data, params, output_file, event_label, bagle_model=None):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    # Current working t0 (after any reparameterization)
    t0_used = params['t0']
    # Optional references for clarity
    t0_mid = params.get('t0_midpoint', t0_used)
    t0_orig = params.get('t0_original', None)
    if 't_ref_mjd' not in params or not np.isfinite(params['t_ref_mjd']):
        raise RuntimeError("plot_validation requires finite t_ref_mjd; validator should have confirmed tref in the .out file.")
    t_ref = float(params['t_ref_mjd'])
    # Use physics-based deblended magnification (requires fs)
    A_gulls = gulls_data['A_deblended']

    def _require_series(container: Dict[str, np.ndarray], key: str, context: str) -> np.ndarray:
        if key not in container:
            raise RuntimeError(f"{context} missing required series '{key}' for plot rendering.")
        arr = np.asarray(container[key], dtype=float)
        if arr.ndim != 1 or arr.size != len(times_bjd):
            raise RuntimeError(
                f"{context} series '{key}' has shape {arr.shape}; expected ({len(times_bjd)},)."
            )
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{context} series '{key}' contains non-finite values.")
        return arr

    gulls_abs_N = _require_series(gulls_data, 'sky_centroid_N_mas_aligned', 'gulls_data')
    gulls_abs_E = _require_series(gulls_data, 'sky_centroid_E_mas_aligned', 'gulls_data')
    bagle_abs_N = _require_series(bagle_data, 'sky_centroid_N_mas_aligned', 'bagle_data')
    bagle_abs_E = _require_series(bagle_data, 'sky_centroid_E_mas_aligned', 'bagle_data')
    lens_rel_x_gulls = _require_series(gulls_data, 'lens_rel_x', 'gulls_data')
    lens_rel_y_gulls = _require_series(gulls_data, 'lens_rel_y', 'gulls_data')
    lens_rel_x_bagle = _require_series(bagle_data, 'lens_rel_x', 'bagle_data')
    lens_rel_y_bagle = _require_series(bagle_data, 'lens_rel_y', 'bagle_data')
    source_x_series = _require_series(gulls_data, 'source_x', 'gulls_data')
    source_y_series = _require_series(gulls_data, 'source_y', 'gulls_data')
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_bjd, A_gulls, 'b.', label='gulls', alpha=0.5, markersize=2)
    ax1.plot(times_bjd, bagle_data['A'], 'r-', label='BAGLE', alpha=0.7)
    # Reference lines: show all available only on the first panel to keep legends tidy
    ax1.axvline(t0_used, color='k', linestyle='--', alpha=0.6, label=f't0_used={t0_used:.2f}')
    ax1.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5, label=f't0_orig={t0_orig:.2f}')
    ax1.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4, label=f't0_mid={t0_mid:.2f}')
    ax1.axvline(t_ref, color='green', linestyle='-.', alpha=0.5, label=f'tref={t_ref:.2f}')
    ax1.set_xlabel('Time (MJD)'); ax1.set_ylabel('Magnification A'); ax1.set_title(f'{event_label}: Magnification')
    ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    A_residual = A_gulls - bagle_data['A']
    ax2.plot(times_bjd, A_residual, 'k.', markersize=2)
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(t0_used, color='k', linestyle='--', alpha=0.6)
    ax2.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5)
    ax2.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4)
    ax2.axvline(t_ref, color='green', linestyle='-.', alpha=0.5)
    ax2.set_xlabel('Time (MJD)'); ax2.set_ylabel('Magnification Residual\n(gulls - BAGLE)'); ax2.set_title('Residuals')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    A_rms = np.sqrt(np.mean(A_residual**2))
    ax3.hist(A_residual, bins=50, edgecolor='black', alpha=0.7); ax3.axvline(0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Magnification Residual'); ax3.set_ylabel('Count'); ax3.set_title(f'RMS={A_rms:.4e}')
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(times_bjd, gulls_abs_N, 'b.', label='gulls sky centroid (aligned)', alpha=0.5, markersize=2)
    ax4.plot(times_bjd, bagle_abs_N, 'r-', label='BAGLE sky centroid (aligned)', alpha=0.7)
    # Optional: show raw shift for reference (faint)
    ax4.plot(times_bjd, bagle_data['shift_N'], color='r', linestyle=':', alpha=0.3, label='BAGLE shift (rel)')
    ax4.axvline(t0_used, color='k', linestyle='--', alpha=0.6)
    ax4.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5)
    ax4.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4)
    ax4.axvline(t_ref, color='green', linestyle='-.', alpha=0.5)
    ax4.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax4.set_xlabel('Time (MJD)'); ax4.set_ylabel('North Shift (mas, aligned to first epoch)'); ax4.set_title('Astrometry: North sky centroid (aligned)')
    ax4.legend(); ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times_bjd, gulls_abs_E, 'b.', label='gulls sky centroid (aligned)', alpha=0.5, markersize=2)
    ax5.plot(times_bjd, bagle_abs_E, 'r-', label='BAGLE sky centroid (aligned)', alpha=0.7)
    ax5.plot(times_bjd, bagle_data['shift_E'], color='r', linestyle=':', alpha=0.3, label='BAGLE shift (rel)')
    ax5.axvline(t0_used, color='k', linestyle='--', alpha=0.6)
    ax5.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5)
    ax5.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4)
    ax5.axvline(t_ref, color='green', linestyle='-.', alpha=0.5)
    ax5.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax5.set_xlabel('Time (MJD)'); ax5.set_ylabel('East Shift (mas, aligned to first epoch)'); ax5.set_title('Astrometry: East sky centroid (aligned)')
    ax5.legend(); ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(gulls_abs_E, gulls_abs_N,
                          c=times_bjd, cmap='viridis', s=20, alpha=0.6, label='gulls sky centroid (aligned)')
    ax6.plot(bagle_abs_E,
             bagle_abs_N, 'r-', alpha=0.5, linewidth=1, label='BAGLE sky centroid (aligned)')
    ax6.plot(bagle_data['shift_E'], bagle_data['shift_N'], color='r', linestyle=':', alpha=0.3, linewidth=1, label='BAGLE shift (rel)')
    if lens_rel_x_gulls.size and lens_rel_x_bagle.size and source_x_series.size == lens_rel_x_gulls.size:
        ax6.plot(lens_rel_x_gulls + source_x_series,
                 lens_rel_y_gulls + source_y_series, color='C0', linestyle='--', alpha=0.4, label='gulls lens-frame (rel)')
        ax6.plot(lens_rel_x_bagle + source_x_series,
                 lens_rel_y_bagle + source_y_series, color='magenta', linestyle='--', alpha=0.4, label='BAGLE lens-frame (rel)')
    ax6.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='Unlensed')
    ax6.set_xlabel('East (mas)'); ax6.set_ylabel('North (mas)'); ax6.set_title('Sky (aligned centroids, N=up, E=right)')
    ax6.legend(); ax6.grid(alpha=0.3); ax6.axis('equal')
    cbar = plt.colorbar(scatter, ax=ax6); cbar.set_label('MJD')
    
    ax7 = fig.add_subplot(gs[2, 0])
    N_residual = gulls_abs_N - bagle_abs_N
    ax7.plot(times_bjd, N_residual, 'k.', markersize=2)
    ax7.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax7.axvline(t0_used, color='k', linestyle='--', alpha=0.6)
    if t0_orig is not None:
        ax7.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5)
    if (t0_mid is not None) and (abs(t0_mid - t0_used) > 1e-9):
        ax7.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4)
    if t_ref is not None and np.isfinite(t_ref):
        ax7.axvline(t_ref, color='green', linestyle='-.', alpha=0.5)
    ax7.set_xlabel('Time (MJD)'); ax7.set_ylabel('North Residual (mas)'); ax7.set_title('North Residuals')
    ax7.grid(alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    E_residual = gulls_abs_E - bagle_abs_E
    ax8.plot(times_bjd, E_residual, 'k.', markersize=2)
    ax8.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax8.axvline(t0_used, color='k', linestyle='--', alpha=0.6)
    if t0_orig is not None:
        ax8.axvline(t0_orig, color='purple', linestyle='--', alpha=0.5)
    if (t0_mid is not None) and (abs(t0_mid - t0_used) > 1e-9):
        ax8.axvline(t0_mid, color='gray', linestyle='--', alpha=0.4)
    if t_ref is not None and np.isfinite(t_ref):
        ax8.axvline(t_ref, color='green', linestyle='-.', alpha=0.5)
    ax8.set_xlabel('Time (MJD)'); ax8.set_ylabel('East Residual (mas)'); ax8.set_title('East Residuals')
    ax8.grid(alpha=0.3)
    
    # On-sky astrometry panel (degrees) — apparent RA/Dec using small-angle conversion about event ra_deg/dec_deg
    ax9 = fig.add_subplot(gs[2, 2])
    # Anchor apparent RA/Dec at the event coordinates ra_deg/dec_deg from the .out file
    ra0 = float(params['raL']); dec0 = float(params['decL'])
    cosd = np.cos(np.deg2rad(dec0)) if np.isfinite(dec0) else 1.0
    # BAGLE apparent sky centroid (mas) -> arcsec (about ra0, dec0)
    cenE_arcsec = np.asarray(bagle_data.get('sky_centroid_E_mas', bagle_data['shift_E']), dtype=float) / 1e3
    cenN_arcsec = np.asarray(bagle_data.get('sky_centroid_N_mas', bagle_data['shift_N']), dtype=float) / 1e3
    # Source and lens reference tracks
    # Source track from PM (arcsec). xS0_E/N are in arcsec per BAGLE model conventions.
    dt_days = np.asarray(times_bjd, dtype=float) - float(params['t0'])
    days_per_year = 365.25
    xS0_E_arcsec = float(params.get('xS0_E', 0.0))
    xS0_N_arcsec = float(params.get('xS0_N', 0.0))
    muS_E_asyr = float(params['muS_E']) * 1e-3
    muS_N_asyr = float(params['muS_N']) * 1e-3
    srcE_arcsec = xS0_E_arcsec + (dt_days / days_per_year) * muS_E_asyr
    srcN_arcsec = xS0_N_arcsec + (dt_days / days_per_year) * muS_N_asyr
    # Lens origin astrometry from BAGLE (arcsec East/North), if available
    lensE_arcsec = lensN_arcsec = None
    try:
        if bagle_model is not None and hasattr(bagle_model, 'get_lens_origin_astrometry'):
            lens_ast = bagle_model.get_lens_origin_astrometry(times_bjd, filt_idx=0)
            lensE_arcsec = np.asarray(lens_ast[:, 1], dtype=float)  # ordering noted earlier as (N, E) vs (E, N); confirm
            lensN_arcsec = np.asarray(lens_ast[:, 0], dtype=float)
    except Exception:
        lensE_arcsec = lensN_arcsec = None

    # Add annual parallax for source to both centroid and source PM tracks (lens origin track includes parallax already)
    try:
        # Determine observer string from model
        model_obs_loc = None
        if bagle_model is not None and hasattr(bagle_model, 'obsLocation'):
            ol = bagle_model.obsLocation
            if isinstance(ol, str):
                model_obs_loc = ol
            elif isinstance(ol, (list, tuple)) and ol and isinstance(ol[0], str):
                model_obs_loc = ol[0]
        # Compute parallax vectors (East, North) via BAGLE helper
        if model_obs_loc is None:
            pvec = _bagle_parallax.parallax_in_direction(ra0, dec0, times_bjd)
        else:
            pvec = _bagle_parallax.parallax_in_direction(ra0, dec0, times_bjd, obsLocation=model_obs_loc)
        # Source parallax amplitude in mas from distance (pc): piS_mas ≈ 1000/dS_pc
        if 'dS' not in params:
            raise RuntimeError("BAGLE validation: missing source distance dS (pc) in params")
        dS_pc = float(params['dS'])
        if not (np.isfinite(dS_pc) and dS_pc > 0):
            raise RuntimeError(f"BAGLE validation: invalid source distance dS={dS_pc} (pc)")
        piS_mas = 1000.0 / dS_pc
        src_par_E_arcsec = (piS_mas * pvec[:, 0]) / 1e3
        src_par_N_arcsec = (piS_mas * pvec[:, 1]) / 1e3
        # Apply source parallax to centroid and source tracks
        cenE_arcsec = cenE_arcsec + src_par_E_arcsec
        cenN_arcsec = cenN_arcsec + src_par_N_arcsec
    except Exception:
        # If any issue, proceed without explicit parallax addition (centroid geometry still reflects parallax internally)
        pass

    # Plot BAGLE centroid apparent RA/Dec
    cenRA_deg = ra0 + (cenE_arcsec / cosd) / 3600.0
    cenDec_deg = dec0 + (cenN_arcsec / 3600.0)
    ax9.plot(cenRA_deg, cenDec_deg, 'r-', label='BAGLE centroid RA/Dec', alpha=0.8)
    # Overlay GULLS true centroid RA/Dec if present
    try:
        gulls_ra = np.asarray(gulls_data['true_centroid_ra_deg'], dtype=float)
        gulls_dec = np.asarray(gulls_data['true_centroid_dec_deg'], dtype=float)
        ax9.plot(gulls_ra, gulls_dec, 'b.', alpha=0.3, markersize=2, label='gulls centroid RA/Dec')
    except Exception:
        pass
    # Plot source PM track as apparent RA/Dec
    # Include source parallax in source track if computed
    try:
        srcE_arcsec = srcE_arcsec + src_par_E_arcsec
        srcN_arcsec = srcN_arcsec + src_par_N_arcsec
    except Exception:
        pass
    srcRA_deg = ra0 + (srcE_arcsec / cosd) / 3600.0
    srcDec_deg = dec0 + (srcN_arcsec / 3600.0)
    ax9.plot(srcRA_deg, srcDec_deg, color='gray', linestyle='--', alpha=0.7, label='source (PM+parallax)')
    # Plot lens origin track as apparent RA/Dec if available
    if lensE_arcsec is not None and lensN_arcsec is not None:
        lensRA_deg = ra0 + (np.asarray(lensE_arcsec)/cosd)/3600.0
        lensDec_deg = dec0 + (np.asarray(lensN_arcsec))/3600.0
        ax9.plot(lensRA_deg, lensDec_deg, color='k', linestyle='--', alpha=0.7, label='lens origin RA/Dec')
    ax9.set_xlabel('RA (deg)'); ax9.set_ylabel('Dec (deg)'); ax9.set_title('On-sky astrometry (apparent RA/Dec, geocentric) — anchored at event ra/dec (J2000)')
    ax9.legend(); ax9.grid(alpha=0.3)
    
    # Compact header: include t0 variants when available
    t0_hdr = f"t0={t0_used:.2f}"
    if t0_orig is not None:
        t0_hdr += f" | t0_orig={t0_orig:.2f}"
    if (t0_mid is not None) and (abs(t0_mid - t0_used) > 1e-9):
        t0_hdr += f" | t0_mid={t0_mid:.2f}"
    if t_ref is not None and np.isfinite(t_ref):
        t0_hdr += f" | tref={t_ref:.2f}"

    plt.suptitle(f'{event_label} | {params["event_type"]} | {t0_hdr} MJD | ' +
                 f'thetaE={params["thetaE"]:.3f} mas | q={params["mLs"]/params["mLp"]:.4f} | ' +
                 f's={params["sep"]/params["thetaE"]:.3f} thetaE | rho={params["rho"]:.4f}\n' +
                 '(Centroid curves are offsets relative to t0_used)', fontsize=12)
    
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
    use_parallax = parallax_enabled
    bagle_model = create_bagle_model(bagle_params, event_type, use_parallax)
    # --- Observer: deterministically resolve from GULLS params and apply to BAGLE ---
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
    # --- Geocentric correction (tref) approximate recentering of t0 ---
    # Goal: In geocentric formulation, tau_total(t) = (t - t0_mid)/tE + tshift(t) with tshift(tref)=0.
    # Closest approach occurs when tau_total ≈ 0. Using first-order approximation, set
    #   t0_geo ≈ t0_mid - tE * tshift(t0_mid).
    # We approximate tshift from BAGLE's parallax vector pvec(E,N) by constructing N/E shifts relative to tref
    # with linear term removed, then projecting onto the mu_rel direction (phi_v).
    try:
        if use_parallax:
            t0_mid = bagle_params.get('t0_midpoint', bagle_params.get('t0'))
            t_ref = bagle_params.get('t_ref_mjd')
            tE_days = bagle_params.get('tE_days')
            piE_amp = bagle_params.get('piE')
            if all(np.isfinite(x) for x in [t0_mid, t_ref, tE_days, piE_amp]):
                # Compute parallax vectors at t0_mid, tref, and derivative near tref
                dt = 0.5  # days for finite difference
                t_arr = np.array([t0_mid, t_ref - dt, t_ref, t_ref + dt], dtype=float)
                pvec = _bagle_parallax.parallax_in_direction(bagle_params['raL'], bagle_params['decL'], t_arr, obsLocation=used_obs)
                # Columns: (East, North). Extract by index for clarity.
                E_t0, N_t0 = float(pvec[0, 0]), float(pvec[0, 1])
                E_m,  N_m  = float(pvec[1, 0]), float(pvec[1, 1])
                E_ref, N_ref = float(pvec[2, 0]), float(pvec[2, 1])
                E_p,  N_p  = float(pvec[3, 0]), float(pvec[3, 1])
                # Finite-difference derivative at tref
                dE_dt = (E_p - E_m) / (2*dt)
                dN_dt = (N_p - N_m) / (2*dt)
                # NE shift relative to reference, with linear term removed
                dt_rel = float(t0_mid - t_ref)
                Eshift = (E_t0 - E_ref) - dt_rel * dE_dt
                Nshift = (N_t0 - N_ref) - dt_rel * dN_dt
                # Project onto trajectory direction using mu_rel angle (approx phi_pi)
                mu_rel_E = bagle_params['muS_E'] - bagle_params['muL_E']
                mu_rel_N = bagle_params['muS_N'] - bagle_params['muL_N']
                phi_v = float(np.arctan2(mu_rel_E, mu_rel_N))  # radians
                cs = np.cos(phi_v); sn = np.sin(phi_v)
                tshift = -piE_amp * (Nshift*cs + Eshift*sn)
                # Recenter t0
                t0_geo = float(t0_mid - tE_days * tshift)
                # Persist diagnostics and apply
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
                # Update working t0 in both params and model
                bagle_params['t0'] = t0_geo
                try:
                    bagle_model.t0 = t0_geo
                except Exception:
                    # If attribute assignment fails (unlikely), we'll rebuild the model below if needed.
                    pass
                if verbose:
                    ore = bagle_params.get('__origin_reparam__', {})
                    dt0_origin = (ore.get('t0_midpoint_MJD') - ore.get('t0_original_MJD')) if ore else np.nan
                    dt0_geo = t0_geo - (t0_mid if np.isfinite(t0_mid) else np.nan)
                    print("    GeoReparam:" +
                          f" tshift(t0)={tshift:.4g}  dE_dt={dE_dt:.3e} dN_dt={dN_dt:.3e} AU/day" +
                          f" | Δt0_origin={dt0_origin:.3g} d from COM→midpoint, Δt0_geo={dt0_geo:.3g} d from midpoint→geo")
    except Exception as _geo_ex:
        if verbose:
            print(f"    GeoReparam skipped: {type(_geo_ex).__name__}: {_geo_ex}")
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
                    print(f"    TimeScaleDiag(parallax): max|Δparallax|≈{max_dmas*1e3:.3f} μas if TT misfed as TDB")
            except Exception as _pdiag_ex:
                print(f"    TimeScaleDiag(parallax) skipped: {_pdiag_ex}")
        except Exception as e_audit:
            # Fail-fast with context; produce audit failure receipt.
            print(f"    Parallax audit failed: {type(e_audit).__name__}: {e_audit}")
            raise
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
        'bagle_sky_centroid_E_mas_aligned': np.asarray(bagle_data.get('sky_centroid_E_mas_aligned', []), dtype=float),
        'bagle_sky_centroid_N_mas_aligned': np.asarray(bagle_data.get('sky_centroid_N_mas_aligned', []), dtype=float),
    # Deprecated compatibility aliases for downstream tooling expecting the old naming.
    'centroid_abs_E_mas': np.asarray(bagle_data.get('sky_centroid_E_mas', []), dtype=float),
    'centroid_abs_N_mas': np.asarray(bagle_data.get('sky_centroid_N_mas', []), dtype=float),
    'centroid_abs_E_aligned': np.asarray(bagle_data.get('sky_centroid_E_mas_aligned', []), dtype=float),
    'centroid_abs_N_aligned': np.asarray(bagle_data.get('sky_centroid_N_mas_aligned', []), dtype=float),
        'bagle_lens_rel_x_er': np.asarray(bagle_data.get('lens_rel_x', []), dtype=float),
        'bagle_lens_rel_y_er': np.asarray(bagle_data.get('lens_rel_y', []), dtype=float),
        'rotation_NE_to_lens': np.asarray(bagle_data.get('rotation_NE_to_lens', np.empty((2, 2)))),
        'thetaE_mas': float(bagle_params.get('thetaE', np.nan)),
    }
    sidecar_path = lc_file.parent / f"{lc_file.stem}_bagle.npz"
    np.savez(sidecar_path, **sidecar)
    # --- Astrometry comparison ---
    # Lens-frame consistency (relative to source position in Einstein radii)
    required_cols = ("true_x_centroid", "true_y_centroid", "source_x", "source_y")
    for key in required_cols:
        if key not in gulls_data:
            raise RuntimeError(f"Lightcurve missing required column '{key}' for lens-frame validation.")
    gulls_true_x = np.asarray(gulls_data['true_x_centroid'], dtype=float)
    gulls_true_y = np.asarray(gulls_data['true_y_centroid'], dtype=float)
    gulls_src_x = np.asarray(gulls_data['source_x'], dtype=float)
    gulls_src_y = np.asarray(gulls_data['source_y'], dtype=float)
    bagle_rel_x = np.asarray(bagle_data.get('lens_rel_x', []), dtype=float)
    bagle_rel_y = np.asarray(bagle_data.get('lens_rel_y', []), dtype=float)
    if not (gulls_true_x.size == gulls_src_x.size == bagle_rel_x.size == len(times_bjd)):
        raise RuntimeError("Lens-frame arrays length mismatch between GULLS and BAGLE predictions.")
    gulls_rel_x = gulls_true_x - gulls_src_x
    gulls_rel_y = gulls_true_y - gulls_src_y
    gulls_data['lens_rel_x'] = gulls_rel_x
    gulls_data['lens_rel_y'] = gulls_rel_y

    lens_rel_diff_x = gulls_rel_x - bagle_rel_x
    lens_rel_diff_y = gulls_rel_y - bagle_rel_y
    lens_rel_rms = float(np.sqrt(np.mean(lens_rel_diff_x**2 + lens_rel_diff_y**2)))

    # Sky-plane NE comparison (mas) reconstructed from RA/Dec
    gulls_sky_centroid_N_mas, gulls_sky_centroid_E_mas = _derive_gulls_sky_centroid_ne(gulls_data, bagle_params)
    bagle_sky_centroid_N_mas = np.asarray(bagle_data.get('sky_centroid_N_mas', []), dtype=float)
    bagle_sky_centroid_E_mas = np.asarray(bagle_data.get('sky_centroid_E_mas', []), dtype=float)
    if not (bagle_sky_centroid_N_mas.size == bagle_sky_centroid_E_mas.size == gulls_sky_centroid_N_mas.size == len(times_bjd)):
        raise RuntimeError("Sky-plane astrometry arrays length mismatch between GULLS and BAGLE predictions.")

    # Align to first epoch to remove constant offsets (lens vs source origin choices)
    bagle_sky_centroid_N_aligned = bagle_sky_centroid_N_mas - bagle_sky_centroid_N_mas[0]
    bagle_sky_centroid_E_aligned = bagle_sky_centroid_E_mas - bagle_sky_centroid_E_mas[0]
    gulls_sky_centroid_N_aligned = gulls_sky_centroid_N_mas - gulls_sky_centroid_N_mas[0]
    gulls_sky_centroid_E_aligned = gulls_sky_centroid_E_mas - gulls_sky_centroid_E_mas[0]

    # BAGLE sky-plane centroid (geocentric, matched to GULLS observer) in mas relative to lens RA/Dec origin.
    bagle_data['sky_centroid_N_mas_aligned'] = bagle_sky_centroid_N_aligned
    bagle_data['sky_centroid_E_mas_aligned'] = bagle_sky_centroid_E_aligned
    # Geocentric sky-plane centroid (North/East, mas) of the lensed image ensemble, blended identically to
    # the light curve and referenced to the lens RA/Dec origin.
    gulls_data['sky_centroid_N_mas_aligned'] = gulls_sky_centroid_N_aligned
    gulls_data['sky_centroid_E_mas_aligned'] = gulls_sky_centroid_E_aligned
    gulls_data['sky_centroid_N_mas'] = gulls_sky_centroid_N_mas
    gulls_data['sky_centroid_E_mas'] = gulls_sky_centroid_E_mas

    N_diff = gulls_sky_centroid_N_aligned - bagle_sky_centroid_N_aligned
    E_diff = gulls_sky_centroid_E_aligned - bagle_sky_centroid_E_aligned
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
        # Persist a detailed diagnostics sidecar for post-mortem analysis. This file records
        # per-epoch quantities (times, gulls true centroids, BAGLE sky centroids and shifts)
        # and reparameterization receipts so we can trace mismatches without altering BAGLE outputs.
        try:
            diag = {
                'times_mjd': times_bjd.astype(float),
                'gulls_true_N_mas': gulls_data.get('true_N_centroid_mas', np.full_like(times_bjd, np.nan)),
                'gulls_true_E_mas': gulls_data.get('true_E_centroid_mas', np.full_like(times_bjd, np.nan)),
                'gulls_true_x_ER': gulls_data.get('true_x_centroid', np.full_like(times_bjd, np.nan)),
                'gulls_true_y_ER': gulls_data.get('true_y_centroid', np.full_like(times_bjd, np.nan)),
                'gulls_src_x_ER': gulls_data.get('source_x', np.full_like(times_bjd, np.nan)),
                'gulls_src_y_ER': gulls_data.get('source_y', np.full_like(times_bjd, np.nan)),
                'gulls_sky_centroid_N_mas': gulls_data.get('sky_centroid_N_mas', np.full_like(times_bjd, np.nan)),
                'gulls_sky_centroid_E_mas': gulls_data.get('sky_centroid_E_mas', np.full_like(times_bjd, np.nan)),
                'gulls_sky_centroid_N_mas_aligned': gulls_data.get('sky_centroid_N_mas_aligned', np.full_like(times_bjd, np.nan)),
                'gulls_sky_centroid_E_mas_aligned': gulls_data.get('sky_centroid_E_mas_aligned', np.full_like(times_bjd, np.nan)),
                'gulls_lens_rel_x_ER': gulls_data.get('lens_rel_x', np.full_like(times_bjd, np.nan)),
                'gulls_lens_rel_y_ER': gulls_data.get('lens_rel_y', np.full_like(times_bjd, np.nan)),
                'bagle_shift_E_mas': np.asarray(bagle_data.get('shift_E', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_shift_N_mas': np.asarray(bagle_data.get('shift_N', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_sky_centroid_E_mas': np.asarray(bagle_data.get('sky_centroid_E_mas', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_sky_centroid_N_mas': np.asarray(bagle_data.get('sky_centroid_N_mas', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_sky_centroid_E_mas_aligned': np.asarray(bagle_data.get('sky_centroid_E_mas_aligned', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_sky_centroid_N_mas_aligned': np.asarray(bagle_data.get('sky_centroid_N_mas_aligned', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_lens_rel_x_ER': np.asarray(bagle_data.get('lens_rel_x', np.full_like(times_bjd, np.nan)), dtype=float),
                'bagle_lens_rel_y_ER': np.asarray(bagle_data.get('lens_rel_y', np.full_like(times_bjd, np.nan)), dtype=float),
                'thetaE_mas': float(bagle_params.get('thetaE', np.nan)),
                't0_original_MJD': bagle_params.get('t0_original'),
                't0_midpoint_MJD': bagle_params.get('t0_midpoint'),
                't0_used_MJD': bagle_params.get('t0'),
                'origin_reparam': bagle_params.get('__origin_reparam__', {}),
                'geo_reparam': bagle_params.get('__geo_reparam__', {}),
            }
            diag_path = lc_file.parent / f"{lc_file.stem}_bagle_diagnostics.npz"
            np.savez(diag_path, **diag)
            if verbose:
                print(f"    Wrote BAGLE diagnostics sidecar: {diag_path.name}")
        except Exception as _diag_ex:
            if verbose:
                print(f"    Warning: failed to write BAGLE diagnostics sidecar: {_diag_ex}")
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
