"""Strict, non-plotting utilities for the smoke test.

- No silent fallbacks.
- No defaulting to zero for physical parameters.
- If a required value is missing or non-finite, raise SmokeTestError loudly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

np.seterr(all="raise")

from .errors import SmokeTestError
from .constants import REPO_ROOT

from VBMicrolensing import VBMicrolensing as VBMicrolensingClass  # type: ignore[attr-defined]
VBM_CLASS = VBMicrolensingClass  # type: ignore

from astropy.coordinates import SkyCoord
import astropy.units as u


def derive_event_key(lc_file: Path) -> Tuple[int, int, int] | None:
    stem = lc_file.stem.split(".", 1)[0]
    parts = stem.rsplit("_", 3)
    if len(parts) < 4:
        raise SmokeTestError(f"Cannot derive event key from lightcurve filename: {lc_file.name}")
    return tuple(int(part) for part in parts[-3:])


def galactic_pm_to_icrs(l_deg: float, b_deg: float, mu_l: float, mu_b: float) -> Tuple[float, float]:
    coord = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        pm_l_cosb=mu_l * u.mas / u.yr,
        pm_b=mu_b * u.mas / u.yr,
        frame="galactic",
    )
    icrs = coord.icrs
    return (
        icrs.pm_ra_cosdec.to_value(u.mas / u.yr),
        icrs.pm_dec.to_value(u.mas / u.yr),
    )


def _require_finite(value: float | None, label: str) -> float:
    if value is None or math.isnan(value):
        raise SmokeTestError(f"Required parameter '{label}' is missing or non-finite.")
    return float(value)


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference between angles a and b in degrees (wrap at 360)."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return float(d)


def validate_summary_vs_event(summary: Dict[str, float], event_vals: List[float] | None) -> None:
    """Cross-validate key physics in summary against #Event header values.

    Required behavior (AGENTS.md):
    - Validation-only (no fallback). If sources differ beyond tight tolerances, raise SmokeTestError.
    - For this codebase, #Event is mandatory; raise if missing or incomplete.
    """
    if not event_vals or len(event_vals) < 8:
        raise SmokeTestError("Lightcurve missing or incomplete #Event header values (need >= 8 entries)")

    # Map expected indices from #Event header
    u0_hdr = float(event_vals[0])
    alpha_hdr = float(event_vals[1])  # degrees
    t0_hdr = float(event_vals[2])
    tE_hdr = float(event_vals[6])
    rho_hdr = float(event_vals[7])

    # Pull summary values (will raise later in compute_vbm_model if missing)
    u0_sum = summary.get("u0")
    alpha_sum = summary.get("alpha_event")
    t0_sum = summary.get("t0")
    tE_sum = summary.get("tE_ref")
    rho_sum = summary.get("rho")

    # Tolerances: match to header print precision from outputLightcurve.cpp (limited sig figs)
    # Use per-field absolute tolerances reflecting typical print precision.
    base_rtol = 1e-8
    field_atol = {
        "u0": 1e-5,
        "t0": 1e-5,
        "tE_ref": 1e-5,
        "rho": 1e-5,
    }

    def _check_close(name: str, s_val: float | None, h_val: float) -> None:
        s = _require_finite(s_val, f"summary.{name}")
        if not np.isfinite(h_val):
            raise SmokeTestError(f"#Event header {name} is non-finite")
        atol = field_atol.get(name, 1e-5)
        if not np.isclose(float(s), float(h_val), rtol=base_rtol, atol=atol):
            raise SmokeTestError(
                f"Summary mismatch for {name}: summary={float(s)} header={float(h_val)}"
            )

    # Validate scalars
    _check_close("u0", u0_sum, u0_hdr)
    _check_close("t0", t0_sum, t0_hdr)
    _check_close("tE_ref", tE_sum, tE_hdr)
    _check_close("rho", rho_sum, rho_hdr)

    # Validate angle with wrap handling
    alpha_val = _require_finite(alpha_sum, "summary.alpha_event")
    if not np.isfinite(alpha_hdr):
        raise SmokeTestError("#Event header alpha_event is non-finite")
    d = abs(_angle_diff_deg(float(alpha_val), alpha_hdr))
    if d > 1e-5:  # validate to 5 decimals for angles
        raise SmokeTestError(
            f"Summary mismatch for alpha_event: summary={float(alpha_val)} deg header={alpha_hdr} deg (Δ={d} deg)"
        )


def compute_vbm_model(
    summary: Dict[str, float],
    planet_vals: List[float],
    event_vals: List[float] | None,
    source_pm_icrs: Tuple[float, float],
    lens_pm_icrs: Tuple[float, float] | None,
    theta_e_float: float,
    source_dist_float: float,
    event_ra_float: float,
    event_dec_float: float,
    alpha_deg_float: float,
    sim_zero_offset: float,
    time: np.ndarray,
    true_x_vals: np.ndarray | None,
    true_y_vals: np.ndarray | None,
) -> Dict[str, np.ndarray | str]:
    # Cross-validate summary against event header (strict; #Event required, no fallbacks)
    validate_summary_vs_event(summary, event_vals)

    # Planet parameters from header (required)
    if not (planet_vals and len(planet_vals) >= 6):
        raise SmokeTestError("Header missing #Planet metadata (expected at least 6 values)")
    q_val = float(planet_vals[4])
    s_val = float(planet_vals[5])
    if not (np.isfinite(q_val) and np.isfinite(s_val) and q_val > 0 and s_val > 0):
        raise SmokeTestError(f"Unphysical planet parameters (q={q_val}, s={s_val})")

    # Required from summary only (no fallback to event_vals)
    rho_val = _require_finite(summary.get("rho"), "rho")
    tE_val = _require_finite(summary.get("tE_ref"), "tE_ref")
    u0_val = _require_finite(summary.get("u0"), "u0")
    alpha_event_val = _require_finite(summary.get("alpha_event"), "alpha_event")
    t0_val = _require_finite(summary.get("t0"), "t0")
    pi_n_val = _require_finite(summary.get("pi_n"), "pi_n")
    pi_e_val = _require_finite(summary.get("pi_e"), "pi_e")

    if not (np.isfinite(theta_e_float) and theta_e_float > 0):
        raise SmokeTestError("theta_E must be positive and finite")
    if not (np.isfinite(source_dist_float) and source_dist_float > 0):
        raise SmokeTestError("source distance must be positive and finite (pc)")
    if not (np.isfinite(event_ra_float) and np.isfinite(event_dec_float)):
        raise SmokeTestError("event RA/Dec must be finite (degrees)")
    if not (np.isfinite(alpha_deg_float)):
        raise SmokeTestError("alpha_event must be finite (degrees)")
    if source_pm_icrs is None or not (np.isfinite(source_pm_icrs[0]) and np.isfinite(source_pm_icrs[1])):
        raise SmokeTestError("source proper motion (ICRS) is required and must be finite")

    pi_s_val = 1.0 / float(source_dist_float)
    if not (np.isfinite(pi_s_val) and pi_s_val > 0):
        raise SmokeTestError("derived source parallax must be positive and finite")

    vbm = VBM_CLASS()  # type: ignore[operator]

    skycoord = SkyCoord(
        ra=float(event_ra_float) * u.deg,
        dec=float(event_dec_float) * u.deg,
    )
    coord_str = (
        f"{skycoord.ra.to_string(unit=u.hour, sep=':', pad=True)} "
        f"{skycoord.dec.to_string(unit=u.deg, sep=':', pad=True, alwayssign=True)}"
    )
    vbm.SetObjectCoordinates(coord_str)

    params_vbm = [
        math.log(float(s_val)),
        math.log(float(q_val)),
        float(u0_val),
        math.radians(float(alpha_event_val)),
        math.log(float(rho_val)),
        math.log(float(tE_val)),
        float(t0_val) + float(sim_zero_offset),
        float(pi_n_val),
        float(pi_e_val),
        float(source_pm_icrs[1]),
        float(source_pm_icrs[0]),
        float(pi_s_val),
        float(theta_e_float),
    ]

    results = vbm.BinaryAstroLightCurve(params_vbm, time + float(sim_zero_offset))

    lens_dec_deg = np.array(results[3], dtype=float)
    lens_ra_deg = np.array(results[4], dtype=float)
    y1 = np.array(results[5], dtype=float)
    y2 = np.array(results[6], dtype=float)

    lensframe_label = "Source Trajectory BinaryAstroLightCurve"
    vbm_x = y1
    vbm_y = y2

    # If truth is available, find the sign/axis convention that best matches it (strictly a diagnostic, not a fallback)
    # Deterministic mapping per VBM: lens-frame axes are x=y1s, y=y2s (see VBMicrolensing::BinaryAstroLightCurve)
    # No heuristic flipping or swapping; tests must expose any mismatch.
    vbm_x, vbm_y = y1, y2

    if len(vbm_x) != len(time):
        raise SmokeTestError("VBM returned mismatched array lengths")

    model: Dict[str, np.ndarray | str] = {
        "lens_x": vbm_x,
        "lens_y": vbm_y,
        "lens_label": lensframe_label,
        "sky_ra": lens_ra_deg,
        "sky_dec": lens_dec_deg,
        "sky_label": "VBM BinaryAstroLightCurve (sky)",
    }
    return model
