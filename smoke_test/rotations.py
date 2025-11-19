"""Rotation utilities shared between smoke-test modules."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def rotation_xy_to_tu(alpha_deg: float, sgn: float) -> np.ndarray:
    """Return the rotation matrix mapping lens-frame (x, y) to trajectory (t, u)."""
    ca = math.cos(math.radians(alpha_deg))
    sa = math.sin(math.radians(alpha_deg))
    return np.array(
        [
            [ca, sa],
            [-sgn * sa, sgn * ca],
        ],
        dtype=float,
    )


def rotation_tu_to_xy(alpha_deg: float, sgn: float) -> np.ndarray:
    """Return the inverse rotation mapping (t, u) offsets to lens-frame (x, y)."""
    return np.linalg.inv(rotation_xy_to_tu(alpha_deg, sgn))


def rotation_xy_to_ne(
    mu_rel_E: float,
    mu_rel_N: float,
    alpha_deg: float,
    sgn: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Map lens-frame (x, y) offsets to observer-frame (North, East)."""
    if not (np.isfinite(mu_rel_E) and np.isfinite(mu_rel_N)):
        raise RuntimeError("Relative proper motion components must be finite for rotation diagnostic.")
    if mu_rel_E == 0.0 and mu_rel_N == 0.0:
        raise RuntimeError("Relative proper motion vector is zero; cannot define along-track axis.")
    if not np.isfinite(alpha_deg):
        raise RuntimeError("alpha_deg must be finite to construct lens-frame rotation.")

    mu_vec = np.array([mu_rel_N, mu_rel_E], dtype=float)
    mu_norm = np.linalg.norm(mu_vec)
    if mu_norm == 0.0:
        raise RuntimeError("Relative proper motion vector is zero; cannot define along-track axis.")
    hat_t_ne = mu_vec / mu_norm
    hat_u_ne = np.array([-sgn * hat_t_ne[1], sgn * hat_t_ne[0]], dtype=float)
    R_TU2NE = np.column_stack((hat_t_ne, hat_u_ne))
    R_XY2TU = rotation_xy_to_tu(alpha_deg, sgn)
    R = R_TU2NE @ R_XY2TU
    if not np.allclose(R @ R.T, np.eye(2), atol=1e-10):
        raise RuntimeError("Derived lens→NE rotation is not orthonormal within tolerance.")
    phi_mu = math.degrees(math.atan2(mu_rel_E, mu_rel_N))
    phi_est = math.degrees(math.atan2(R[0, 1], R[0, 0]))
    diag = {
        'phi_mu_deg': float(phi_mu),
        'alpha_deg': float(alpha_deg),
        'phi_est_deg': float(phi_est),
    }
    return R, diag


def rotation_ne_to_xy(
    mu_rel_E: float,
    mu_rel_N: float,
    alpha_deg: float,
    sgn: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return the inverse rotation mapping observer-frame (N, E) offsets to lens-frame (x, y)."""
    R_xy_ne, diag = rotation_xy_to_ne(mu_rel_E, mu_rel_N, alpha_deg, sgn)
    return np.linalg.inv(R_xy_ne), diag

