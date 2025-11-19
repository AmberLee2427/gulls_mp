from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_rotations_module():
    here = Path(__file__).resolve()
    rotations_path = here.parents[1] / "rotations.py"
    spec = importlib.util.spec_from_file_location("rotations_under_test", rotations_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load smoke_test/rotations.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


rotations = _load_rotations_module()
rotation_xy_to_tu = rotations.rotation_xy_to_tu
rotation_tu_to_xy = rotations.rotation_tu_to_xy
rotation_xy_to_ne = rotations.rotation_xy_to_ne
rotation_ne_to_xy = rotations.rotation_ne_to_xy


def _assert_orthonormal(R: np.ndarray, atol: float = 1e-10) -> None:
    assert np.allclose(R @ R.T, np.eye(2), atol=atol)


@pytest.mark.parametrize("alpha_deg, sgn", [(0.0, 1.0), (37.5, 1.0), (123.4, -1.0)])
def test_rotation_xy_tu_inverses_consistent(alpha_deg: float, sgn: float) -> None:
    r_xy_tu = rotation_xy_to_tu(alpha_deg, sgn)
    r_tu_xy = rotation_tu_to_xy(alpha_deg, sgn)
    _assert_orthonormal(r_xy_tu)
    _assert_orthonormal(r_tu_xy)
    assert np.allclose(r_tu_xy, np.linalg.inv(r_xy_tu), atol=1e-12)
    vec_xy = np.array([0.3, -1.7])
    vec_back = r_tu_xy @ (r_xy_tu @ vec_xy)
    assert np.allclose(vec_back, vec_xy, atol=1e-12)


def test_rotation_ne_xy_roundtrip_matches_xy_ne() -> None:
    mu_rel_E = 4.0
    mu_rel_N = -2.0
    alpha_deg = 41.25
    sgn = -1
    r_xy_ne, diag_xy = rotation_xy_to_ne(mu_rel_E, mu_rel_N, alpha_deg, sgn)
    r_ne_xy, diag_ne = rotation_ne_to_xy(mu_rel_E, mu_rel_N, alpha_deg, sgn)
    _assert_orthonormal(r_xy_ne)
    _assert_orthonormal(r_ne_xy)
    assert np.allclose(r_ne_xy @ r_xy_ne, np.eye(2), atol=1e-12)
    assert diag_xy == diag_ne
    vec_xy = np.array([1.2, -0.4])
    vec_back = r_ne_xy @ (r_xy_ne @ vec_xy)
    assert np.allclose(vec_back, vec_xy, atol=1e-12)


def test_rotation_xy_ne_fails_for_zero_mu() -> None:
    with pytest.raises(RuntimeError):
        rotation_xy_to_ne(0.0, 0.0, 10.0, 1.0)
