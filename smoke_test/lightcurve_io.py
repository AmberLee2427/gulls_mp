"""Backward-compatible import shim for lightcurve I/O utilities.

Historically the smoke test exposed `lightcurve_io.read_gulls_lightcurve`.
The module was renamed to `gulls_io`, but several modules (and downstream
scripts/tests) still import from `.lightcurve_io`.  Keep a thin alias to
avoid churn until all callers are updated explicitly.
"""
from __future__ import annotations

from .gulls_io import *  # noqa: F401,F403
