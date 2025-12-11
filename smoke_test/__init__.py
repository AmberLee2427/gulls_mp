"""Smoke test package exports.

This package exposes the smoke test CLI entrypoint and shared utilities
like read_gulls_lightcurve for reuse across validators.
"""

from .runner import main
from .gulls_io import read_gulls_lightcurve

__all__ = ["main", "read_gulls_lightcurve"]
