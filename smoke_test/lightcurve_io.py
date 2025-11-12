"""Shared GULLS lightcurve I/O utilities (fail-fast, no silent swallowing).

Parses header metadata then loads the whitespace-delimited table with pandas.
Header lines begin with '#'. Recognized:
  #Planet: <vals...>   -> planet parameter vector (floats)
  #Event:  <vals...>   -> event parameter vector (floats)
  #fs: <float>         -> source fractional flux (fs) if present

Returns a (DataFrame, meta) tuple where meta contains any parsed header lists
and scalars. Raises exceptions for malformed numeric tokens or missing required
columns.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

def read_gulls_lightcurve(lc_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    planet_vals: List[float] | None = None
    event_vals: List[float] | None = None
    fs_val: float | None = None
    with lc_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            stripped = line.strip()
            if stripped.startswith('#Planet:'):
                tokens = stripped.split()[1:]
                if not tokens:
                    raise ValueError(f"Malformed #Planet header in {lc_file.name}: {stripped}")
                try:
                    planet_vals = [float(x) for x in tokens]
                except Exception as ex:
                    raise ValueError(f"Non-numeric token in #Planet header of {lc_file.name}: {ex}") from ex
            elif stripped.startswith('#Event:'):
                tokens = stripped.split()[1:]
                if not tokens:
                    raise ValueError(f"Malformed #Event header in {lc_file.name}: {stripped}")
                try:
                    event_vals = [float(x) for x in tokens]
                except Exception as ex:
                    raise ValueError(f"Non-numeric token in #Event header of {lc_file.name}: {ex}") from ex
            elif stripped.startswith('#fs:'):
                parts = stripped.split(':', 1)
                if len(parts) < 2:
                    raise ValueError(f"Malformed #fs header in {lc_file.name}: {stripped}")
                token = parts[1].strip().split()[0]
                try:
                    fs_val = float(token)
                except Exception as ex:
                    raise ValueError(f"Non-numeric fs value in {lc_file.name}: {ex}") from ex
    df = pd.read_csv(lc_file, sep=r"\s+", comment="#")
    if df.empty:
        raise ValueError(f"Lightcurve file is empty: {lc_file}")
    meta: Dict[str, Any] = {}
    if planet_vals is not None:
        meta['planet_vals'] = planet_vals
    if event_vals is not None:
        meta['event_vals'] = event_vals
    if fs_val is not None:
        meta['fs'] = fs_val
    return df, meta

__all__ = ["read_gulls_lightcurve"]
