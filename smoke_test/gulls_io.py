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


def parse_out_file(out_file: Path) -> Dict[str, Any]:
    with open(out_file, 'r') as f:
        columns = f.readline().strip().split()
        values = f.readline().strip().split()
    data = {col: val for col, val in zip(columns, values)}
    return data


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
    # Enforce presence of #Event header for current codebase outputs (outputLightcurve.cpp always writes it)
    if event_vals is None:
        raise ValueError(f"Missing required #Event header in {lc_file.name}")
    meta['event_vals'] = event_vals
    if fs_val is not None:
        meta['fs'] = fs_val
    return df, meta

def read_gulls_observatory_settings(params: Dict[str, str]) -> Dict[str, Any]:
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
    from .gulls_io import read_gulls_lightcurve

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


def calculate_magnification_from_lightcurve(lc_meta: Dict[str, Any], gulls_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute deblended magnification from GULLS lightcurve data and header fs.
    
    Assume only one observatory in gulls simulation (usual smoke test case)."""
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
    return A_gulls


__all__ = ["read_gulls_lightcurve", "read_gulls_observatory_settings", "parse_out_file", "load_lightcurve"]
