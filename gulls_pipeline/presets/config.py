"""Persistent source configuration for gulls-ui presets.

The config file intentionally stays small and JSON-serializable:

{
  "version": 1,
  "local_preset_dirs": [
    {"id": "local:...", "path": "/abs/path", "label": "Parameterfiles"}
  ],
  "catalog_roots": [
    {"id": "catalog:stars:...", "kind": "stars", "path": "/abs/path", "label": "Stars"}
  ]
}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR_ENV = "GULLS_UI_CONFIG_DIR"
CONFIG_FILENAME = "sources.json"
CONFIG_VERSION = 1
CATALOG_KINDS = frozenset({"stars", "planets"})


def get_config_dir() -> Path:
    """Return the gulls-ui config directory.

    Tests and alternate deployments can override the default with
    ``GULLS_UI_CONFIG_DIR``.
    """

    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "gulls-ui"


def get_config_path() -> Path:
    """Return the path to the gulls-ui preset source config file."""

    return get_config_dir() / CONFIG_FILENAME


def default_config() -> dict[str, Any]:
    """Return an empty source config."""

    return {
        "version": CONFIG_VERSION,
        "local_preset_dirs": [],
        "catalog_roots": [],
    }


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``sources.json`` or return an empty config when it is absent."""

    config_path = Path(path) if path is not None else get_config_path()
    if not config_path.exists():
        return default_config()

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return _normalize_config(data)


def save_config(
    config: dict[str, Any], path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Validate and save a source config to ``sources.json``."""

    config_path = Path(path) if path is not None else get_config_path()
    normalized = _normalize_config(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.write("\n")
    return normalized


def load_sources(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Alias for ``load_config`` using the UI terminology."""

    return load_config(path)


def save_sources(
    config: dict[str, Any], path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Alias for ``save_config`` using the UI terminology."""

    return save_config(config, path)


def load_sources_json(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Alias for ``load_config`` that names the backing file."""

    return load_config(path)


def save_sources_json(
    config: dict[str, Any], path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Alias for ``save_config`` that names the backing file."""

    return save_config(config, path)


def list_local_preset_dirs(
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return configured local preset directories."""

    data = _normalize_config(config) if config is not None else load_config()
    return list(data["local_preset_dirs"])


def add_local_preset_dir(
    path: str | os.PathLike[str],
    label: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add or update a local preset directory and persist the config."""

    data = _normalize_config(config) if config is not None else load_config()
    directory = _validate_directory(path)
    entry = {
        "id": _entry_id("local", directory),
        "path": str(directory),
        "label": _clean_label(label) or directory.name,
    }

    entries = [item for item in data["local_preset_dirs"] if item["id"] != entry["id"]]
    entries.append(entry)
    data["local_preset_dirs"] = entries
    return save_config(data)


def add_local_preset_directory(
    path: str | os.PathLike[str],
    label: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias for ``add_local_preset_dir``."""

    return add_local_preset_dir(path, label, config=config)


def list_local_preset_directories(
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Alias for ``list_local_preset_dirs``."""

    return list_local_preset_dirs(config)


def remove_local_preset_dir(
    source_id_or_path: str | os.PathLike[str],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove a local preset directory by id or path and persist the config."""

    data = _normalize_config(config) if config is not None else load_config()
    key = str(source_id_or_path)
    path_key = _path_key_if_possible(source_id_or_path)
    data["local_preset_dirs"] = [
        entry
        for entry in data["local_preset_dirs"]
        if entry["id"] != key and entry["path"] != path_key
    ]
    return save_config(data)


def remove_local_preset_directory(
    source_id_or_path: str | os.PathLike[str],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias for ``remove_local_preset_dir``."""

    return remove_local_preset_dir(source_id_or_path, config=config)


def list_catalog_roots(
    kind: str | dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return configured external catalog roots, optionally filtered by kind."""

    if isinstance(kind, dict) and config is None:
        config = kind
        kind = None
    if kind is not None:
        _validate_catalog_kind(kind)
    data = _normalize_config(config) if config is not None else load_config()
    roots = data["catalog_roots"]
    if kind is None:
        return list(roots)
    return [entry for entry in roots if entry["kind"] == kind]


def add_catalog_root(
    path: str | os.PathLike[str],
    kind: str,
    label: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add or update an external stars or planets catalog root."""

    data = _normalize_config(config) if config is not None else load_config()
    catalog_kind = _validate_catalog_kind(kind)
    directory = _validate_directory(path)
    entry = {
        "id": _entry_id(f"catalog:{catalog_kind}", directory),
        "kind": catalog_kind,
        "path": str(directory),
        "label": _clean_label(label) or directory.name,
    }

    roots = [item for item in data["catalog_roots"] if item["id"] != entry["id"]]
    roots.append(entry)
    data["catalog_roots"] = roots
    return save_config(data)


def add_external_catalog_root(
    path: str | os.PathLike[str],
    kind: str,
    label: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias for ``add_catalog_root``."""

    return add_catalog_root(path, kind, label, config=config)


def remove_catalog_root(
    root_id_or_path: str | os.PathLike[str],
    kind: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove an external catalog root by id or path and persist the config."""

    if kind is not None:
        _validate_catalog_kind(kind)
    data = _normalize_config(config) if config is not None else load_config()
    key = str(root_id_or_path)
    path_key = _path_key_if_possible(root_id_or_path)

    data["catalog_roots"] = [
        entry
        for entry in data["catalog_roots"]
        if not (
            (entry["id"] == key or entry["path"] == path_key)
            and (kind is None or entry["kind"] == kind)
        )
    ]
    return save_config(data)


def list_external_catalog_roots(
    kind: str | dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Alias for ``list_catalog_roots``."""

    return list_catalog_roots(kind, config)


def remove_external_catalog_root(
    root_id_or_path: str | os.PathLike[str],
    kind: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Alias for ``remove_catalog_root``."""

    return remove_catalog_root(root_id_or_path, kind, config=config)


def _normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return default_config()
    if not isinstance(config, dict):
        raise ValueError("source config must be a dictionary")

    normalized = default_config()
    normalized["version"] = int(config.get("version", CONFIG_VERSION))
    normalized["local_preset_dirs"] = [
        _normalize_local_entry(entry) for entry in config.get("local_preset_dirs", [])
    ]
    normalized["catalog_roots"] = [
        _normalize_catalog_entry(entry) for entry in config.get("catalog_roots", [])
    ]
    return normalized


def _normalize_local_entry(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError("local preset directory entries must be objects")
    path = _path_string(entry.get("path"))
    label = _clean_label(entry.get("label")) or Path(path).name
    entry_id = _clean_label(entry.get("id")) or _entry_id("local", Path(path))
    return {"id": entry_id, "path": path, "label": label}


def _normalize_catalog_entry(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError("catalog root entries must be objects")
    kind = _validate_catalog_kind(entry.get("kind"))
    path = _path_string(entry.get("path"))
    label = _clean_label(entry.get("label")) or Path(path).name
    entry_id = _clean_label(entry.get("id")) or _entry_id(f"catalog:{kind}", Path(path))
    return {"id": entry_id, "kind": kind, "path": path, "label": label}


def _validate_directory(path: str | os.PathLike[str]) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"configured path does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"configured path is not a directory: {directory}")
    return directory


def _validate_catalog_kind(kind: Any) -> str:
    if not isinstance(kind, str):
        raise ValueError("catalog root kind must be 'stars' or 'planets'")
    normalized = kind.strip().lower()
    if normalized not in CATALOG_KINDS:
        raise ValueError("catalog root kind must be 'stars' or 'planets'")
    return normalized


def _entry_id(prefix: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def _path_string(path: Any) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("configured entries require a non-empty path")
    return str(Path(path).expanduser())


def _clean_label(label: Any) -> str:
    if label is None:
        return ""
    if not isinstance(label, str):
        raise ValueError("labels must be strings")
    return label.strip()


def _path_key_if_possible(path: str | os.PathLike[str]) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path)
