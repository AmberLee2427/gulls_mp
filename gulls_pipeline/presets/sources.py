"""Preset source providers for gulls-ui.

This layer only discovers parameter files and normalizes source metadata.  It
does not resolve transitive dependencies; later pipeline stages own that work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from . import config, parser, remote_cache


@dataclass(frozen=True)
class SourceStatus:
    """Status for a preset provider discovery attempt."""

    source_type: str
    status: str
    message: str | None = None
    stale: bool = False
    root: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly status metadata."""

        return {
            "source_type": self.source_type,
            "status": self.status,
            "message": self.message,
            "stale": self.stale,
            "root": str(self.root) if self.root is not None else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PresetRecord:
    """Normalized preset metadata for a discovered ``.prm`` file."""

    id: str
    source_type: str
    display_name: str
    prm_path: Path
    source_root: Path
    preset_root: Path
    parsed_settings: dict[str, str]
    provenance: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    catalog_requirements: dict[str, Any] = field(default_factory=dict)
    referenced_files: list[dict[str, Any]] = field(default_factory=list)
    missing_dependencies: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for API/UI callers."""

        return {
            "id": self.id,
            "source_type": self.source_type,
            "display_name": self.display_name,
            "prm_path": str(self.prm_path),
            "source_root": str(self.source_root),
            "preset_root": str(self.preset_root),
            "parsed_settings": dict(self.parsed_settings),
            "catalog_requirements": self.catalog_requirements,
            "referenced_files": self.referenced_files,
            "missing_dependencies": self.missing_dependencies,
            "provenance": self.provenance,
            "source_metadata": self.source_metadata,
        }


class RemoteCacheProvider:
    """Discover presets from the cached upstream Parameterfiles repository."""

    source_type = "remote-cache"
    repo_url = "https://github.com/gulls-microlensing/Parameterfiles.git"

    def __init__(
        self,
        *,
        cache_root: str | os.PathLike[str] | None = None,
        opener: remote_cache.UrlOpen | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.cache_root = cache_root
        self.opener = opener
        self.timeout = timeout
        self.status = SourceStatus(source_type=self.source_type, status="not-run")

    def discover(self) -> list[PresetRecord]:
        """Refresh the cache and discover ``general_input/**/*.prm`` presets."""

        try:
            result = remote_cache.refresh_parameterfiles_cache(
                cache_root=self.cache_root,
                opener=self.opener,
                timeout=self.timeout,
            )
        except remote_cache.RemoteCacheError as exc:
            root = (
                Path(self.cache_root).expanduser()
                if self.cache_root is not None
                else remote_cache.get_cache_root()
            )
            self.status = SourceStatus(
                source_type=self.source_type,
                status="unavailable",
                message=str(exc),
                stale=True,
                root=root,
            )
            return []

        self.status = SourceStatus(
            source_type=self.source_type,
            status=result.status,
            message=result.error,
            stale=result.stale,
            root=result.repo_path,
            metadata={
                "cache_root": str(result.cache_root),
                "commit_sha": result.commit_sha,
                "repo_url": self.repo_url,
            },
        )
        if result.repo_path is None:
            return []

        source_root = result.repo_path.resolve()
        commit = result.commit_sha or source_root.name
        return [
            record
            for path in _sorted_prm_files(source_root / "general_input", "**/*.prm")
            if (
                record := _record_from_prm(
                    prm_path=path,
                    source_type=self.source_type,
                    source_root=source_root,
                    id_prefix=f"remote:{commit}",
                    display_root=source_root / "general_input",
                    provenance={
                        "repo_url": self.repo_url,
                        "commit": commit,
                        "relative_prm": _posix_relative(path, source_root),
                    },
                    source_metadata={
                        "stale": result.stale,
                        "status": result.status,
                        "cache_root": str(result.cache_root),
                    },
                )
            )
            is not None
        ]


class SmokeProvider:
    """Discover smoke-test presets from a Gulls checkout."""

    source_type = "smoke"

    def __init__(self, start_path: str | os.PathLike[str] | None = None) -> None:
        self.start_path = Path(start_path).expanduser() if start_path else Path.cwd()
        self.status = SourceStatus(source_type=self.source_type, status="not-run")

    def discover(self) -> list[PresetRecord]:
        repo_root = find_smoke_repo_root(self.start_path)
        if repo_root is None:
            self.status = SourceStatus(
                source_type=self.source_type,
                status="unavailable",
                message="Could not find smoke_test/parameterfiles from start path",
                root=self.start_path,
            )
            return []

        source_root = repo_root.resolve()
        parameter_root = source_root / "smoke_test" / "parameterfiles"
        asset_roots = [
            source_root / "smoke_test" / "assets",
            parameter_root,
        ]
        self.status = SourceStatus(
            source_type=self.source_type,
            status="available",
            root=source_root,
            metadata={"asset_roots": [str(path) for path in asset_roots]},
        )

        return [
            record
            for path in _sorted_prm_files(parameter_root, "*.prm")
            if (
                record := _record_from_prm(
                    prm_path=path,
                    source_type=self.source_type,
                    source_root=source_root,
                    id_prefix="smoke",
                    relative_root=parameter_root,
                    provenance={
                        "repo_root": str(source_root),
                        "relative_prm": _posix_relative(path, source_root),
                        "asset_roots": [str(path) for path in asset_roots],
                    },
                    source_metadata={"asset_roots": [str(path) for path in asset_roots]},
                )
            )
            is not None
        ]


class LocalDirectoryProvider:
    """Discover presets from user-configured local preset directories."""

    source_type = "local-directory"

    def __init__(
        self,
        *,
        config_path: str | os.PathLike[str] | None = None,
        source_config: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = config_path
        self.source_config = source_config
        self.status = SourceStatus(source_type=self.source_type, status="not-run")
        self.source_statuses: list[SourceStatus] = []

    def discover(self) -> list[PresetRecord]:
        try:
            local_preset_dirs = (
                config.list_local_preset_dirs(self.source_config)
                if self.source_config is not None
                else config.list_local_preset_dirs(config.load_config(self.config_path))
            )
        except Exception as exc:
            self.status = SourceStatus(
                source_type=self.source_type,
                status="unavailable",
                message=str(exc),
            )
            self.source_statuses = [self.status]
            return []

        records: list[PresetRecord] = []
        statuses: list[SourceStatus] = []
        for entry in local_preset_dirs:
            root = Path(entry["path"]).expanduser()
            label = entry.get("label") or root.name
            metadata = {"id": entry["id"], "label": label}
            if not root.is_dir():
                statuses.append(
                    SourceStatus(
                        source_type=self.source_type,
                        status="missing",
                        message=f"Configured local preset directory is missing: {root}",
                        root=root,
                        metadata=metadata,
                    )
                )
                continue

            resolved_root = root.resolve()
            matches = _sorted_prm_files(resolved_root, "**/*.prm")
            statuses.append(
                SourceStatus(
                    source_type=self.source_type,
                    status="available",
                    root=resolved_root,
                    metadata={**metadata, "preset_count": len(matches)},
                )
            )
            records.extend(
                record
                for path in matches
                if (
                    record := _record_from_prm(
                        prm_path=path,
                        source_type=self.source_type,
                        source_root=resolved_root,
                        id_prefix=entry["id"],
                        provenance={
                            "source_id": entry["id"],
                            "label": label,
                            "relative_prm": _posix_relative(path, resolved_root),
                        },
                        source_metadata={"source_id": entry["id"], "label": label},
                    )
                )
                is not None
            )

        self.source_statuses = statuses
        missing = sum(1 for status in statuses if status.status == "missing")
        self.status = SourceStatus(
            source_type=self.source_type,
            status="available" if not missing else "partial",
            message=f"{missing} configured local preset directories are missing"
            if missing
            else None,
            metadata={
                "configured_count": len(local_preset_dirs),
                "available_count": len(statuses) - missing,
                "missing_count": missing,
            },
        )
        return records


def find_smoke_repo_root(start_path: str | os.PathLike[str]) -> Path | None:
    """Walk upward from ``start_path`` until smoke parameter files are found."""

    start = Path(start_path).expanduser()
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "smoke_test" / "parameterfiles").is_dir():
            return candidate
    return None


def discover_all_presets(
    *,
    start_path: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    include_remote: bool = True,
) -> tuple[list[PresetRecord], list[SourceStatus]]:
    """Discover presets from all configured providers."""

    providers: list[Any] = []
    if include_remote:
        providers.append(RemoteCacheProvider())
    providers.extend(
        [
            SmokeProvider(start_path=start_path),
            LocalDirectoryProvider(config_path=config_path),
        ]
    )

    records: list[PresetRecord] = []
    statuses: list[SourceStatus] = []
    for provider in providers:
        records.extend(provider.discover())
        statuses.append(provider.status)
        source_statuses = getattr(provider, "source_statuses", None)
        if source_statuses:
            statuses.extend(source_statuses)
    return records, statuses


def _record_from_prm(
    *,
    prm_path: Path,
    source_type: str,
    source_root: Path,
    id_prefix: str,
    provenance: dict[str, Any],
    source_metadata: dict[str, Any],
    relative_root: Path | None = None,
    display_root: Path | None = None,
) -> PresetRecord | None:
    prm_path = prm_path.resolve()
    source_root = source_root.resolve()
    root_for_relative = (
        relative_root.resolve() if relative_root is not None else source_root
    )
    relative_prm = _posix_relative(prm_path, root_for_relative)
    display_relative_prm = _posix_relative(
        prm_path,
        display_root.resolve() if display_root is not None else root_for_relative,
    )
    document = parser.parse_prm_file(prm_path)
    if not _is_runnable_preset(document.settings):
        return None
    return PresetRecord(
        id=f"{id_prefix}:{relative_prm}",
        source_type=source_type,
        display_name=display_name_from_relative_path(display_relative_prm),
        prm_path=prm_path,
        source_root=source_root,
        preset_root=prm_path.parent,
        parsed_settings=dict(document.settings),
        provenance=provenance,
        source_metadata=source_metadata,
    )


def _is_runnable_preset(settings: dict[str, str]) -> bool:
    return bool(settings.get("RUN_NAME") and settings.get("EXECUTABLE"))


def display_name_from_relative_path(path: str | os.PathLike[str]) -> str:
    """Build a stable, readable display name from a relative ``.prm`` path."""

    rel = Path(path)
    parts = list(rel.with_suffix("").parts)
    return " / ".join(parts)


def _sorted_prm_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob(pattern) if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def _posix_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
