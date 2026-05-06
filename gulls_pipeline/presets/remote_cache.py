"""Remote Parameterfiles cache for gulls-ui presets.

This module intentionally uses only the Python standard library.  It keeps a
commit-addressed cache of the upstream Parameterfiles repository so later
provider/scanner code can discover presets from a stable local path.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Callable
from urllib.request import Request, urlopen
import zipfile

CACHE_DIR_ENV = "GULLS_UI_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gulls-ui" / "parameterfiles"
REPO_OWNER = "gulls-microlensing"
REPO_NAME = "Parameterfiles"
REPO_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/main"
ZIP_ARCHIVE_URL = f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/zip/{{sha}}"
USER_AGENT = "gulls-ui-parameterfiles-cache"

UrlOpen = Callable[..., Any]


class RemoteCacheError(RuntimeError):
    """Raised when the remote cache cannot be refreshed or reused."""


@dataclass(frozen=True)
class RemoteCacheResult:
    """Metadata returned by remote cache refresh operations."""

    cache_root: Path
    repo_path: Path | None
    commit_sha: str | None
    status: str
    stale: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        """Return JSON-friendly status metadata."""

        return {
            "cache_root": str(self.cache_root),
            "repo_path": str(self.repo_path) if self.repo_path is not None else None,
            "commit_sha": self.commit_sha,
            "status": self.status,
            "stale": self.stale,
            "error": self.error,
        }


def get_cache_root() -> Path:
    """Return the Parameterfiles cache root.

    Tests and alternate deployments can override the default with
    ``GULLS_UI_CACHE_DIR``.
    """

    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CACHE_DIR.expanduser()


def fetch_latest_commit_sha(
    *, opener: UrlOpen | None = None, timeout: float = 10.0
) -> str:
    """Fetch the latest ``main`` commit SHA from the GitHub API."""

    data = _read_json_url(REPO_API_URL, opener=opener, timeout=timeout)
    sha = data.get("sha")
    if not isinstance(sha, str) or not sha:
        raise RemoteCacheError("GitHub API response did not contain a commit SHA")
    return sha


def download_zip_archive(
    sha: str, *, opener: UrlOpen | None = None, timeout: float = 30.0
) -> bytes:
    """Download the GitHub zip archive for ``sha``."""

    return _read_bytes_url(
        ZIP_ARCHIVE_URL.format(sha=sha), opener=opener, timeout=timeout
    )


def refresh_parameterfiles_cache(
    *,
    cache_root: str | os.PathLike[str] | None = None,
    opener: UrlOpen | None = None,
    timeout: float = 30.0,
) -> RemoteCacheResult:
    """Refresh and return the local Parameterfiles cache.

    If the network refresh fails, the newest existing cached commit directory is
    reused and the result is marked ``stale``.  If no cached commit exists, a
    ``RemoteCacheError`` is raised.
    """

    root = Path(cache_root).expanduser() if cache_root is not None else get_cache_root()
    root.mkdir(parents=True, exist_ok=True)

    try:
        sha = fetch_latest_commit_sha(opener=opener, timeout=timeout)
    except Exception as exc:
        return _fallback_result(root, exc)

    repo_path = root / sha
    if _is_usable_cache_dir(repo_path):
        return RemoteCacheResult(
            cache_root=root,
            repo_path=repo_path,
            commit_sha=sha,
            status="synced",
            stale=False,
        )

    try:
        archive = download_zip_archive(sha, opener=opener, timeout=timeout)
    except Exception as exc:
        return _fallback_result(root, exc)

    _extract_archive_for_sha(archive, repo_path)
    return RemoteCacheResult(
        cache_root=root,
        repo_path=repo_path,
        commit_sha=sha,
        status="synced",
        stale=False,
    )


def get_parameterfiles_cache(
    *,
    cache_root: str | os.PathLike[str] | None = None,
    opener: UrlOpen | None = None,
    timeout: float = 30.0,
) -> RemoteCacheResult:
    """Alias for ``refresh_parameterfiles_cache`` using UI terminology."""

    return refresh_parameterfiles_cache(
        cache_root=cache_root, opener=opener, timeout=timeout
    )


def ensure_parameterfiles_cache(
    *,
    cache_root: str | os.PathLike[str] | None = None,
    opener: UrlOpen | None = None,
    timeout: float = 30.0,
) -> RemoteCacheResult:
    """Alias for callers that want a usable cache directory."""

    return refresh_parameterfiles_cache(
        cache_root=cache_root, opener=opener, timeout=timeout
    )


def latest_cached_sha(
    cache_root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Return the most recently modified usable cached SHA directory."""

    root = Path(cache_root).expanduser() if cache_root is not None else get_cache_root()
    if not root.exists():
        return None

    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and _is_usable_cache_dir(child)
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.name


def _fallback_result(root: Path, exc: Exception) -> RemoteCacheResult:
    cached = latest_cached_sha(root)
    if cached is None:
        raise RemoteCacheError(
            f"Unable to refresh Parameterfiles cache and no cached copy exists: {exc}"
        ) from exc
    return RemoteCacheResult(
        cache_root=root,
        repo_path=root / cached,
        commit_sha=cached,
        status="stale",
        stale=True,
        error=str(exc),
    )


def _read_json_url(
    url: str, *, opener: UrlOpen | None, timeout: float
) -> dict[str, Any]:
    payload = _read_bytes_url(url, opener=opener, timeout=timeout)
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise RemoteCacheError("GitHub API response was not a JSON object")
    return data


def _read_bytes_url(url: str, *, opener: UrlOpen | None, timeout: float) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    open_func = opener or urlopen
    with open_func(request, timeout=timeout) as response:
        return response.read()


def _is_usable_cache_dir(path: Path) -> bool:
    return path.is_dir() and (path / "general_input").is_dir()


def _extract_archive_for_sha(archive: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp_name:
        tmp_path = Path(tmp_name)
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            _safe_extract_zip(zip_file, tmp_path)

        extracted_root = _find_extracted_repo_root(tmp_path)
        if not (extracted_root / "general_input").is_dir():
            raise RemoteCacheError("Downloaded archive did not contain general_input/")

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted_root), destination)


def _safe_extract_zip(zip_file: zipfile.ZipFile, destination: Path) -> None:
    for member in zip_file.infolist():
        relative = _safe_member_path(member.filename)
        if relative is None:
            continue

        target = destination / relative
        resolved_target = target.resolve()
        resolved_destination = destination.resolve()
        if (
            resolved_target != resolved_destination
            and resolved_destination not in resolved_target.parents
        ):
            raise RemoteCacheError(f"Unsafe zip member path: {member.filename}")

        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(member) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def _safe_member_path(filename: str) -> PurePosixPath | None:
    path = PurePosixPath(filename)
    if not filename or path.is_absolute():
        raise RemoteCacheError(f"Unsafe zip member path: {filename}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RemoteCacheError(f"Unsafe zip member path: {filename}")
    if path.name == "":
        return None
    return path


def _find_extracted_repo_root(tmp_path: Path) -> Path:
    children = [child for child in tmp_path.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    if (tmp_path / "general_input").is_dir():
        return tmp_path
    raise RemoteCacheError("Downloaded archive had an unexpected directory layout")
