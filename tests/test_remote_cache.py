import io
import json
import zipfile
from pathlib import Path

import pytest

from gulls_pipeline.presets import remote_cache


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(remote_cache.CACHE_DIR_ENV, str(cache_dir))
    return cache_dir


def make_zip(files):
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return handle.getvalue()


def test_get_cache_root_uses_environment_override(isolated_cache_dir):
    assert remote_cache.get_cache_root() == isolated_cache_dir


def test_refresh_fetches_latest_sha_and_extracts_archive(isolated_cache_dir):
    sha = "abc123"
    archive = make_zip(
        {
            f"Parameterfiles-{sha}/general_input/example/test.prm": "RUN_NAME test\n",
            f"Parameterfiles-{sha}/README.md": "docs\n",
        }
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url == remote_cache.REPO_API_URL:
            return FakeResponse(json.dumps({"sha": sha}).encode("utf-8"))
        if request.full_url == remote_cache.ZIP_ARCHIVE_URL.format(sha=sha):
            return FakeResponse(archive)
        raise AssertionError(f"unexpected URL {request.full_url}")

    result = remote_cache.refresh_parameterfiles_cache(opener=fake_urlopen)

    assert result.status == "synced"
    assert result.stale is False
    assert result.commit_sha == sha
    assert result.repo_path == isolated_cache_dir / sha
    assert (
        isolated_cache_dir / sha / "general_input" / "example" / "test.prm"
    ).read_text(encoding="utf-8") == "RUN_NAME test\n"
    assert calls == [
        remote_cache.REPO_API_URL,
        remote_cache.ZIP_ARCHIVE_URL.format(sha=sha),
    ]


def test_refresh_reuses_cached_sha_without_downloading(isolated_cache_dir):
    sha = "cached123"
    cached = isolated_cache_dir / sha / "general_input"
    cached.mkdir(parents=True)

    def fake_urlopen(request, timeout):
        if request.full_url == remote_cache.REPO_API_URL:
            return FakeResponse(json.dumps({"sha": sha}).encode("utf-8"))
        raise AssertionError("cached SHA should not be downloaded")

    result = remote_cache.refresh_parameterfiles_cache(opener=fake_urlopen)

    assert result.status == "synced"
    assert result.repo_path == isolated_cache_dir / sha
    assert result.commit_sha == sha


def test_refresh_falls_back_to_most_recent_cached_sha_when_offline(
    isolated_cache_dir,
):
    old = isolated_cache_dir / "oldsha" / "general_input"
    new = isolated_cache_dir / "newsha" / "general_input"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    old_repo = old.parent
    new_repo = new.parent
    old_repo.touch()
    new_repo.touch()

    def fake_urlopen(request, timeout):
        raise OSError("offline")

    result = remote_cache.refresh_parameterfiles_cache(opener=fake_urlopen)

    assert result.status == "stale"
    assert result.stale is True
    assert result.commit_sha == "newsha"
    assert result.repo_path == new_repo
    assert "offline" in result.error


def test_refresh_raises_when_offline_and_no_cached_copy():
    def fake_urlopen(request, timeout):
        raise OSError("offline")

    with pytest.raises(remote_cache.RemoteCacheError, match="no cached copy"):
        remote_cache.refresh_parameterfiles_cache(opener=fake_urlopen)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../evil.txt",
        "/absolute/evil.txt",
        "Parameterfiles-main/../../evil.txt",
    ],
)
def test_refresh_rejects_unsafe_zip_paths(unsafe_name):
    sha = "badsha"
    archive = make_zip(
        {
            f"Parameterfiles-{sha}/general_input/test.prm": "",
            unsafe_name: "evil",
        }
    )

    def fake_urlopen(request, timeout):
        if request.full_url == remote_cache.REPO_API_URL:
            return FakeResponse(json.dumps({"sha": sha}).encode("utf-8"))
        if request.full_url == remote_cache.ZIP_ARCHIVE_URL.format(sha=sha):
            return FakeResponse(archive)
        raise AssertionError(f"unexpected URL {request.full_url}")

    with pytest.raises(remote_cache.RemoteCacheError, match="Unsafe zip member path"):
        remote_cache.refresh_parameterfiles_cache(opener=fake_urlopen)


def test_result_as_dict_is_json_friendly(isolated_cache_dir):
    result = remote_cache.RemoteCacheResult(
        cache_root=isolated_cache_dir,
        repo_path=isolated_cache_dir / "sha",
        commit_sha="sha",
        status="synced",
    )

    assert result.as_dict() == {
        "cache_root": str(isolated_cache_dir),
        "repo_path": str(isolated_cache_dir / "sha"),
        "commit_sha": "sha",
        "status": "synced",
        "stale": False,
        "error": None,
    }
