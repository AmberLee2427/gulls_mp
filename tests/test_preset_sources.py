from pathlib import Path

from gulls_pipeline.presets import remote_cache, sources


def write_prm(path: Path, run_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"RUN_NAME={run_name}\nEXECUTABLE=gulls_general.x\n",
        encoding="utf-8",
    )
    return path


def test_display_name_from_relative_path_drops_extension_and_keeps_dirs():
    assert (
        sources.display_name_from_relative_path("general_input/RGES/CCS_m30.prm")
        == "general_input / RGES / CCS_m30"
    )


def test_remote_cache_provider_discovers_general_input_presets(tmp_path, monkeypatch):
    repo = tmp_path / "cache" / "abc123"
    first = write_prm(repo / "general_input" / "RGES" / "CCS_m30.prm", "rges_m30")
    write_prm(repo / "other" / "ignored.prm", "ignored")

    def fake_refresh_parameterfiles_cache(**kwargs):
        return remote_cache.RemoteCacheResult(
            cache_root=tmp_path / "cache",
            repo_path=repo,
            commit_sha="abc123",
            status="synced",
        )

    monkeypatch.setattr(
        remote_cache,
        "refresh_parameterfiles_cache",
        fake_refresh_parameterfiles_cache,
    )

    provider = sources.RemoteCacheProvider()
    records = provider.discover()

    assert provider.status.status == "synced"
    assert provider.status.stale is False
    assert provider.status.metadata["commit_sha"] == "abc123"
    assert len(records) == 1
    assert records[0].id == "remote:abc123:general_input/RGES/CCS_m30.prm"
    assert records[0].display_name == "RGES / CCS_m30"
    assert records[0].prm_path == first.resolve()
    assert records[0].source_root == repo.resolve()
    assert records[0].preset_root == first.parent.resolve()
    assert records[0].parsed_settings["RUN_NAME"] == "rges_m30"
    assert records[0].provenance == {
        "repo_url": sources.RemoteCacheProvider.repo_url,
        "commit": "abc123",
        "relative_prm": "general_input/RGES/CCS_m30.prm",
    }
    assert records[0].referenced_files == []
    assert records[0].missing_dependencies == []


def test_remote_cache_provider_reports_unavailable_without_network(
    tmp_path, monkeypatch
):
    def fake_refresh_parameterfiles_cache(**kwargs):
        raise remote_cache.RemoteCacheError("offline")

    monkeypatch.setattr(
        remote_cache,
        "refresh_parameterfiles_cache",
        fake_refresh_parameterfiles_cache,
    )

    provider = sources.RemoteCacheProvider(cache_root=tmp_path / "cache")
    records = provider.discover()

    assert records == []
    assert provider.status.status == "unavailable"
    assert provider.status.stale is True
    assert provider.status.root == tmp_path / "cache"
    assert "offline" in provider.status.message


def test_remote_cache_provider_reports_stale_cached_result(tmp_path, monkeypatch):
    repo = tmp_path / "cache" / "oldsha"
    write_prm(repo / "general_input" / "Survey" / "cached.prm", "cached")

    def fake_refresh_parameterfiles_cache(**kwargs):
        return remote_cache.RemoteCacheResult(
            cache_root=tmp_path / "cache",
            repo_path=repo,
            commit_sha="oldsha",
            status="stale",
            stale=True,
            error="offline",
        )

    monkeypatch.setattr(
        remote_cache,
        "refresh_parameterfiles_cache",
        fake_refresh_parameterfiles_cache,
    )

    provider = sources.RemoteCacheProvider()
    records = provider.discover()

    assert provider.status.status == "stale"
    assert provider.status.stale is True
    assert provider.status.message == "offline"
    assert records[0].source_metadata["stale"] is True


def test_smoke_provider_walks_upward_and_discovers_repo_parameterfiles(tmp_path):
    repo = tmp_path / "checkout"
    start = repo / "nested" / "dir"
    start.mkdir(parents=True)
    placeholder = repo / "smoke_test" / "parameterfiles" / "smoke.prm"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text(
        "### Placeholder only\n",
        encoding="utf-8",
    )
    first = write_prm(
        repo / "smoke_test" / "parameterfiles" / "smoke_1s1l.prm",
        "smoke",
    )
    write_prm(
        repo / "smoke_test" / "parameterfiles" / "nested" / "ignored.prm",
        "ignored",
    )

    provider = sources.SmokeProvider(start)
    records = provider.discover()

    assert sources.find_smoke_repo_root(start) == repo
    assert provider.status.status == "available"
    assert provider.status.root == repo.resolve()
    assert provider.status.metadata["asset_roots"] == [
        str(repo.resolve() / "smoke_test" / "assets"),
        str(repo.resolve() / "smoke_test" / "parameterfiles"),
    ]
    assert len(records) == 1
    assert records[0].id == "smoke:smoke_1s1l.prm"
    assert records[0].display_name == "smoke_1s1l"
    assert records[0].prm_path == first.resolve()
    assert records[0].provenance["relative_prm"] == (
        "smoke_test/parameterfiles/smoke_1s1l.prm"
    )


def test_smoke_provider_reports_unavailable_when_repo_not_found(tmp_path):
    provider = sources.SmokeProvider(tmp_path / "not" / "a" / "repo")

    assert provider.discover() == []
    assert provider.status.status == "unavailable"
    assert "smoke_test/parameterfiles" in provider.status.message


def test_local_directory_provider_discovers_configured_dirs_and_missing_dirs(
    tmp_path,
):
    local_root = tmp_path / "local"
    first = write_prm(local_root / "Survey" / "first.prm", "first")
    second = write_prm(local_root / "second.prm", "second")
    missing = tmp_path / "missing"
    source_config = {
        "version": 1,
        "local_preset_dirs": [
            {"id": "local:one", "path": str(local_root), "label": "Local Presets"},
            {"id": "local:missing", "path": str(missing), "label": "Missing"},
        ],
        "catalog_roots": [],
    }

    provider = sources.LocalDirectoryProvider(source_config=source_config)
    records = provider.discover()

    assert provider.status.status == "partial"
    assert provider.status.metadata == {
        "configured_count": 2,
        "available_count": 1,
        "missing_count": 1,
    }
    assert [status.status for status in provider.source_statuses] == [
        "available",
        "missing",
    ]
    assert [record.id for record in records] == [
        "local:one:Survey/first.prm",
        "local:one:second.prm",
    ]
    assert [record.display_name for record in records] == [
        "Survey / first",
        "second",
    ]
    assert records[0].prm_path == first.resolve()
    assert records[1].prm_path == second.resolve()
    assert records[0].source_root == local_root.resolve()
    assert records[0].provenance == {
        "source_id": "local:one",
        "label": "Local Presets",
        "relative_prm": "Survey/first.prm",
    }


def test_local_directory_provider_loads_config_from_path(tmp_path):
    local_root = tmp_path / "local"
    write_prm(local_root / "preset.prm", "loaded")
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        (
            '{"version": 1, "local_preset_dirs": ['
            f'{{"id": "local:file", "path": "{local_root}", "label": "From File"}}'
            '], "catalog_roots": []}'
        ),
        encoding="utf-8",
    )

    provider = sources.LocalDirectoryProvider(config_path=config_path)
    records = provider.discover()

    assert provider.status.status == "available"
    assert len(records) == 1
    assert records[0].id == "local:file:preset.prm"
    assert records[0].parsed_settings["RUN_NAME"] == "loaded"


def test_discover_all_presets_includes_local_summary_and_source_details(tmp_path):
    repo = tmp_path / "checkout"
    write_prm(repo / "smoke_test" / "parameterfiles" / "smoke.prm", "smoke")
    local_root = tmp_path / "local"
    write_prm(local_root / "preset.prm", "local")
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        (
            '{"version": 1, "local_preset_dirs": ['
            f'{{"id": "local:file", "path": "{local_root}", "label": "From File"}}'
            '], "catalog_roots": []}'
        ),
        encoding="utf-8",
    )

    records, statuses = sources.discover_all_presets(
        start_path=repo,
        config_path=config_path,
        include_remote=False,
    )

    assert [record.source_type for record in records] == [
        "smoke",
        "local-directory",
    ]
    assert [(status.source_type, status.status) for status in statuses] == [
        ("smoke", "available"),
        ("local-directory", "available"),
        ("local-directory", "available"),
    ]
    assert statuses[1].metadata == {
        "configured_count": 1,
        "available_count": 1,
        "missing_count": 0,
    }
    assert statuses[2].metadata["preset_count"] == 1


def test_preset_record_and_source_status_as_dict_are_json_friendly(tmp_path):
    prm = write_prm(tmp_path / "preset.prm", "json")
    record = sources.PresetRecord(
        id="local:test:preset.prm",
        source_type="local-directory",
        display_name="preset",
        prm_path=prm,
        source_root=tmp_path,
        preset_root=tmp_path,
        parsed_settings={"RUN_NAME": "json"},
        provenance={"relative_prm": "preset.prm"},
    )
    status = sources.SourceStatus(
        source_type="local-directory",
        status="available",
        root=tmp_path,
    )

    assert record.as_dict()["prm_path"] == str(prm)
    assert record.as_dict()["source_root"] == str(tmp_path)
    assert status.as_dict()["root"] == str(tmp_path)
