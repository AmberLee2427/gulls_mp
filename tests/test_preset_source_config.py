import json
from pathlib import Path

import pytest

from gulls_pipeline.presets import config


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(config_dir))
    return config_dir


def test_get_config_path_uses_environment_override(isolated_config_dir):
    assert config.get_config_dir() == isolated_config_dir
    assert config.get_config_path() == isolated_config_dir / "sources.json"


def test_load_missing_config_returns_empty_schema():
    assert config.load_config() == {
        "version": 1,
        "local_preset_dirs": [],
        "catalog_roots": [],
    }


def test_save_and_load_sources_json_round_trip(tmp_path):
    preset_dir = tmp_path / "parameterfiles"
    stars_dir = tmp_path / "stars"
    preset_dir.mkdir()
    stars_dir.mkdir()

    saved = config.add_local_preset_dir(preset_dir, label="Local Presets")
    saved = config.add_catalog_root(stars_dir, "stars", label="Roman Stars", config=saved)

    loaded = config.load_sources()
    assert loaded == saved
    assert json.loads(config.get_config_path().read_text(encoding="utf-8")) == saved


def test_add_local_preset_dir_validates_directory_and_updates_existing(tmp_path):
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()

    saved = config.add_local_preset_dir(preset_dir, label="First")
    saved = config.add_local_preset_dir(preset_dir, label="Renamed", config=saved)

    entries = config.list_local_preset_dirs(saved)
    assert len(entries) == 1
    assert entries[0]["path"] == str(preset_dir.resolve())
    assert entries[0]["label"] == "Renamed"
    assert entries[0]["id"].startswith("local:")


def test_add_local_preset_dir_rejects_missing_or_file_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.add_local_preset_dir(tmp_path / "missing")

    file_path = tmp_path / "not-a-dir"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        config.add_local_preset_dir(file_path)


def test_remove_local_preset_dir_by_id_or_path(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    saved = config.add_local_preset_dir(first)
    saved = config.add_local_preset_dir(second, config=saved)

    first_id = config.list_local_preset_dirs(saved)[0]["id"]
    saved = config.remove_local_preset_dir(first_id, config=saved)
    assert [entry["path"] for entry in config.list_local_preset_dirs(saved)] == [
        str(second.resolve())
    ]

    saved = config.remove_local_preset_dir(second, config=saved)
    assert config.list_local_preset_dirs(saved) == []


def test_add_list_and_remove_catalog_roots_by_kind(tmp_path):
    stars_dir = tmp_path / "stars"
    planets_dir = tmp_path / "planets"
    stars_dir.mkdir()
    planets_dir.mkdir()

    saved = config.add_catalog_root(stars_dir, "stars", label="Stars Root")
    saved = config.add_catalog_root(planets_dir, "planets", label="Planet Root", config=saved)

    stars = config.list_catalog_roots("stars", saved)
    planets = config.list_catalog_roots("planets", saved)
    assert len(stars) == 1
    assert len(planets) == 1
    assert stars[0] == {
        "id": stars[0]["id"],
        "kind": "stars",
        "path": str(stars_dir.resolve()),
        "label": "Stars Root",
    }
    assert planets[0]["kind"] == "planets"
    assert planets[0]["label"] == "Planet Root"

    saved = config.remove_catalog_root(stars[0]["id"], config=saved)
    assert config.list_catalog_roots("stars", saved) == []
    assert len(config.list_catalog_roots("planets", saved)) == 1

    saved = config.remove_catalog_root(planets_dir, kind="planets", config=saved)
    assert config.list_catalog_roots(saved) == []


def test_catalog_root_rejects_invalid_kind_and_path(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()

    with pytest.raises(ValueError):
        config.add_catalog_root(root, "galaxies")

    with pytest.raises(FileNotFoundError):
        config.add_catalog_root(tmp_path / "missing", "stars")


def test_saved_config_is_simple_json_serializable(tmp_path):
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()

    saved = config.add_local_preset_directory(Path(preset_dir), label="Presets")
    json.dumps(saved)
    assert config.load_sources()["local_preset_dirs"][0]["label"] == "Presets"
