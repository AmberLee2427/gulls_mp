from __future__ import annotations

from pathlib import Path

from gulls_pipeline import launcher


def test_generate_slurm_bundle_from_auto_fields(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "gulls_general.x"
    executable.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    executable.chmod(0o755)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "ESPL.tbl").write_text("stub\n", encoding="utf-8")

    catalog_root = tmp_path / "catalogs"
    source_dir = catalog_root / "sources"
    source_dir.mkdir(parents=True)
    (source_dir / "fields.sources").write_text(
        "10 1.0 2.0\n11 1.0 2.0\n12 1.0 2.0\n",
        encoding="utf-8",
    )

    config = {
        "run_name": "demo",
        "scheduler": "slurm",
        "output_dir": str(tmp_path / "runs"),
        "final_dir": str(tmp_path / "final"),
        "executable": "gulls_general.x",
        "registry_lookups": {
            "catalogs": str(catalog_root),
            "sources": "fields.sources",
        },
        "prm_settings": {
            "NSUBRUNS": 2,
            "SUBRUNSIZE": 100,
        },
        "slurm": {
            "field_spec": "auto",
            "fields_per_task": 2,
            "max_concurrent": 3,
            "walltime": "01:00:00",
        },
    }

    result = launcher.generate_slurm_bundle(config, cwd=tmp_path)

    assert result.status == "bundle-created"
    assert result.bundle_dir is not None
    assert (result.bundle_dir / "submit.sbatch").exists()
    assert (result.bundle_dir / "run_task.sh").exists()
    assert (result.bundle_dir / "env.sh").exists()
    assert (result.bundle_dir / "fields/chunk_0000.txt").read_text(encoding="utf-8") == "10\n11\n"
    assert (result.bundle_dir / "fields/chunk_0001.txt").read_text(encoding="utf-8") == "12\n"

    task_map = (result.bundle_dir / "task_map.tsv").read_text(encoding="utf-8").splitlines()
    assert task_map[0] == "task_id\tsubrun\tfields_file"
    assert len(task_map) == 5
    assert "0\t0\tfields/chunk_0000.txt" in task_map
    assert "3\t1\tfields/chunk_0001.txt" in task_map

    submit = (result.bundle_dir / "submit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3%3" in submit
    assert "#SBATCH --time=01:00:00" in submit


def test_resolve_fields_from_explicit_spec():
    config = {"slurm": {"field_spec": "1,3-5,9"}}

    assert launcher.resolve_fields(config) == ["1", "3", "4", "5", "9"]
