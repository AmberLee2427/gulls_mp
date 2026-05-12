from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("smoke_test/output/ui")


@dataclass(frozen=True)
class LaunchResult:
    status: str
    run_dir: Path
    prm_path: Path
    message: str
    bundle_dir: Path | None = None
    job_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "run_dir": str(self.run_dir),
            "prm_path": str(self.prm_path),
            "message": self.message,
        }
        if self.bundle_dir is not None:
            data["bundle_dir"] = str(self.bundle_dir)
        if self.job_id is not None:
            data["job_id"] = self.job_id
        return data


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_root_from_config(config: dict[str, Any]) -> Path:
    base_dir = config.get("ui_run_root") or config.get("output_dir") or DEFAULT_RUN_ROOT
    return Path(str(base_dir)).expanduser()


def run_dir_from_config(config: dict[str, Any]) -> Path:
    run_name = _required_text(config, "run_name")
    return run_root_from_config(config) / run_name


def write_parameter_file(config: dict[str, Any], run_dir: Path) -> Path:
    run_name = _required_text(config, "run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(str(config.get("output_dir") or (run_dir / "output"))).expanduser()
    final_dir = Path(str(config.get("final_dir") or output_dir)).expanduser()
    (output_dir / run_name).mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    prm_path = input_dir / f"{run_name}.prm"
    registry = config.get("registry_lookups") or {}
    settings = dict(config.get("prm_settings") or {})
    settings.pop("RUN_NAME", None)
    settings.pop("EXECUTABLE", None)
    settings.pop("OUTPUT_DIR", None)
    settings.pop("FINAL_DIR", None)

    with prm_path.open("w", encoding="utf-8") as handle:
        handle.write(f"RUN_NAME={run_name}\n")
        handle.write(f"OUTPUT_DIR={_with_trailing_slash(output_dir)}\n")
        handle.write(f"FINAL_DIR={_with_trailing_slash(final_dir)}\n")
        handle.write(f"EXECUTABLE={_required_text(config, 'executable')}\n")

        handle.write("\n# OBSERVATORIES\n")
        handle.write(
            f"OBSERVATORY_DIR={config.get('observatory_dir') or 'smoke_test/assets/observatories/'}\n"
        )
        handle.write(f"OBSERVATORY_LIST={config.get('observatory_list') or 'smoke.list'}\n")
        handle.write(f"WEATHER_PROFILE_DIR={settings.pop('WEATHER_PROFILE_DIR', 'smoke_test/assets/weather/')}\n")

        base_catalog = str(registry.get("catalogs") or "")
        handle.write("\n# REGISTRIES\n")
        handle.write(f"STARFIELD_DIR={_catalog_dir(base_catalog, 'starfields')}\n")
        handle.write(f"STARFIELD_LIST={registry.get('starfields', '')}\n")
        handle.write(f"SOURCE_DIR={_catalog_dir(base_catalog, 'sources')}\n")
        handle.write(f"SOURCE_LIST={registry.get('sources', '')}\n")
        handle.write(f"LENS_DIR={_catalog_dir(base_catalog, 'lenses')}\n")
        handle.write(f"LENS_LIST={registry.get('lenses', '')}\n")
        handle.write(f"PLANET_DIR={_catalog_dir(base_catalog, 'planets')}\n")
        handle.write(f"PLANET_ROOT={registry.get('planet_root', '')}\n")
        handle.write(f"RATES_FILE={registry.get('rates', '')}\n")

        settings.setdefault("SOURCE_COLOURS", 0)
        settings.setdefault("LENS_COLOURS", 0)
        handle.write("\n# DYNAMIC SETTINGS\n")
        for key in sorted(settings):
            value = settings[key]
            if value is None or value == "":
                continue
            handle.write(f"{key}={value}\n")

    return prm_path


def run_local(config: dict[str, Any], cwd: Path | None = None) -> LaunchResult:
    cwd = cwd or Path.cwd()
    run_dir = run_dir_from_config(config)
    prm_path = write_parameter_file(config, run_dir)
    executable = _required_text(config, "executable")
    exec_path = resolve_executable(executable, cwd)
    gulls_base_dir = resolve_runtime_base(run_dir, cwd, exec_path)
    log_path = run_dir / "gulls_run.log"

    env = os.environ.copy()
    env["GULLS_BASE_DIR"] = _with_trailing_slash(gulls_base_dir)
    env["GULLS_BIN_DIR"] = str(exec_path.parent)
    if env.get("GULLS_STARS_DIR") is None:
        env["GULLS_STARS_DIR"] = env["GULLS_BASE_DIR"]

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [str(exec_path), "-i", str(prm_path), "-s", "0"],
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        proc.wait()

    return LaunchResult(
        status="completed",
        run_dir=run_dir,
        prm_path=prm_path,
        message=f"Local run finished. Log: {log_path}",
    )


def generate_slurm_bundle(config: dict[str, Any], cwd: Path | None = None) -> LaunchResult:
    cwd = cwd or Path.cwd()
    run_name = _required_text(config, "run_name")
    executable = _required_text(config, "executable")
    run_dir = run_dir_from_config(config)
    prm_path = write_parameter_file(config, run_dir)
    exec_path = resolve_executable(executable, cwd)
    gulls_base_dir = resolve_runtime_base(run_dir, cwd, exec_path)
    slurm_dir = run_dir / "slurm"
    fields_dir = slurm_dir / "fields"
    logs_dir = run_dir / "logs"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    slurm_config = config.get("slurm") or {}
    fields = resolve_fields(config, cwd)
    fields_per_task = max(1, int(slurm_config.get("fields_per_task") or 1))
    subruns = max(1, int((config.get("prm_settings") or {}).get("NSUBRUNS") or 1))
    chunks = _chunk(fields, fields_per_task)

    for idx, chunk in enumerate(chunks):
        (fields_dir / f"chunk_{idx:04d}.txt").write_text(
            "\n".join(chunk) + "\n",
            encoding="utf-8",
        )

    task_map_path = slurm_dir / "task_map.tsv"
    task_id = 0
    with task_map_path.open("w", encoding="utf-8") as handle:
        handle.write("task_id\tsubrun\tfields_file\n")
        for subrun in range(subruns):
            for idx in range(len(chunks)):
                handle.write(f"{task_id}\t{subrun}\tfields/chunk_{idx:04d}.txt\n")
                task_id += 1

    env_path = slurm_dir / "env.sh"
    env_path.write_text(
        _render_env_script(
            config,
            cwd,
            default_gulls_base_dir=gulls_base_dir,
            default_bin_dir=exec_path.parent,
        ),
        encoding="utf-8",
    )
    _make_executable(env_path)

    run_task_path = slurm_dir / "run_task.sh"
    run_task_path.write_text(
        _render_run_task_script(run_name, executable),
        encoding="utf-8",
    )
    _make_executable(run_task_path)

    submit_path = slurm_dir / "submit.sbatch"
    submit_path.write_text(
        _render_submit_script(run_name, slurm_config, max(task_id - 1, 0)),
        encoding="utf-8",
    )
    _make_executable(submit_path)

    manifest = {
        "run_name": run_name,
        "executable": executable,
        "prm_path": str(prm_path),
        "tasks": task_id,
        "subruns": subruns,
        "fields": len(fields),
        "fields_per_task": fields_per_task,
        "submit_script": str(submit_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    job_id = None
    if slurm_config.get("submit"):
        proc = subprocess.run(
            ["sbatch", str(submit_path)],
            cwd=slurm_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = proc.stdout.strip()

    message = f"SLURM bundle generated at {slurm_dir}. Submit with: sbatch {submit_path.name}"
    if job_id:
        message = f"Submitted SLURM job: {job_id}"
    return LaunchResult(
        status="submitted" if job_id else "bundle-created",
        run_dir=run_dir,
        prm_path=prm_path,
        bundle_dir=slurm_dir,
        job_id=job_id,
        message=message,
    )


def resolve_fields(config: dict[str, Any], cwd: Path | None = None) -> list[str]:
    cwd = cwd or Path.cwd()
    slurm_config = config.get("slurm") or {}
    field_spec = str(slurm_config.get("field_spec") or "auto").strip()
    if field_spec and field_spec.lower() != "auto":
        fields = _parse_field_spec(field_spec)
        if fields:
            return fields

    registry = config.get("registry_lookups") or {}
    source_list = str(registry.get("sources") or "").strip()
    if not source_list:
        raise ValueError("Cannot auto-discover fields because SOURCE_LIST is empty.")

    candidates = []
    catalogs = str(registry.get("catalogs") or "").split("|")
    for catalog in catalogs:
        catalog = catalog.strip()
        if not catalog:
            continue
        candidates.append(cwd / catalog / "sources" / source_list)
        candidates.append(cwd / catalog / source_list)
        candidates.append(Path(catalog).expanduser() / "sources" / source_list)
        candidates.append(Path(catalog).expanduser() / source_list)
    candidates.append(cwd / source_list)
    candidates.append(Path(source_list).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return _read_field_ids(candidate)

    searched = ", ".join(str(path) for path in candidates[:6])
    raise FileNotFoundError(f"Could not auto-discover fields from SOURCE_LIST. Tried: {searched}")


def resolve_executable(executable: str, cwd: Path | None = None) -> Path:
    """Find a Gulls executable in the source tree or installed package."""

    cwd = cwd or Path.cwd()
    candidates = [
        cwd / "bin" / executable,
        _package_path("gulls_pipeline.bin", executable),
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        _make_executable(candidate)
        if os.access(candidate, os.X_OK):
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        f"{executable} not found. Build Gulls or install a binary wheel. Searched: {searched}"
    )


def resolve_runtime_base(run_dir: Path, cwd: Path, exec_path: Path) -> Path:
    """Return a GULLS_BASE_DIR containing runtime files expected by C++."""

    source_bin = (cwd / "bin").resolve()
    try:
        if exec_path.resolve().parent == source_bin and (cwd / "src" / "ESPL.tbl").is_file():
            return cwd.resolve()
    except FileNotFoundError:
        pass
    return prepare_packaged_runtime_base(run_dir, cwd)


def prepare_packaged_runtime_base(run_dir: Path, cwd: Path) -> Path:
    """Stage runtime files beside a run when using packaged executables."""

    runtime_root = run_dir / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    espl_source = cwd / "src" / "ESPL.tbl"
    if not espl_source.is_file():
        espl_source = _package_path("gulls_pipeline.runtime", "src", "ESPL.tbl")
    if espl_source is None or not espl_source.is_file():
        raise FileNotFoundError(
            "Packaged runtime is missing src/ESPL.tbl; rebuild the binary wheel."
        )
    _link_or_copy(espl_source, runtime_root / "src" / "ESPL.tbl")

    smoke_root = _package_path("smoke_test")
    if smoke_root is not None and smoke_root.is_dir():
        _link_or_copy(smoke_root, runtime_root / "smoke_test")

    return runtime_root.resolve()


def _render_env_script(
    config: dict[str, Any],
    cwd: Path,
    *,
    default_gulls_base_dir: Path | None = None,
    default_bin_dir: Path | None = None,
) -> str:
    slurm_config = config.get("slurm") or {}
    prologue = str(slurm_config.get("prologue") or "").rstrip()
    gulls_base = str(slurm_config.get("gulls_base_dir") or default_gulls_base_dir or cwd)
    stars_dir = str(slurm_config.get("gulls_stars_dir") or "${GULLS_BASE_DIR}")
    bin_dir = str(slurm_config.get("bin_dir") or default_bin_dir or "${GULLS_BASE_DIR}/bin")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Cluster-specific setup. Edit this block for modules, conda, MPI, or filesystem setup.",
    ]
    if prologue:
        lines.append(prologue)
    else:
        lines.append("# module load gcc")
        lines.append("# conda activate gulls")
    lines.extend(
        [
            "",
            f"export GULLS_BASE_DIR=${{GULLS_BASE_DIR:-{_shell_default(_with_trailing_slash(gulls_base))}}}",
            f"export GULLS_STARS_DIR=${{GULLS_STARS_DIR:-{_shell_default(_with_trailing_slash(stars_dir))}}}",
            f"export GULLS_BIN_DIR=${{GULLS_BIN_DIR:-{_shell_default(_without_trailing_slash(bin_dir))}}}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_run_task_script(run_name: str, executable: str) -> str:
    quoted_executable = shlex.quote(executable)
    quoted_run_name = shlex.quote(run_name)
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"

TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
TASK_LINE="$(awk -F '\\t' -v id="$TASK_ID" 'NR > 1 && $1 == id {{ print; exit }}' "$SCRIPT_DIR/task_map.tsv")"
if [ -z "$TASK_LINE" ]; then
    echo "No task_map.tsv row for task $TASK_ID" >&2
    exit 2
fi

IFS=$'\\t' read -r _TASK_ID SUBRUN FIELDS_FILE <<< "$TASK_LINE"
RUN_NAME={quoted_run_name}
EXECUTABLE={quoted_executable}
PARAMFILE="$RUN_DIR/input/${{RUN_NAME}}.prm"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

while read -r FIELD; do
    [ -z "$FIELD" ] && continue
    [[ "$FIELD" =~ ^# ]] && continue
    LOG_FILE="$LOG_DIR/${{RUN_NAME}}_${{SUBRUN}}_${{FIELD}}.gullsstdout"
    echo "Running $EXECUTABLE subrun=${{SUBRUN}} field=${{FIELD}}"
    "$GULLS_BIN_DIR/$EXECUTABLE" -i "$PARAMFILE" -s "$SUBRUN" -f "$FIELD" > "$LOG_FILE" 2>&1
done < "$SCRIPT_DIR/$FIELDS_FILE"
"""


def _render_submit_script(run_name: str, slurm_config: dict[str, Any], max_task_id: int) -> str:
    throttle = int(slurm_config.get("max_concurrent") or 20)
    partition = str(slurm_config.get("partition") or "").strip()
    account = str(slurm_config.get("account") or "").strip()
    walltime = str(slurm_config.get("walltime") or "06:00:00").strip()
    cpus = int(slurm_config.get("cpus_per_task") or 1)
    memory = str(slurm_config.get("memory") or "").strip()
    extra = str(slurm_config.get("extra_sbatch") or "").strip()

    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=gulls-{run_name}",
        f"#SBATCH --array=0-{max_task_id}%{throttle}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --cpus-per-task={cpus}",
        "#SBATCH --output=slurm-%A_%a.out",
        "#SBATCH --error=slurm-%A_%a.err",
    ]
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if account:
        directives.append(f"#SBATCH --account={account}")
    if memory:
        directives.append(f"#SBATCH --mem={memory}")
    if extra:
        for line in extra.splitlines():
            line = line.strip()
            if line:
                directives.append(line if line.startswith("#SBATCH") else f"#SBATCH {line}")
    directives.extend(
        [
            "",
            "set -euo pipefail",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'bash "$SCRIPT_DIR/run_task.sh"',
            "",
        ]
    )
    return "\n".join(directives)


def _parse_field_spec(field_spec: str) -> list[str]:
    fields: list[str] = []
    for part in field_spec.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            fields.extend(str(value) for value in range(start, end + step, step))
        else:
            fields.append(part)
    return fields


def _read_field_ids(path: Path) -> list[str]:
    fields: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields.append(stripped.split()[0])
    if not fields:
        raise ValueError(f"No field ids found in {path}")
    return fields


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _catalog_dir(base_catalog: str, leaf: str) -> str:
    if not base_catalog or "|" in base_catalog:
        return base_catalog
    path = Path(base_catalog)
    if path.name == leaf:
        return _with_trailing_slash(str(path))
    return _with_trailing_slash(str(path / leaf))


def _required_text(config: dict[str, Any], key: str) -> str:
    value = str(config.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _with_trailing_slash(value: str | os.PathLike[str]) -> str:
    text = str(value)
    return text if text.endswith("/") else f"{text}/"


def _without_trailing_slash(value: str | os.PathLike[str]) -> str:
    text = str(value)
    return text[:-1] if text.endswith("/") else text


def _package_path(package: str, *parts: str) -> Path | None:
    try:
        ref = resources.files(package)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    for part in parts:
        ref = ref.joinpath(part)
    path = Path(str(ref))
    return path if path.exists() else None


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, dest, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)


def _shell_default(value: str) -> str:
    if "$" in value or "`" in value:
        return value
    return shlex.quote(value)


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
