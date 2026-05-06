# Gulls UI Presets Implementation Plan

**Goal:** Make `gulls-ui` discover, inspect, edit, and launch runs from real parameter-file presets without requiring users to manually clone `Parameterfiles` or edit paths.

**Architecture:** Add a backend preset layer with three providers: remote `Parameterfiles` cache, in-repo smoke presets, and user-configured local preset directories. Every provider emits the same normalized preset model, and every launch materializes lightweight run-control inputs into a per-run `input/` snapshot while keeping heavyweight catalogs in their existing configured locations before writing `generated.prm` and starting Gulls.

**Tech Stack:** Python stdlib, FastAPI, Vue in `gulls_pipeline/web/index.html`, existing Gulls executables, optional network access to GitHub zip archives.

---

## Current Constraints

- The UI currently hardcodes one smoke-style form in `gulls_pipeline/web/index.html`.
- `gulls_pipeline/cli.py` writes a partial `.prm` directly from JSON rather than preserving a selected preset.
- Users should be able to run `gulls-ui` after cloning/building Gulls and get useful presets immediately.
- The UI should fetch the latest remote presets before users configure a run, so dependent parameters are visible before launch.
- Presets should also come from smoke tests and user-added local directories.
- A launched run must be reproducible even if the remote preset cache updates later.
- Catalog assets can be huge. Do not copy starfield/source/lens/planet catalogs into `input/` except for tiny smoke-test assets.

## Gulls Path Resolution Rules

These rules must drive `generated.prm`; do not invent UI-only path semantics.

- `GULLS_BASE_DIR` is required. `src/readParamfile.cpp` exits if it is missing.
- `GULLS_STARS_DIR` is optional. If unset, Gulls uses `GULLS_BASE_DIR`.
- The `.prm` path passed with `-i` is opened exactly as passed; it is not resolved relative to `GULLS_BASE_DIR`.
- `OBSERVATORY_DIR`, `WEATHER_PROFILE_DIR`, and `PLANET_DIR` are prefixed with `GULLS_BASE_DIR`.
- `STARFIELD_DIR`, `SOURCE_DIR`, and `LENS_DIR` are prefixed with `GULLS_STARS_DIR`.
- `OUTPUT_DIR` is not prefixed with `GULLS_BASE_DIR`; Gulls computes `OUTPUT_DIR + RUN_NAME + "/"`.
- Observatory-list entries are joined as `OBSERVATORY_DIR + entry`, not relative to the list file.
- Observatory subfiles are also relative to `OBSERVATORY_DIR`: `OBSERVATION_SEQUENCE`, `DETECTOR`, `THROUGHPUT`, and file-like `ORBIT` values.
- `WEATHER_PROFILE` values in observatory files are relative to `WEATHER_PROFILE_DIR`.
- Starfield/source/lens list entries are relative to their respective directories.
- Planet files are opened as `PLANET_DIR + PLANET_ROOT + [field.] + instance`.
- Because planet files currently use `GULLS_BASE_DIR` instead of `GULLS_STARS_DIR`, real planet-population assets need special handling until the executable path semantics are changed.

Recommended launch layout:

```text
gulls_runs/<run_name>/
  input/
    generated.prm
    observatories/
    weather/
    rates/
    planets -> /abs/path/to/external/planet/root/  # symlink only when needed
  output/
    <run_name>/
  logs/
    gulls_run.log
  provenance.json
```

Recommended launch environment:

```text
GULLS_BASE_DIR=/abs/path/gulls_runs/<run_name>/input/
GULLS_STARS_DIR=/abs/path/to/external/catalog/root/
```

Recommended generated directory settings:

```text
OUTPUT_DIR=/abs/path/gulls_runs/<run_name>/output/
FINAL_DIR=/abs/path/gulls_runs/<run_name>/output/
OBSERVATORY_DIR=observatories/
WEATHER_PROFILE_DIR=weather/
STARFIELD_DIR=SynthPop_Corrected/
SOURCE_DIR=SynthPop_Corrected/
LENS_DIR=SynthPop_Corrected/
PLANET_DIR=planets/
```

Pre-create `/abs/path/gulls_runs/<run_name>/output/<run_name>/` before launch because Gulls appends `RUN_NAME/`.

For smoke presets, `GULLS_STARS_DIR` may point to `input/` because the assets are intentionally tiny and can be snapshotted. For real survey presets, `GULLS_STARS_DIR` must point to the external catalog root selected by the user or inferred from the local environment.

## Preset Sources

Implement sources as providers. Provider-specific logic should stop at discovery and source-file resolution. The UI and launch pipeline should consume normalized presets only.

### Remote Cache Provider

- Source: `https://github.com/gulls-microlensing/Parameterfiles`.
- Cache root: `~/.cache/gulls-ui/parameterfiles/`.
- Startup behavior:
  - Fetch latest `main` commit SHA.
  - If that SHA is not cached, download the GitHub zip archive and extract to `~/.cache/gulls-ui/parameterfiles/<sha>/`.
  - If network fails, reuse the most recent cached SHA and mark the source as stale/offline.
- Discovery pattern: `<cache>/<sha>/general_input/**/*.prm`.
- Each `.prm` is one preset.

### Smoke Provider

- Active when `gulls-ui` is run from a Gulls clone or can find the repo root.
- Discovery pattern: `<repo>/smoke_test/parameterfiles/*.prm`.
- Asset roots:
  - `<repo>/smoke_test/assets/`
  - `<repo>/smoke_test/parameterfiles/`
- Smoke presets should appear even if the remote cache is unavailable.

### Local Directory Provider

- User adds directories from a UI settings panel.
- Persist config in `~/.config/gulls-ui/sources.json`.
- Discovery pattern: `<configured_dir>/**/*.prm`.
- Each configured source may also define optional lightweight asset roots and heavyweight catalog roots.
- Missing dependencies should be reported in the preset detail panel before launch.

## Catalog Root Policy

Catalogs are not Parameterfiles-style inputs. They are data dependencies that should stay where they are.

Treat these as heavyweight catalog-backed settings:

- `STARFIELD_DIR`
- `SOURCE_DIR`
- `LENS_DIR`
- `STARFIELD_LIST` and the catalog files listed inside it
- `SOURCE_LIST` and the catalog files listed inside it
- `LENS_LIST` and the catalog files listed inside it
- `PLANET_DIR` and files addressed by `PLANET_ROOT`

Rules:

- Do not copy real starfield/source/lens/planet catalog files into per-run `input/`.
- Allow copying smoke-test catalog assets because they are small and self-contained.
- Remote `Parameterfiles` presets should be treated as parameter/control presets unless they also contain tiny catalog fixtures.
- For real presets, resolve `GULLS_STARS_DIR` to an existing external catalog root and keep `STARFIELD_DIR`, `SOURCE_DIR`, and `LENS_DIR` relative to that root.
- The UI must show missing catalog roots before launch and give users a way to configure them.
- Store configured catalog roots in `~/.config/gulls-ui/sources.json` separately from preset directories.
- Record external catalog roots in `provenance.json`; do not claim the run is fully self-contained when it depends on external catalogs.
- Because current Gulls prefixes `PLANET_DIR` with `GULLS_BASE_DIR`, materialize a symlink under `input/` for external planet roots, or block launch with a clear message if symlinks are unavailable. Longer term, consider changing Gulls so `PLANET_DIR` can be rooted like other catalog data.

## Normalized Preset Model

Create a normalized backend representation for every discovered `.prm`:

```json
{
  "id": "remote:abc123:general_input/RGES_Yield_SEP2025/CCS_m30.prm",
  "source_type": "remote-cache",
  "display_name": "RGES Yield SEP2025 / CCS_m30",
  "prm_path": "/abs/path/to/CCS_m30.prm",
  "source_root": "/abs/path/to/provider/root",
  "preset_root": "/abs/path/to/general_input/RGES_Yield_SEP2025",
  "parsed_settings": {
    "RUN_NAME": "RGES_m30",
    "EXECUTABLE": "gulls_croin.x"
  },
  "catalog_requirements": {
    "stars_root": {
      "status": "missing",
      "requested_dirs": ["SynthPop_Corrected/"]
    },
    "planet_root": {
      "status": "missing",
      "requested_dir": "planet/m+30/"
    }
  },
  "referenced_files": [],
  "missing_dependencies": [],
  "provenance": {
    "repo_url": "https://github.com/gulls-microlensing/Parameterfiles.git",
    "commit": "abc123",
    "relative_prm": "general_input/RGES_Yield_SEP2025/CCS_m30.prm"
  }
}
```

IDs must be stable across refreshes when possible, but remote IDs should include the commit SHA so cached versions remain distinguishable in run provenance.

## Dependency Resolution Policy

During discovery, parse enough files to make the UI informative:

1. Parse `.prm` into key/value settings.
2. Resolve root input files using candidate paths:
   - Existing absolute path from the `.prm`, if any.
   - Path relative to the `.prm` directory.
   - Path relative to the provider source root.
   - Provider-specific asset roots.
   - Basename fallback inside the preset root, only when exactly one matching filename exists.
3. Parse `OBSERVATORY_LIST`.
4. For each observatory-list row, resolve the observatory file from `OBSERVATORY_DIR`.
5. Parse each observatory file for:
   - `WEATHER_PROFILE`
   - `OBSERVATION_SEQUENCE`
   - `DETECTOR`
   - `THROUGHPUT`
   - `ORBIT` when it is a file-like value rather than a numeric mode.
6. Resolve weather files from `WEATHER_PROFILE_DIR`.
7. Resolve observatory subfiles from `OBSERVATORY_DIR`.
8. Resolve `STARFIELD_LIST`, `SOURCE_LIST`, and `LENS_LIST` against candidate catalog roots, then parse listed catalog filenames when files exist locally. Mark these as external catalog dependencies unless the source is a smoke preset.
9. Resolve `PLANET_DIR` and `PLANET_ROOT` only as far as possible without knowing field/instance; mark directory/root status in the UI and treat real planet populations as external catalog dependencies.
10. Resolve `RATES_FILE` if present, but do not block launch on it unless a validator proves the executable needs it.

Do not silently rewrite unresolved paths during discovery. Surface them as missing dependencies with enough context for the user to fix them.

## Materialization Policy

On launch, create a per-run lightweight input snapshot under `gulls_runs/<run_name>/input/`.

Rules:

- Copy required lightweight files for the selected preset: `.prm`, observatory lists, observatory files, weather files, detector files, throughput files, sequences, orbit files, and small rates/configuration files.
- Do not copy real starfield/source/lens/planet catalogs.
- For smoke presets only, copy the tiny catalog fixtures into `input/` and set `GULLS_STARS_DIR` to `input/`.
- For real presets, set `GULLS_STARS_DIR` to the selected external catalog root and leave catalog directory settings relative to that root.
- For external planet populations, create a symlink from `input/planets` to the selected planet root when the generated `.prm` uses `PLANET_DIR=planets/`.
- Preserve Gulls-relative paths inside destination directories.
- If a source preset uses a flat layout but its `.prm` expects `observatories/`, materialize into the layout Gulls expects.
- If a dependency is missing, block launch with a clear error unless the dependency is explicitly optional.
- Write `generated.prm` from the original preset plus UI overrides.
- Rewrite lightweight path keys to canonical launch values relative to `GULLS_BASE_DIR`.
- Rewrite starfield/source/lens path keys relative to the external `GULLS_STARS_DIR` for real presets.
- Keep list entries and observatory subfile values relative to their owning directories.
- Write `provenance.json` with source, commit, original settings, final settings, overrides, copied files, external catalog roots, symlinks, and missing optional files.

Example `generated.prm` path section:

```text
RUN_NAME=my_custom_run
OUTPUT_DIR=/abs/path/gulls_runs/my_custom_run/output/
FINAL_DIR=/abs/path/gulls_runs/my_custom_run/output/
EXECUTABLE=gulls_general.x

OBSERVATORY_DIR=observatories/
OBSERVATORY_LIST=smoke.list
WEATHER_PROFILE_DIR=weather/

STARFIELD_DIR=starfields/
STARFIELD_LIST=smoke.starfields
SOURCE_DIR=sources/
SOURCE_LIST=smoke.sources
LENS_DIR=lenses/
LENS_LIST=smoke.lenses
PLANET_DIR=planets/
PLANET_ROOT=smoke_general_pspl.planets.
```

Example real-catalog `generated.prm` path section:

```text
RUN_NAME=my_custom_roman_run
OUTPUT_DIR=/abs/path/gulls_runs/my_custom_roman_run/output/
FINAL_DIR=/abs/path/gulls_runs/my_custom_roman_run/output/
EXECUTABLE=gulls_croin.x

OBSERVATORY_DIR=observatories/
OBSERVATORY_LIST=CCS_yield.list
WEATHER_PROFILE_DIR=weather/

STARFIELD_DIR=SynthPop_Corrected/
STARFIELD_LIST=gulls_surot2d_H2024.starfields
SOURCE_DIR=SynthPop_Corrected/
SOURCE_LIST=gulls_surot2d_H2024.sources
LENS_DIR=SynthPop_Corrected/
LENS_LIST=gulls_surot2d_H2024.lenses
PLANET_DIR=planets/
PLANET_ROOT=m+30.planets.
```

For the real-catalog example, launch with:

```text
GULLS_BASE_DIR=/abs/path/gulls_runs/my_custom_roman_run/input/
GULLS_STARS_DIR=/abs/path/to/catalog/root/
```

and make `input/planets` a symlink to the external planet directory if `PLANET_ROOT` points at a large planet population.

## API Design

Add these endpoints in `gulls_pipeline/server.py`:

- `GET /api/preset-sources`
  - Returns remote/smoke/local source status.
- `POST /api/preset-sources/local`
  - Adds a local directory source to `~/.config/gulls-ui/sources.json`.
- `DELETE /api/preset-sources/local/{source_id}`
  - Removes a configured local source.
- `POST /api/catalog-roots`
  - Adds or updates an external catalog root used for starfield/source/lens or planet data.
- `DELETE /api/catalog-roots/{root_id}`
  - Removes a configured external catalog root.
- `POST /api/presets/refresh`
  - Refreshes the remote cache and rescans all sources.
- `GET /api/presets`
  - Returns normalized preset summaries.
- `GET /api/presets/{preset_id}`
  - Returns parsed settings and dependency status for one preset.
- `POST /api/launch`
  - Accepts selected `preset_id`, UI overrides, output configuration, and launch options.
  - Materializes input snapshot, writes `generated.prm`, launches executable, and returns run metadata.

The existing `/api/launch` payload should be migrated rather than expanded indefinitely. Preserve backward compatibility temporarily if useful for smoke development.

## UI Design

Modify `gulls_pipeline/web/index.html` to make presets first-class:

- Add a preset source status strip:
  - Remote cache status: synced, stale/offline, refreshing, error.
  - Smoke source status: available/unavailable.
  - Local source count and add/remove controls.
- Add a preset selector at the top of the Simulation Runner.
- When a preset is selected:
  - Load actual `.prm` values into the form.
  - Show source type, commit/path, executable, and dependency status.
  - Flag missing dependencies before launch.
- Add catalog-root controls for real presets:
  - Select an external stars root for `STARFIELD_DIR`, `SOURCE_DIR`, and `LENS_DIR`.
  - Select an external planet root when `PLANET_DIR` points at large planet populations.
  - Clearly label these as external data dependencies, not copied run inputs.
- Keep advanced form editing, but distinguish:
  - Values loaded from preset.
  - Values changed by the user.
  - Values forced by launch materialization, such as `OUTPUT_DIR` and input directories.
- On launch, send `preset_id` plus overrides, not the entire form as if it were source of truth.

## Implementation Tasks

### Task 1: Add PRM Parsing Utilities

**Files:**

- Create: `gulls_pipeline/presets/parser.py`
- Test: `tests/test_preset_parser.py`

Implement a parser that:

- Reads `KEY=VALUE`.
- Strips comments beginning with `#`.
- Trims whitespace around keys and values.
- Preserves original key order.
- Handles duplicate keys by using the last value and recording duplicates.
- Can serialize settings back to `.prm` text with selected overrides.

Verification:

```bash
python -m pytest tests/test_preset_parser.py -v
```

### Task 2: Add Source Configuration

**Files:**

- Create: `gulls_pipeline/presets/config.py`
- Test: `tests/test_preset_source_config.py`

Implement:

- `get_config_path()` returning `~/.config/gulls-ui/sources.json` by default.
- Read/write helpers for local preset directories.
- Read/write helpers for external catalog roots, including labels for stars roots and planet roots.
- Validation that configured paths exist and are directories.
- Environment override for tests, for example `GULLS_UI_CONFIG_DIR`.

Verification:

```bash
python -m pytest tests/test_preset_source_config.py -v
```

### Task 3: Add Remote Cache Provider

**Files:**

- Create: `gulls_pipeline/presets/remote_cache.py`
- Test: `tests/test_remote_cache.py`

Implement:

- Fetch latest commit SHA for `gulls-microlensing/Parameterfiles` `main`.
- Download GitHub zip archive for uncached SHA.
- Extract safely under `~/.cache/gulls-ui/parameterfiles/<sha>/`.
- Reuse latest cached SHA when offline.
- Avoid requiring the `git` executable.
- Environment override for tests, for example `GULLS_UI_CACHE_DIR`.

Verification:

```bash
python -m pytest tests/test_remote_cache.py -v
```

### Task 4: Add Preset Provider Scanner

**Files:**

- Create: `gulls_pipeline/presets/sources.py`
- Test: `tests/test_preset_sources.py`

Implement:

- `RemoteCacheProvider`.
- `SmokeProvider`.
- `LocalDirectoryProvider`.
- Common `PresetRecord` dataclass.
- Discovery from `.prm` files.
- Stable display names from relative paths.
- Source status reporting.

Verification:

```bash
python -m pytest tests/test_preset_sources.py -v
```

### Task 5: Add Dependency Resolver

**Files:**

- Create: `gulls_pipeline/presets/dependencies.py`
- Test: `tests/test_preset_dependencies.py`

Implement:

- Root path resolution policy from this document.
- Observatory-list parsing.
- Observatory-file parsing for subfiles.
- Weather, sequence, detector, throughput, orbit dependency tracking.
- Starfield/source/lens list dependency tracking against external `GULLS_STARS_DIR` candidates.
- Planet directory/root status tracking, including whether launch will use a symlink under `input/`.
- Missing dependency reporting with key, requested value, and searched paths.

Verification:

```bash
python -m pytest tests/test_preset_dependencies.py -v
```

### Task 6: Add Materializer

**Files:**

- Create: `gulls_pipeline/presets/materializer.py`
- Test: `tests/test_preset_materializer.py`

Implement:

- Create run directory layout.
- Copy resolved lightweight dependencies into `input/`.
- Keep real starfield/source/lens catalogs in their external catalog root.
- Copy smoke catalog fixtures only when the selected preset source is smoke.
- Create symlinks for external planet roots when needed by current `PLANET_DIR` semantics.
- Preserve expected Gulls-relative subpaths.
- Write `generated.prm`.
- Pre-create `output/<RUN_NAME>/`.
- Write `provenance.json`.
- Return launch environment containing `GULLS_BASE_DIR=input/` and `GULLS_STARS_DIR=<external catalog root>` for real presets, or both pointing to `input/` for smoke presets.

Verification:

```bash
python -m pytest tests/test_preset_materializer.py -v
```

### Task 7: Integrate Launch Path

**Files:**

- Modify: `gulls_pipeline/cli.py`
- Modify: `gulls_pipeline/server.py`
- Test: `tests/test_launch_payload.py`

Implement:

- New launch payload with `preset_id` and `overrides`.
- Server-side lookup of selected preset.
- Materialization before process start.
- Run executable with generated `.prm`.
- Set `GULLS_BASE_DIR` to run `input/`.
- Set `GULLS_STARS_DIR` to the configured external catalog root for real presets, or run `input/` for smoke presets.
- Preserve existing run log behavior under `logs/gulls_run.log`.

Verification:

```bash
python -m pytest tests/test_launch_payload.py -v
```

### Task 8: Add Preset APIs

**Files:**

- Modify: `gulls_pipeline/server.py`
- Test: `tests/test_preset_api.py`

Implement API endpoints listed above. Keep remote refresh non-blocking if it becomes slow, but initial startup should trigger a sync attempt so users see current presets without manual setup.

Verification:

```bash
python -m pytest tests/test_preset_api.py -v
```

### Task 9: Update Vue UI

**Files:**

- Modify: `gulls_pipeline/web/index.html`

Implement:

- Preset source status.
- Preset dropdown populated from `/api/presets`.
- Preset detail/dependency panel.
- Catalog-root selector and missing-catalog warnings.
- Form hydration from `/api/presets/{preset_id}`.
- Override tracking.
- Launch request using `preset_id` plus overrides.

Manual verification:

```bash
gulls-ui --headless
```

Open `http://localhost:8080`, confirm remote/smoke/local presets load, select a preset, edit a field, and inspect the live backend payload.

### Task 10: End-to-End Smoke Verification

**Files:**

- No new files unless test fixtures are needed.

Run:

```bash
cmake -S . -B build
cmake --build build
gulls-ui --headless
```

Then launch a smoke preset from the UI and verify:

- `gulls_runs/<run_name>/input/generated.prm` exists.
- `provenance.json` records source and overrides.
- `provenance.json` records external catalog roots for any non-smoke run.
- `output/<run_name>/` was pre-created.
- `logs/gulls_run.log` exists.
- The log shows `GULLS_BASE_DIR` set to the run `input/` directory.
- Smoke runs can set `GULLS_STARS_DIR` to `input/`; real runs set `GULLS_STARS_DIR` to the selected external catalog root.
- Real runs do not copy heavyweight starfield/source/lens/planet catalogs into `input/`.
- The run does not depend on lightweight files inside the remote cache after launch.

## Migration Notes

- Keep the current hardcoded smoke form only as an emergency fallback while the preset API is being built.
- After preset hydration works, remove hardcoded default values that duplicate smoke `.prm` files.
- Do not expose `GULLS_BASE_DIR` to novice users as a normal form field. It is launch infrastructure.
- Use `Gulls` in user-facing text. Use `GULLS_*` only for environment variable names and existing code identifiers.

## Open Decisions

- Default run root: use `gulls_runs/` at repo root, `~/gulls_runs/`, or user-configurable. The implementation should support a config value even if the first UI defaults to repo-local `gulls_runs/`.
- Remote cache refresh timing: block initial page load until first scan completes, or serve immediately with a loading state. The UX should clearly state when presets are stale/offline.
- Planet-root handling: symlink external planet roots under `input/` for now, or change Gulls path resolution so `PLANET_DIR` can use a dedicated data root like star/source/lens catalogs.
- `RATES_FILE`: source parsing marks it as a dependency, but current C++ path search did not find runtime use. Treat missing rates as a warning until confirmed otherwise.
