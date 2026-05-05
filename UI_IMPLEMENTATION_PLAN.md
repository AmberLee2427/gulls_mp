# GULLS Python Pipeline & Web UI: Implementation Plan

## Objective
Modernize the user experience of `gulls_general` by introducing a Python-based wrapper, an interactive local/cloud-ready Web UI, and making the resulting package pip-installable while respecting the licensing of the underlying C++ source code (avoiding the distribution of built binaries containing proprietary Numerical Recipes routines on PyPI).

## 1. Distribution Strategy (Licensing & Installation)
Because `gulls_general` utilizes Numerical Recipes (which cannot be legally redistributed in binary or open-source payload form without a commercial license), we cannot put compiled wheels on PyPI.

**Solution: Source Distribution with Runtime Check**
1. **GitHub Repository:** The `gulls_general` C++ source remains exactly where it is.
2. **`gulls-runner` Python Package:** A lightweight Python wrapper packaged alongside the UI in a `frontend/` directory.
3. **Pip Installation:** Users run `pip install .` inside the repository.
   - We will write a `setup.py` / `pyproject.toml` that installs the Python dependencies (FastAPI, Uvicorn) and creates the CLI entry points (`gulls-ui` and `gulls-runner`).
   - The Python code will assume the C++ executables live in `./bin/` relative to the workspace root, or require a `GULLS_BIN_DIR` environment variable to locate them.
4. **Compile Prompts:** If `gulls-runner` is executed but fails to find `gulls_general.x`, it will gracefully fail with instructions: *"Executables not found. Please run `./configure.sh` and `make` in the repository root to compile."*

## 2. Directory Structure

```text
gulls_general/
├── bin/                  # User's compiled outputs (ignored by git ideally)
├── src/                  # Existing C++ source
├── gulls_pipeline/       # NEW: The Python package
│   ├── __init__.py
│   ├── cli.py            # Entry point for 'gulls-runner' (executes JSON)
│   ├── server.py         # Entry point for 'gulls-ui' (runs FastAPI)
│   ├── generator.py      # Logic to parse JSON and write .prm/.observatory files
│   └── web/              # Static assets for the Web UI
│       └── index.html    # The Tailwind/Vue UI mockup
├── setup.py              # NEW: Makes gulls_pipeline pip installable
```

## 3. The Backend Architecture (FastAPI)
The backend (`server.py`) acts as the bridge between the browser UI and the local file system.

**Endpoints Needed:**
- `GET /`: Serves the `index.html`.
- `GET /api/system_state`: Returns the `cwd`, user, and checks if `./bin/gulls_general.x` exists.
- `GET /api/registries`: Scans a designated assets folder (e.g., `smoke_test/assets/` or a user-defined path) to populate UI dropdowns dynamically (listing available `.detector` files, `.weather` files, population databases).
- `POST /api/launch`: Receives the JSON payload from the UI, hands it to `generator.py` to write the parameters, and uses `subprocess.Popen` to launch the C++ binary.
- `GET /api/stream`: A WebSocket or polling endpoint to stream `stdout` from the running subprocess back to the UI.

## 4. The GUI Export Feature (Headless Support)
To support the "Split" workflow (run UI locally, execute on Unity):
- The Web UI will feature an **"Export JSON"** button. This leverages standard browser APIs to download the JSON blob directly to the user's laptop without needing server interaction.
- The user can then `scp` this file to their compute cluster and run `gulls-runner custom_run.json`.

## 5. Development Phases

**Phase 1: Package Scaffolding**
- Create `setup.py` and ensure `pip install -e .` correctly creates the `gulls-ui` command.
- Set up a barebones FastAPI server serving the current `mockup_ui.html`.

**Phase 2: Backend API & File System Bridging**
- Write the `/api/registries` logic to read the local disk and feed it into the Vue.js UI, turning the static mockups into dynamic dropdowns based on local files.

**Phase 3: The Pipeline Logic (generator.py)**
- Build the Python class that takes the UI's JSON schema and robustly templates out the `.prm`, `.list`, and `.observatory` text files.

**Phase 4: Execution & Feedback**
- Wire up the `subprocess` call.
- Add live terminal streaming back to the UI to monitor the simulation progress.