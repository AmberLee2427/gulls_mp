import argparse
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import subprocess
import tempfile
import math
from pathlib import Path
try:
    from astropy.time import Time
except ImportError:
    pass

app = FastAPI()

# Get the directory of the current file to locate static assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Serve specific static files if needed, but primarily we just want index.html
@app.get("/")
@app.get("/app")
def read_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.post("/api/launch")
async def launch_gulls(request: Request, background_tasks: BackgroundTasks):
    config = await request.json()
    
    # Store the JSON payload in a temporary file to hand off to runner
    fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="gulls_ui_")
    with os.fdopen(fd, 'w') as f:
        json.dump(config, f)
        
    def run_job(json_path):
        import sys
        # Note: in a true package install `gulls-runner` would be in the PATH.
        # Here we just run the python script directly for safety
        runner_path = Path(__file__).parent / "cli.py"
        subprocess.Popen([sys.executable, str(runner_path), json_path])

    # Launch it natively in the background so HTTP thread isn't blocked
    background_tasks.add_task(run_job, temp_path)
    
    return JSONResponse({"status": "launched", "config": temp_path})

@app.get("/api/runs")
def list_runs():
    """Scan the ui output directory and return list of runs"""
    out_dir = Path(os.getcwd()) / "smoke_test" / "output" / "ui"
    if not out_dir.exists():
        return {"runs": []}
        
    runs = []
    # Any directory in there with a gulls_run.log is considered a run
    for d in out_dir.iterdir():
        if d.is_dir() and (d / "gulls_run.log").exists():
            runs.append({"name": d.name})
    return {"runs": runs}

@app.get("/api/runs/{run_name}/log")
def get_run_log(run_name: str, lines: int = 50):
    """Tail the last N lines of a run's log file"""
    log_path = Path(os.getcwd()) / "smoke_test" / "output" / "ui" / run_name / "gulls_run.log"
    if not log_path.exists():
        return {"error": "Log not found"}
        
    # Read the last N lines efficiently
    try:
        proc = subprocess.run(["tail", "-n", str(lines), str(log_path)], capture_output=True, text=True)
        return {"log": proc.stdout}
    except Exception as e:
        return {"error": str(e)}

# Example API endpoint for Phase 2
@app.get("/api/system_state")
def get_system_state():
    cwd = os.getcwd()
    bin_path = os.path.join(cwd, "bin", "gulls_general.x")
    return {
        "cwd": cwd,
        "gulls_found": os.path.exists(bin_path)
    }

@app.get("/api/time/to_gregorian")
def get_gregorian(bjd: float):
    try:
        # BJD is JD corrected for barycenter tracking, but for daily start times JD is equivalent
        t = Time(bjd, format='jd')
        return {"gregorian": t.isot.split('T')[0]}  # YYYY-MM-DD
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/time/to_bjd")
def get_bjd(gregorian: str):
    try:
        # Parse standard YYYY-MM-DD string
        t = Time(gregorian)
        # Round down to nearest whole day as requested
        bjd = math.floor(t.jd) + 0.0
        return {"bjd": bjd}
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Launch the GULLS Configuration Web UI")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--headless", action="store_true", help="Do not attempt to open a browser window")
    args = parser.parse_args()

    print(f"Starting GULLS UI server on http://localhost:{args.port}")
    if not args.headless:
        import webbrowser
        # Need a slight delay or thread to open after server starts, 
        # but for simplicity, OS will usually catch up.
        # A more robust way is using threading.Timer
        import threading
        def open_browser():
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Timer(1.0, open_browser).start()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")

if __name__ == "__main__":
    main()