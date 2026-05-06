import sys
import json
import argparse
import subprocess
from pathlib import Path

def generate_files_and_run(config_path: str):
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    run_name = config["run_name"]
    
    # We define the primary run directory
    base_dir = config.get("output_dir", "smoke_test/output/ui/")
    run_dir = Path(base_dir) / run_name
    
    # GULLS internally appends run_name + "/" to the OUTPUT_DIR, so we
    # must construct the parent `out_dir` but actually create the target 
    # folder to prevent the "Unable to open output file" write crash.
    out_dir = run_dir / "output"
    actual_target = out_dir / run_name
    actual_target.mkdir(parents=True, exist_ok=True)
    
    # Create the root parameter file
    prm_path = run_dir / f"{run_name}.prm"
    
    # Write the PRM file
    with open(prm_path, 'w') as f:
        # Mandatory routing
        f.write(f"RUN_NAME={run_name}\n")
        f.write(f"OUTPUT_DIR={out_dir}/\n")
        f.write(f"FINAL_DIR={out_dir}/\n")
        f.write(f"EXECUTABLE={config['executable']}\n")
        
        f.write(f"\n# OBSERVATORIES\n")
        f.write(f"OBSERVATORY_DIR={config.get('observatory_dir', 'smoke_test/assets/observatories/')}\n")
        f.write(f"OBSERVATORY_LIST={config.get('observatory_list', 'smoke.list')}\n")
        
        # Weather directory is needed by the new structure
        f.write(f"\nWEATHER_PROFILE_DIR=smoke_test/assets/weather/\n")
        
        f.write(f"\n# REGISTRIES\n")
        reg = config.get("registry_lookups", {})
        base_cat = reg.get("catalogs", "")
        f.write(f"STARFIELD_DIR={base_cat}starfields/\n")
        f.write(f"STARFIELD_LIST={reg.get('starfields', '')}\n")
        f.write(f"SOURCE_DIR={base_cat}sources/\n")
        f.write(f"SOURCE_LIST={reg.get('sources', '')}\n")
        f.write(f"LENS_DIR={base_cat}lenses/\n")
        f.write(f"LENS_LIST={reg.get('lenses', '')}\n")
        f.write(f"PLANET_DIR={base_cat}planets/\n")
        f.write(f"PLANET_ROOT={reg.get('planet_root', '')}\n")
        f.write(f"RATES_FILE={reg.get('rates', '')}\n")
        
        # Assume colors are 0 for now as it's default
        f.write("SOURCE_COLOURS=0\nLENS_COLOURS=0\n")
        
        f.write(f"\n# DYNAMIC SETTINGS\n")
        for k, v in config["prm_settings"].items():
            f.write(f"{k}={v}\n")

    # 3. Launch
    exec_path = Path("bin") / config["executable"]
    if not exec_path.exists():
        print(f"Error: {exec_path} not found. Please compile gulls.")
        sys.exit(1)
        
    print(f"Running: {exec_path} -i {prm_path} -s 0")
    
    import os
    env = os.environ.copy()
    env["GULLS_BASE_DIR"] = str(Path.cwd()) + "/"
    
    # Write stdout and stderr to a log file inside the wrapper directory
    log_path = run_dir / "gulls_run.log"
    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen(
            [str(exec_path), "-i", str(prm_path), "-s", "0"],
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        proc.wait()
        
    print(f"Done! Check {log_path} for output.")

def main():
    parser = argparse.ArgumentParser(description="Consume a GULLS JSON config and run it")
    parser.add_argument("config_json", help="Path to the JSON config file")
    args = parser.parse_args()
    
    generate_files_and_run(args.config_json)

if __name__ == "__main__":
    main()