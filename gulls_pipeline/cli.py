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
    
    # OUTPUT AND FINAL DIRECTORIES
    out_dir = Path(config.get("output_dir", f"runs/{run_name}/output/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the root parameter file
    prm_path = out_dir.parent / f"{run_name}.prm"
    obs_dir = out_dir.parent / "observatories"
    obs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Write the observatories and their list
    obs_list_path = obs_dir / "run.list"
    obs_filenames = []
    
    for obs in config["observatories"]:
        obs_filename = f"{obs['name']}.observatory"
        obs_filenames.append(obs_filename)
        with open(obs_dir / obs_filename, 'w') as f:
            f.write(f"NAME {obs['name']}\n")
            for k, v in obs["settings"].items():
                f.write(f"{k} {v}\n")
                
    with open(obs_list_path, 'w') as f:
        f.write("\n".join(obs_filenames) + "\n")
        
    # 2. Write the PRM file
    with open(prm_path, 'w') as f:
        # Mandatory routing
        f.write(f"RUN_NAME={run_name}\n")
        f.write(f"OUTPUT_DIR={out_dir}/\n")
        f.write(f"FINAL_DIR={out_dir}/\n")
        f.write(f"EXECUTABLE={config['executable']}\n")
        
        f.write(f"\n# OBSERVATORIES\n")
        f.write(f"OBSERVATORY_DIR={obs_dir}/\n")
        f.write(f"OBSERVATORY_LIST=run.list\n")
        
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
        
    print(f"Running: {exec_path} -i {prm_path}")
    
    # Write stdout and stderr to a log file inside the parent wrapper directory
    log_path = out_dir.parent / "gulls_run.log"
    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen(
            [str(exec_path), "-i", str(prm_path)],
            cwd=Path.cwd(),
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