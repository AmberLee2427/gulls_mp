import argparse

from gulls_pipeline import launcher


def generate_files_and_run(config_path: str):
    config = launcher.load_config(config_path)
    scheduler = str(config.get("scheduler") or "local").lower()
    if scheduler == "slurm":
        result = launcher.generate_slurm_bundle(config)
    else:
        result = launcher.run_local(config)
    print(result.message)


def main():
    parser = argparse.ArgumentParser(description="Consume a Gulls UI JSON config and run or stage it")
    parser.add_argument("config_json", help="Path to the JSON config file")
    args = parser.parse_args()
    generate_files_and_run(args.config_json)


if __name__ == "__main__":
    main()
