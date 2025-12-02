import argparse
import json
from orchestrator import Orchestrator
from utils.logging import setup_logger
from config import load_config
from utils.input_normalizer import normalize_input

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to cycle JSON file")
    parser.add_argument("--config", help="Path to YAML config file", default="config.yml")
    parser.add_argument("--cycle-id", help="ID of the cycle to select from input JSON (if multiple)")
    parser.add_argument("--no-read-files", dest="read_files", action="store_false", help="Do not attempt to read file contents from paths in the JSON")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize input into canonical cycle_spec
    cycle_spec = normalize_input(raw, cycle_id=args.cycle_id, read_files=args.read_files)

    cfg = load_config(args.config)

    orch = Orchestrator(config=cfg)
    results = orch.run_pipeline(cycle_spec, prompt=cycle_spec.get("prompt"))

    print("Description:\n", results.get("description"))
    print("Proposal:\n", results.get("proposal"))


if __name__ == "__main__":
    main()
