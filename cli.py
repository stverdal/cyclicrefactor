import argparse
import argparse
import json
from orchestrator import Orchestrator
from utils.logging import setup_logger
from config import load_config

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to cycle JSON file")
    parser.add_argument("--config", help="Path to YAML config file", default="config.yml")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        cycle_spec = json.load(f)

    cfg = load_config(args.config)

    orch = Orchestrator(config=cfg)
    results = orch.run_pipeline(cycle_spec, prompt=cycle_spec.get("prompt"))

    print("Description:\n", results.get("description"))
    print("Proposal:\n", results.get("proposal"))


if __name__ == "__main__":
    main()
