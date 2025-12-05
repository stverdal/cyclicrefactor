import argparse
import json
from orchestrator import Orchestrator
from utils.logging import setup_logger, configure_from_config
from config import load_config
from utils.input_normalizer import normalize_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to cycle JSON file")
    parser.add_argument("--config", help="Path to YAML config file", default="config.yml")
    parser.add_argument("--cycle-id", help="ID of the cycle to select from input JSON (if multiple)")
    parser.add_argument("--no-read-files", dest="read_files", action="store_false", help="Do not attempt to read file contents from paths in the JSON")
    parser.add_argument("--log-level", help="Override log level (DEBUG, INFO, WARNING, ERROR)", default=None)
    args = parser.parse_args()

    # Load config first
    cfg = load_config(args.config)

    # Configure logging from config (or override with CLI arg)
    log_config = cfg.logging.copy() if cfg.logging else {}
    if args.log_level:
        log_config["level"] = args.log_level
    logger = configure_from_config(log_config)
    logger.info(f"Pipeline starting with config: {args.config}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize input into canonical CycleSpec model
    cycle_spec = normalize_input(raw, cycle_id=args.cycle_id, read_files=args.read_files)
    logger.info(f"Loaded cycle: id={cycle_spec.id}, files={len(cycle_spec.files)}, nodes={len(cycle_spec.graph.nodes)}")

    orch = Orchestrator(config=cfg)
    # CycleSpec model doesn't have a prompt field; pass None or extract from metadata
    prompt = cycle_spec.metadata.get("prompt") if cycle_spec.metadata else None
    results = orch.run_pipeline(cycle_spec, prompt=prompt)

    logger.info(f"Pipeline completed: status={results.get('status')}, iterations={results.get('iterations')}")
    print("Description:\n", results.get("description"))
    print("Proposal:\n", results.get("proposal"))


if __name__ == "__main__":
    main()
