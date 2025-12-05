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
    
    logger.info("="*60)
    logger.info("CYCLE REFACTORING PIPELINE")
    logger.info("="*60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Input file: {args.input_json}")
    if args.cycle_id:
        logger.info(f"Target cycle ID: {args.cycle_id}")
    logger.info(f"LLM provider: {cfg.llm.provider if cfg.llm else 'none'}")
    logger.info(f"LLM model: {cfg.llm.model if cfg.llm else 'none'}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize input into canonical CycleSpec model
    cycle_spec = normalize_input(raw, cycle_id=args.cycle_id, read_files=args.read_files)
    logger.info(f"Loaded cycle: id={cycle_spec.id}, files={len(cycle_spec.files)}, nodes={len(cycle_spec.graph.nodes)}")

    orch = Orchestrator(config=cfg)
    # CycleSpec model doesn't have a prompt field; pass None or extract from metadata
    prompt = cycle_spec.metadata.get("prompt") if cycle_spec.metadata else None
    results = orch.run_pipeline(cycle_spec, prompt=prompt)

    # Print summary
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    status = results.get('status', 'unknown')
    iterations = results.get('iterations', 0)
    
    if status == 'approved':
        logger.info(f"✓ Status: APPROVED after {iterations} iteration(s)")
    elif status == 'max_iterations_reached':
        logger.warning(f"⚠ Status: MAX ITERATIONS REACHED ({iterations})")
    elif status == 'error':
        logger.error(f"✗ Status: ERROR - {results.get('error', 'unknown error')}")
    else:
        logger.info(f"Status: {status} after {iterations} iteration(s)")
    
    # Show artifacts location
    artifact_dir = f"{cfg.io.artifacts_dir}/{cycle_spec.id}"
    logger.info(f"Artifacts saved to: {artifact_dir}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Description summary
    desc = results.get("description", {})
    if isinstance(desc, dict) and desc.get("text"):
        print(f"\nDescription:\n{desc['text'][:500]}{'...' if len(desc.get('text', '')) > 500 else ''}")
    
    # Proposal summary
    proposal = results.get("proposal", {})
    if isinstance(proposal, dict) and proposal.get("patches"):
        patches = proposal["patches"]
        changed = [p for p in patches if p.get("diff")]
        print(f"\nProposal: {len(changed)}/{len(patches)} files modified")
        for p in changed[:5]:
            print(f"  - {p.get('path')}")
        if len(changed) > 5:
            print(f"  ... and {len(changed) - 5} more")


if __name__ == "__main__":
    main()
