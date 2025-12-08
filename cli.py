import argparse
import json
import sys
from orchestrator import Orchestrator
from utils.logging import setup_logger, configure_from_config
from config import load_config
from utils.input_normalizer import normalize_input


def run_pipeline_mode(args, cfg, logger):
    """Run the standard pipeline mode with a cycle JSON file."""
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
    
    return results


def run_analyze_mode(args, cfg, logger):
    """Run directory analysis mode to discover and refactor cycles."""
    logger.info("="*60)
    logger.info("DIRECTORY ANALYSIS MODE")
    logger.info("="*60)
    logger.info(f"Project directory: {args.analyze}")
    logger.info(f"Extensions: {args.extensions or 'auto-detect'}")
    logger.info(f"Max cycles: {args.max_cycles}")
    
    orch = Orchestrator(config=cfg)
    
    # Parse extensions if provided
    extensions = None
    if args.extensions:
        extensions = [e.strip().lstrip('.') for e in args.extensions.split(',')]
    
    # Parse exclude patterns
    exclude_patterns = None
    if args.exclude:
        exclude_patterns = [p.strip() for p in args.exclude.split(',')]
    
    if args.analyze_only:
        # Just analyze, don't run refactoring
        logger.info("Mode: Analyze only (--analyze-only)")
        results = orch.analyze_directory(
            args.analyze,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            max_cycles=args.max_cycles,
        )
        
        if not results["success"]:
            logger.error(f"Analysis failed: {results.get('error')}")
            return results
        
        # Print cycle summary
        print("\n" + "="*60)
        print("DETECTED CYCLES")
        print("="*60)
        
        cycles = results.get("cycles", [])
        summary = results.get("summary", {})
        
        print(f"\nTotal cycles found: {len(cycles)}")
        print(f"  - Critical: {summary.get('critical', 0)}")
        print(f"  - Major: {summary.get('major', 0)}")
        print(f"  - Minor: {summary.get('minor', 0)}")
        
        if cycles:
            print("\nTop 10 cycles by severity:")
            for i, cycle in enumerate(cycles[:10]):
                nodes_str = " -> ".join(cycle.nodes[:4])
                if len(cycle.nodes) > 4:
                    nodes_str += f" -> ... ({len(cycle.nodes)} total)"
                print(f"  {i+1}. [{cycle.severity.upper()}] {nodes_str}")
        
        # Save cycle specs to file if requested
        if args.output:
            specs = results.get("cycle_specs", [])
            output_data = [s.model_dump() for s in specs]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved {len(specs)} cycle specs to: {args.output}")
        
        return results
    else:
        # Full analysis + refactoring
        logger.info("Mode: Full analysis and refactoring")
        results = orch.run_full_analysis(
            args.analyze,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            max_cycles=args.max_cycles,
            priority=args.priority,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FULL ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nCycles found: {results.get('cycles_found', 0)}")
        print(f"Cycles processed: {results.get('cycles_processed', 0)}")
        print(f"Approved: {results.get('approved_count', 0)}")
        print(f"Failed: {results.get('failed_count', 0)}")
        print(f"Total time: {results.get('total_time_seconds', 0):.1f}s")
        
        # Show per-cycle results
        pipeline_results = results.get("pipeline_results", [])
        if pipeline_results:
            print("\nPer-cycle results:")
            for pr in pipeline_results[:20]:
                status_icon = "✓" if pr.get("status") == "approved" else "✗"
                print(f"  {status_icon} {pr.get('cycle_id')}: {pr.get('status')}")
            if len(pipeline_results) > 20:
                print(f"  ... and {len(pipeline_results) - 20} more")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Cycle Refactoring Pipeline - Automatically break circular dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode: Process a single cycle from JSON file
  python cli.py cycle.json
  
  # Analyze a TypeScript project for cycles (discovery only)
  python cli.py --analyze /path/to/project --analyze-only
  
  # Analyze and automatically refactor all cycles
  python cli.py --analyze /path/to/project
  
  # Analyze with custom options
  python cli.py --analyze /path/to/project --extensions ts,tsx --max-cycles 10
  
  # Save discovered cycles to file for later processing
  python cli.py --analyze /path/to/project --analyze-only --output cycles.json
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "input_json", 
        nargs='?',
        help="Path to cycle JSON file (standard pipeline mode)"
    )
    mode_group.add_argument(
        "--analyze", 
        metavar="DIR",
        help="Analyze a project directory for cycles (directory analysis mode)"
    )
    
    # Standard pipeline options
    parser.add_argument("--config", help="Path to YAML config file", default="config.yml")
    parser.add_argument("--cycle-id", help="ID of the cycle to select from input JSON (if multiple)")
    parser.add_argument("--no-read-files", dest="read_files", action="store_false", 
                        help="Do not attempt to read file contents from paths in the JSON")
    parser.add_argument("--log-level", help="Override log level (DEBUG, INFO, WARNING, ERROR)", default=None)
    
    # Directory analysis options
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze and detect cycles, don't run refactoring")
    parser.add_argument("--extensions", 
                        help="Comma-separated file extensions to analyze (e.g., ts,tsx,js)")
    parser.add_argument("--exclude", 
                        help="Comma-separated patterns to exclude (e.g., node_modules,dist)")
    parser.add_argument("--max-cycles", type=int, default=50,
                        help="Maximum number of cycles to detect (default: 50)")
    parser.add_argument("--priority", choices=["severity_first", "size_first", "impact_first"],
                        default="severity_first",
                        help="Cycle processing priority (default: severity_first)")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Save discovered cycle specs to JSON file (with --analyze-only)")
    
    args = parser.parse_args()

    # Validate arguments
    if args.input_json is None and args.analyze is None:
        parser.error("Either input_json or --analyze is required")

    # Load config first
    cfg = load_config(args.config)

    # Configure logging from config (or override with CLI arg)
    # Convert Pydantic model to dict for configure_from_config
    log_config = cfg.logging.model_dump() if cfg.logging else {}
    if args.log_level:
        log_config["level"] = args.log_level
    logger = configure_from_config(log_config)
    
    logger.info("="*60)
    logger.info("CYCLE REFACTORING PIPELINE")
    logger.info("="*60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"LLM provider: {cfg.llm.provider if cfg.llm else 'none'}")
    logger.info(f"LLM model: {cfg.llm.model if cfg.llm else 'none'}")

    # Run appropriate mode
    if args.analyze:
        run_analyze_mode(args, cfg, logger)
    else:
        logger.info(f"Input file: {args.input_json}")
        if args.cycle_id:
            logger.info(f"Target cycle ID: {args.cycle_id}")
        run_pipeline_mode(args, cfg, logger)


if __name__ == "__main__":
    main()
