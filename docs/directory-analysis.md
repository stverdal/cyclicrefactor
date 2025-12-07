# Directory Analysis Mode

The Cycle Refactoring Pipeline supports two operation modes:

1. **Standard Mode**: Process a single cycle from a JSON file
2. **Directory Analysis Mode**: Automatically discover and process cycles from a project

This document explains how to use Directory Analysis Mode.

## Overview

Directory Analysis Mode scans a TypeScript/JavaScript project, builds a dependency graph,
detects cyclic dependencies, and optionally runs the refactoring pipeline on each cycle.

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Project Dir    │ ──► │ DependencyAnalyzer│ ──► │  DependencyGraph  │
│  (source code)  │     │  (madge/regex)    │     │  (nodes + edges)  │
└─────────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   CycleSpec[]   │ ◄── │  CycleDetector   │ ◄── │ Cycle Detection   │
│ (for pipeline)  │     │  (Tarjan + Johnson)│     │ (find SCCs)       │
└────────┬────────┘     └──────────────────┘     └───────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Existing Refactoring Pipeline                     │
│   Describer ──► Refactor ──► Validator ──► Explainer                │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### For Best Results: Install madge

Directory Analysis uses [madge](https://github.com/pahen/madge) for accurate
TypeScript/JavaScript dependency extraction.

```bash
# Install globally
npm install -g madge

# Or install as project dependency
npm install --save-dev madge
```

### Fallback Mode

If madge is not available, the analyzer falls back to regex-based parsing.
This works but may be less accurate with:
- Path aliases (tsconfig paths)
- Complex re-exports
- Barrel files

## Usage

### Analyze Only (Discovery)

Scan a project to see what cycles exist without running refactoring:

```bash
python cli.py --analyze /path/to/project --analyze-only
```

Output:
```
DETECTED CYCLES
============================================================

Total cycles found: 5
  - Critical: 1
  - Major: 3
  - Minor: 1

Top 10 cycles by severity:
  1. [CRITICAL] src/services/UserService.ts -> src/services/AuthService.ts -> ...
  2. [MAJOR] src/models/Order.ts -> src/models/Customer.ts -> ...
  ...
```

### Save Cycles for Later Processing

```bash
python cli.py --analyze /path/to/project --analyze-only --output cycles.json
```

This saves the discovered cycles in the standard CycleSpec JSON format,
which can be processed later with standard mode:

```bash
python cli.py cycles.json --cycle-id cycle-0
```

### Full Analysis + Refactoring

Discover cycles AND run the refactoring pipeline on each:

```bash
python cli.py --analyze /path/to/project
```

This will:
1. Scan the project
2. Detect all cycles
3. Prioritize them (critical first)
4. Run the refactoring pipeline on each cycle
5. Report overall results

### Custom Options

```bash
# Only analyze TypeScript files
python cli.py --analyze /path/to/project --extensions ts,tsx

# Custom exclusions
python cli.py --analyze /path/to/project --exclude node_modules,dist,coverage

# Limit number of cycles
python cli.py --analyze /path/to/project --max-cycles 10

# Change prioritization
python cli.py --analyze /path/to/project --priority size_first
```

## CLI Reference

### Mode Selection

| Flag | Description |
|------|-------------|
| `input_json` | Process a cycle JSON file (standard mode) |
| `--analyze DIR` | Analyze a project directory |

### Analysis Options

| Flag | Description | Default |
|------|-------------|---------|
| `--analyze-only` | Only detect cycles, don't refactor | Off |
| `--extensions EXT` | File extensions (comma-separated) | ts,tsx,js,jsx |
| `--exclude PATTERNS` | Exclude patterns (comma-separated) | node_modules,dist,... |
| `--max-cycles N` | Maximum cycles to process | 50 |
| `--priority STRATEGY` | Prioritization strategy | severity_first |
| `--output FILE` | Save cycle specs to file | None |

### Priority Strategies

| Strategy | Description |
|----------|-------------|
| `severity_first` | Critical → Major → Minor (default) |
| `size_first` | Smaller cycles first (easier to fix) |
| `impact_first` | Most connected cycles first |

## Cycle Severity Classification

Cycles are classified by severity to help prioritize fixes:

| Severity | Criteria | Impact |
|----------|----------|--------|
| **Critical** | 5+ nodes, complex interdependencies | Architecture issue, blocks testing |
| **Major** | 3-4 nodes, clear cycle | Should be fixed |
| **Minor** | 2 nodes (mutual imports) | May be intentional |

## Example Workflows

### Workflow 1: Assess Technical Debt

```bash
# See what cycles exist in your codebase
python cli.py --analyze ./src --analyze-only --output cycles.json

# Review cycles.json to understand the scope
# Decide which cycles to prioritize

# Process critical cycles first
python cli.py cycles.json --cycle-id critical-cycle-1
```

### Workflow 2: Automated Cleanup

```bash
# Run full analysis with dry-run first (edit config.yml: dry_run: true)
python cli.py --analyze ./src

# Review proposed changes in artifacts directory
# If satisfied, disable dry-run and run again

# Or process cycles one at a time
python cli.py --analyze ./src --analyze-only --output cycles.json
for id in $(jq -r '.[].id' cycles.json | head -5); do
    python cli.py cycles.json --cycle-id "$id"
done
```

### Workflow 3: CI Integration

```yaml
# .github/workflows/detect-cycles.yml
name: Detect Cycles
on: [push]
jobs:
  detect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - uses: actions/setup-python@v4
      
      - run: npm install -g madge
      - run: pip install -r requirements.txt
      
      - name: Detect cycles
        run: python cli.py --analyze ./src --analyze-only --output cycles.json
      
      - name: Check for critical cycles
        run: |
          CRITICAL=$(python -c "import json; print(len([c for c in json.load(open('cycles.json')) if c.get('severity')=='critical']))")
          if [ "$CRITICAL" -gt 0 ]; then
            echo "::error::Found $CRITICAL critical cycles!"
            exit 1
          fi
```

## Programmatic Usage

You can use the analysis API directly in Python:

```python
from orchestrator import Orchestrator
from config import load_config

# Initialize
cfg = load_config("config.yml")
orch = Orchestrator(config=cfg)

# Analyze only
result = orch.analyze_directory(
    "/path/to/project",
    extensions=["ts", "tsx"],
    max_cycles=20,
)

if result["success"]:
    print(f"Found {len(result['cycles'])} cycles")
    
    for cycle in result["cycles"]:
        print(f"  - {cycle.severity}: {' -> '.join(cycle.nodes)}")
    
    # Process a specific cycle
    if result["cycle_specs"]:
        orch.run_pipeline(result["cycle_specs"][0])

# Or run full analysis
results = orch.run_full_analysis(
    "/path/to/project",
    priority="severity_first",
)

print(f"Approved: {results['approved_count']}/{results['cycles_processed']}")
```

## Comparison of Modes

| Feature | Standard Mode | Directory Analysis |
|---------|--------------|-------------------|
| Input | JSON cycle spec | Project directory |
| Discovery | Manual | Automatic |
| Multi-cycle | One at a time | Batch processing |
| Prerequisites | None | Node.js + madge (optional) |
| Use case | Known cycles | Exploration/audit |

## Troubleshooting

### "madge not found"

Install madge globally or use npx:
```bash
npm install -g madge
# or
npx madge --help
```

### "No cycles found"

1. Check that extensions match your project (e.g., `.ts` vs `.tsx`)
2. Check exclude patterns aren't too aggressive
3. Try with regex fallback: set `prefer_madge: false` in config

### "Too many cycles"

Use `--max-cycles` to limit:
```bash
python cli.py --analyze ./src --max-cycles 10
```

### Path Aliases Not Resolved

If using TypeScript path aliases:
1. Ensure `tsconfig.json` is in the project root
2. Use madge (regex fallback doesn't fully support aliases)
3. Try specifying tsconfig: `--tsconfig ./tsconfig.json`
