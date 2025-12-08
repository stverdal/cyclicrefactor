Scaffold for agentic LangChain pipeline

## Overview

This pipeline analyzes and refactors cyclic dependencies in codebases using a multi-agent architecture:

1. **Describer Agent** - Analyzes the cycle and produces a human-readable description
2. **Refactor Agent** - Proposes code patches to break/reduce the cycle
3. **Validator Agent** - Validates the proposed changes (linting, syntax checks)
4. **Explainer Agent** - Generates documentation for the refactoring

## Project Structure

- `cli.py`: CLI entrypoint. Run with `python cli.py example_cycle.json`.
- `orchestrator.py`: In-process orchestrator that runs the agent pipeline.
- `agents/`: Agent implementations and base class.
- `models/schemas.py`: Pydantic schemas for CycleSpec, CycleDescription, and RefactorProposal.
- `rag/`: RAG (Retrieval-Augmented Generation) module for PDF knowledge base.
- `utils/`: Utility modules (logging, persistence, prompt loading).
- `config.yml`: Configuration file for LLM, pipeline settings, and RAG.

## Quick Setup

### 1. Create Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install and Configure Ollama (Local LLM)

This pipeline is designed for **air-gapped environments** - all inference runs locally.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # Start the server
ollama pull qwen2.5-coder:7b  # Or your preferred model
```

**Windows:**
Download from https://ollama.com/download and install. Then:
```powershell
ollama pull qwen2.5-coder:7b
```

Update `config.yml` to use Ollama:
```yaml
llm:
  provider: ollama
  model: qwen2.5-coder:7b
```

## RAG Module Setup

The RAG (Retrieval-Augmented Generation) module allows agents to retrieve relevant context from PDF documents (architecture guides, refactoring patterns, etc.) during the pipeline run.

### 1. Add PDF Documents

Place your PDF reference documents in the `data/pdf/` folder:
```
data/
└── pdf/
    ├── architecture_patterns.pdf
    ├── refactoring_guide.pdf
    └── ...
```

### 2. Index the Documents

Run the PDF indexer to create embeddings and store them in ChromaDB:

```bash
python -m rag.pdf_indexer --data-dir data/pdf --persist-dir cache/chroma_db
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/pdf` | Directory containing PDF files |
| `--persist-dir` | `cache/chroma_db` | Where to store the vector database |
| `--embedding-provider` | `huggingface` | `huggingface` (local) or `ollama` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Model for generating embeddings |
| `--chunk-size` | `1000` | Characters per text chunk |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--collection` | `architecture_docs` | ChromaDB collection name |

**Example with Ollama embeddings:**
```bash
python -m rag.pdf_indexer \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text
```

### 3. Configure RAG in config.yml

```yaml
retriever:
  type: chroma
  persist_dir: cache/chroma_db
  data_dir: data/pdf
  embedding_provider: huggingface  # or 'ollama'
  embedding_model: all-MiniLM-L6-v2
  collection_name: architecture_docs
  search_type: similarity  # or 'mmr' for diversity
  search_kwargs:
    k: 4  # Number of documents to retrieve
```

### 4. Verify RAG is Working

```python
from config import load_config
from rag.rag_service import RAGService

config = load_config()
service = RAGService(config.retriever)

print(f"Available: {service.is_available()}")
results = service.query("dependency inversion principle", k=2)
print(f"Found {len(results)} results")
```

## Running the Pipeline

### Basic Usage

```bash
python cli.py example_cycle.json --config config.yml
```

### Using the Orchestrator Directly

```python
from config import load_config
from orchestrator import Orchestrator

config = load_config("config.yml")
orch = Orchestrator(config=config)

cycle_spec = {
    "id": "cycle-001",
    "graph": {
        "nodes": ["ModuleA", "ModuleB"],
        "edges": [["ModuleA", "ModuleB"], ["ModuleB", "ModuleA"]]
    },
    "files": [
        {"path": "src/ModuleA.cs", "content": "..."},
        {"path": "src/ModuleB.cs", "content": "..."}
    ]
}

results = orch.run_pipeline(cycle_spec)
```

## Output Artifacts

All outputs are saved to the `artifacts/` directory (source files are **never modified**):

```
artifacts/
└── {cycle-id}/
    ├── input/cycle_spec.json       # Original input
    ├── describer/description.txt   # Cycle description
    ├── proposal/proposal.json      # Refactoring proposal
    ├── patches/{file_path}         # Patched file versions
    ├── diffs/{file_path}.diff      # Unified diffs
    ├── validation/report.json      # Validation results
    └── explanation/explanation.md  # Human-readable summary
```

## Operating Modes

The pipeline supports multiple operating modes for different use cases. These are configured in `config.yml` under `refactor:`.

### Mode Comparison

| Mode | Purpose | Applies Patches? | Output |
|------|---------|-----------------|--------|
| **Standard** | Full automated refactoring | Yes | Patched files + diffs |
| **Suggestion** | Human-reviewable suggestions | No | Markdown guide for operators |
| **Roadmap** | Post-attempt progress report | After attempt | What worked, what didn't, next steps |
| **Simple Format** | Smaller LLM support (7B-14B) | Optional | Text-based format, easier to parse |
| **Line-Based** | Precise line-level patching | Yes | Line-numbered changes |

### Suggestion Mode vs Roadmap Mode

These two modes serve different purposes and are often confused:

#### Suggestion Mode (`suggestion_mode: true`)
- **When**: BEFORE any changes are applied
- **Purpose**: Generate a human-reviewable plan with step-by-step instructions
- **Use case**: When you want a human to review and apply changes manually
- **Output**: Comprehensive markdown report with:
  - Cycle context (what's the problem)
  - Suggested changes (what to do)
  - Step-by-step manual instructions
  - Common pitfalls to avoid
  - Verification steps
  - Copy-paste ready code

#### Roadmap Mode (`roadmap_mode: true`)
- **When**: AFTER an automated refactoring attempt
- **Purpose**: Show progress on an attempted refactoring (post-mortem)
- **Use case**: Demos and visibility when full automation doesn't succeed
- **Output**: Progress report with:
  - What patches succeeded
  - What patches failed (and why)
  - Classification of failures (hallucination, syntax, etc.)
  - Remaining work for humans

**TL;DR**: Use **Suggestion Mode** when you want a human to apply changes manually. Use **Roadmap Mode** to understand what happened during an automated attempt.

### Configuring Modes

In `config.yml`:

```yaml
refactor:
  # Suggestion mode - for human operators to apply manually
  suggestion_mode: true
  suggestion_output_format: markdown  # or "json"
  suggestion_context_lines: 7         # Lines of context before/after
  
  # Roadmap mode - for visibility into automated attempts
  roadmap_mode: false                 # Usually used with suggestion_mode: false
  
  # Simple format - for smaller LLMs (7B-14B)
  simple_format_mode: auto            # "auto", true, or false
  
  # Line-based patching - more precise than search/replace
  line_based_patching: true
```

### Mode Priority

When multiple modes are enabled, they are applied in this order:

1. `suggestion_mode: true` → Runs suggestion mode (no patches applied)
2. `simple_format_mode: true/auto` → Uses simple text format for LLM response
3. `line_based_patching: true` → Uses line-number based patching
4. Standard mode → Search/replace based patching

## Notes

- **Air-gapped operation**: No external API calls - all LLM inference and embeddings run locally via Ollama or HuggingFace.
- **Read-only**: Source files are never modified. Patches are stored in `artifacts/`.
- If your input JSON references absolute file paths, run the tool on the machine with access to those paths or embed file contents in the JSON.
- Logs are written to `langCodeUnderstanding.log` (configurable in `config.yml`).
