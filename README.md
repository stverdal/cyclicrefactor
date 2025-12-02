Scaffold for agentic LangChain pipeline

- `cli.py`: CLI entrypoint. Run with `python cli.py example_cycle.json`.
- `orchestrator.py`: Simple in-process orchestrator that runs Describer then Refactor.
- `agents/`: Agent implementations and base class.
- `models/schemas.py`: Pydantic schemas for CycleSpec, CycleDescription, and RefactorProposal.

Next steps:
- Implement LLM calls inside `agents/describer.py` and `agents/refactor_agent.py`.
- Add persistence of artifacts and provenance.
- Add validators and iterative loop support.

Quick setup (virtualenv)
 - Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

 - On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the CLI (example):

```bash
python cli.py example_cycle.json --config config.yml
```

Notes:
- If your input JSON references absolute file paths, run the tool on the machine with access to those paths or embed file contents in the JSON and use `--no-read-files`.
- Keep API keys in environment variables (see `config.yml` for expected names).
