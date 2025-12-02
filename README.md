Scaffold for agentic LangChain pipeline

- `cli.py`: CLI entrypoint. Run with `python cli.py example_cycle.json`.
- `orchestrator.py`: Simple in-process orchestrator that runs Describer then Refactor.
- `agents/`: Agent implementations and base class.
- `models/schemas.py`: Pydantic schemas for CycleSpec, CycleDescription, and RefactorProposal.

Next steps:
- Implement LLM calls inside `agents/describer.py` and `agents/refactor_agent.py`.
- Add persistence of artifacts and provenance.
- Add validators and iterative loop support.
