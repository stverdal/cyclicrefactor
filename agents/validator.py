from typing import Dict, Any
from .agent_base import Agent, AgentResult


class ValidatorAgent(Agent):
    name = "validator"
    version = "0.1"

    def __init__(self, linters: Dict[str, str] = None, test_command: str = None):
        """Scaffold validator.

        - `linters`: mapping of linter name to command (not executed in scaffold)
        - `test_command`: command string to run tests (not executed in scaffold)
        """
        self.linters = linters or {}
        self.test_command = test_command

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run validations on the proposal.

        Expected inputs:
        - `cycle_spec`: dict
        - `proposal`: dict produced by RefactorAgent

        Returns a report dict with structure:
        {"passed": bool, "issues": [{"path":..., "issue":..., "severity":...}], "notes": ...}
        """
        cycle = input_data.get("cycle_spec")
        proposal = input_data.get("proposal")

        if cycle is None or proposal is None:
            return AgentResult(status="error", output=None, logs="Missing inputs for validator")

        # Scaffold behaviour: do not execute linters/tests yet. Provide hooks and a basic sanity check.
        issues = []

        # Sanity checks: ensure patches cover known files and are non-empty
        known_paths = {f.get("path") for f in cycle.get("files", [])}
        patches = proposal.get("patches", []) if isinstance(proposal, dict) else []

        for p in patches:
            path = p.get("path")
            if path not in known_paths and path.split('/')[-1] not in known_paths:
                issues.append({"path": path, "issue": "Patch targets unknown file", "severity": "medium"})
            if not p.get("patched"):
                issues.append({"path": path, "issue": "Patched content empty", "severity": "high"})

        passed = len(issues) == 0

        report = {"passed": passed, "issues": issues, "notes": "Scaffold validator: integrate linters and test runners to perform real checks."}

        return AgentResult(status="success", output=report)
