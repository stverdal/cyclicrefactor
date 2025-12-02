from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    status: str
    output: Any
    logs: str = ""
    metadata: Dict[str, Any] = None


class Agent:
    """Base class for agents."""

    name: str = "base"
    version: str = "0.1"

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError()
