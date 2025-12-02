from typing import List, Dict, Any, Optional
from agents.agent_base import AgentResult
from agents.describer import DescriberAgent
from agents.refactor_agent import RefactorAgent
from agents.validator import ValidatorAgent
from utils.persistence import Persistor
from agents.llm_utils import create_llm_from_config
from config import AppConfig
import time
import json


class Orchestrator:
    def __init__(self, agents: List = None, config: Optional[AppConfig] = None):
        self.agents = agents or []
        self.config = config or AppConfig()
        # create persistor with configured base dir
        self.persistor = Persistor(base_dir=self.config.io.artifacts_dir)
        # create a shared LLM client from config
        try:
            self.llm = create_llm_from_config(self.config.llm)
        except Exception:
            self.llm = None

    def register(self, agent):
        self.agents.append(agent)

    def run_pipeline(self, cycle_spec: Dict[str, Any], prompt: str = None) -> Dict[str, Any]:
        results = {}

        # Decide artifact id
        artifact_id = cycle_spec.get("id") or f"run-{int(time.time())}"
        persistor = self.persistor

        # Persist input
        try:
            persistor.persist_cycle_input(artifact_id, cycle_spec)
        except Exception:
            pass

        # Run Describer
        describer = DescriberAgent(llm=self.llm, prompt_template=self.config.prompts.get("describer") if hasattr(self.config, "prompts") else None)
        desc_res: AgentResult = describer.run({"cycle_spec": cycle_spec, "prompt": prompt})
        results["description"] = desc_res.output

        # Persist description and any logs
        try:
            if desc_res.output and isinstance(desc_res.output, dict):
                persistor.persist_description(artifact_id, desc_res.output.get("text", ""))
            if desc_res.logs:
                persistor.save_text(artifact_id, "describer/logs.txt", desc_res.logs)
        except Exception:
            pass

        # Run Refactor
        refactor = RefactorAgent(llm=self.llm, prompt_template=self.config.prompts.get("refactor") if hasattr(self.config, "prompts") else None)
        ref_res: AgentResult = refactor.run({"cycle_spec": cycle_spec, "cycle_description": desc_res.output, "prompt": prompt})
        results["proposal"] = ref_res.output

        # Persist refactor outputs
        try:
            if ref_res.output:
                # Save raw LLM response if present
                if isinstance(ref_res.output, dict) and "llm_response" in ref_res.output:
                    persistor.persist_llm_response(artifact_id, "refactor", ref_res.output.get("llm_response"))

                persistor.persist_proposal(artifact_id, ref_res.output if isinstance(ref_res.output, dict) else {"raw": str(ref_res.output)})

            if ref_res.logs:
                persistor.save_text(artifact_id, "refactor/logs.txt", ref_res.logs)
        except Exception:
            pass

        # Run Validator (scaffold)
        try:
            validator = ValidatorAgent()
            val_res = validator.run({"cycle_spec": cycle_spec, "proposal": ref_res.output})
            results["validation"] = val_res.output
            # persist validation
            try:
                persistor.save_json(artifact_id, "validation/report.json", val_res.output)
            except Exception:
                pass
        except Exception:
            # best-effort: don't fail pipeline
            results["validation"] = None

        # Save a run metadata file
        try:
            persistor.save_json(artifact_id, "run_metadata.json", {"artifact_id": artifact_id, "timestamp": int(time.time()), "description_status": desc_res.status, "refactor_status": ref_res.status})
        except Exception:
            pass

        return results
