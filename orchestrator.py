from typing import List, Dict, Any, Optional, Union
from agents.agent_base import AgentResult
from agents.describer import DescriberAgent
from agents.refactor_agent import RefactorAgent
from agents.validator import ValidatorAgent
from agents.explainer import ExplainerAgent
from utils.persistence import Persistor
from utils.logging import get_logger
from agents.llm_utils import create_llm_from_config
from config import AppConfig
from models.schemas import (
    CycleSpec,
    CycleDescription,
    RefactorProposal,
    ValidationReport,
    Explanation,
)
from rag.rag_service import RAGService
import time
import json

logger = get_logger("orchestrator")


class Orchestrator:
    """Orchestrates the cycle-refactoring pipeline.

    Pipeline flow:
    1. Describer - analyzes and describes the cycle
    2. Refactor - proposes patches to break the cycle
    3. Validator - reviews the proposal (retry loop if rejected)
    4. Explainer - generates human-readable summary (only if approved)
    """

    def __init__(self, agents: List = None, config: Optional[AppConfig] = None):
        self.agents = agents or []
        self.config = config or AppConfig()
        self.persistor = Persistor(base_dir=self.config.io.artifacts_dir)

        # Create shared LLM client from config
        try:
            self.llm = create_llm_from_config(self.config.llm)
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
            self.llm = None
        
        # Create RAG service from config
        try:
            self.rag_service = RAGService(self.config.retriever)
            if self.rag_service.is_available():
                logger.info("RAG service initialized and available")
            else:
                logger.warning("RAG service not available - run pdf_indexer first")
                self.rag_service = None
        except Exception as e:
            logger.warning(f"Failed to create RAG service: {e}")
            self.rag_service = None

        # Pipeline settings
        self.max_iterations = getattr(self.config.pipeline, "max_iterations", 2)
        self.agents_order = getattr(self.config.pipeline, "agents_order", ["describer", "refactor", "validator"])
        self.enabled_agents = getattr(self.config.pipeline, "enable", {})

    def _is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in config."""
        return self.enabled_agents.get(agent_name, True)

    def _get_prompt_template(self, agent_name: str) -> Optional[str]:
        """Get prompt template path for an agent from config."""
        if hasattr(self.config, "prompts") and isinstance(self.config.prompts, dict):
            return self.config.prompts.get(agent_name)
        return None

    def register(self, agent):
        self.agents.append(agent)

    def run_pipeline(self, cycle_spec: Union[CycleSpec, Dict[str, Any]], prompt: str = None) -> Dict[str, Any]:
        """Run the full refactoring pipeline.

        Args:
            cycle_spec: Cycle specification (CycleSpec model or dict with id, graph, files).
            prompt: Optional additional prompt/instructions.

        Returns:
            Dict with results from each pipeline stage.
        """
        # Convert dict to CycleSpec if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)

        results = {}
        artifact_id = cycle_spec.id or f"run-{int(time.time())}"

        logger.info(f"Starting pipeline for artifact_id={artifact_id}")

        # Persist input (serialize CycleSpec to dict for JSON storage)
        try:
            self.persistor.persist_cycle_input(artifact_id, cycle_spec.model_dump())
        except Exception as e:
            logger.warning(f"Failed to persist cycle input: {e}")

        # -------------------------------------------------------------------------
        # Stage 1: Describer
        # -------------------------------------------------------------------------
        desc_result = None
        if self._is_agent_enabled("describer"):
            logger.info("Running Describer agent")
            describer = DescriberAgent(
                llm=self.llm,
                prompt_template=self._get_prompt_template("describer"),
                rag_service=self.rag_service,
            )
            desc_result = describer.run(cycle_spec, prompt=prompt)
            results["description"] = desc_result.output

            # Persist description
            try:
                if desc_result.output and isinstance(desc_result.output, dict):
                    self.persistor.persist_description(artifact_id, desc_result.output.get("text", ""))
                if desc_result.logs:
                    self.persistor.save_text(artifact_id, "describer/logs.txt", desc_result.logs)
            except Exception as e:
                logger.warning(f"Failed to persist description: {e}")

            if desc_result.status != "success":
                logger.error(f"Describer failed: {desc_result.logs}")
                results["status"] = "error"
                results["error"] = f"Describer failed: {desc_result.logs}"
                return results
        else:
            # Provide minimal description if describer is disabled
            desc_result = AgentResult(status="success", output={"text": "Describer disabled"})
            results["description"] = desc_result.output

        # -------------------------------------------------------------------------
        # Stage 2 & 3: Refactor + Validator (with retry loop)
        # -------------------------------------------------------------------------
        ref_result = None
        val_result = None
        validator_feedback = None
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Refactor/Validate iteration {iteration}/{self.max_iterations}")

            # Run Refactor agent
            if self._is_agent_enabled("refactor"):
                refactor = RefactorAgent(
                    llm=self.llm,
                    prompt_template=self._get_prompt_template("refactor"),
                    rag_service=self.rag_service,
                )
                # Convert description to CycleDescription model if needed
                description = desc_result.output
                if isinstance(description, dict):
                    description = CycleDescription.model_validate(description)
                
                ref_result = refactor.run(
                    cycle_spec,
                    description=description,
                    validator_feedback=validator_feedback,
                    prompt=prompt,
                )
                results["proposal"] = ref_result.output

                # Persist refactor outputs
                try:
                    if ref_result.output:
                        if isinstance(ref_result.output, dict) and "llm_response" in ref_result.output:
                            self.persistor.persist_llm_response(artifact_id, f"refactor_iter{iteration}", ref_result.output.get("llm_response"))
                        self.persistor.persist_proposal(artifact_id, ref_result.output if isinstance(ref_result.output, dict) else {"raw": str(ref_result.output)})
                    if ref_result.logs:
                        self.persistor.save_text(artifact_id, f"refactor/logs_iter{iteration}.txt", ref_result.logs)
                except Exception as e:
                    logger.warning(f"Failed to persist refactor output: {e}")

                if ref_result.status != "success":
                    logger.error(f"Refactor failed: {ref_result.logs}")
                    results["status"] = "error"
                    results["error"] = f"Refactor failed: {ref_result.logs}"
                    return results
            else:
                logger.info("Refactor agent disabled, skipping")
                break

            # Run Validator agent
            if self._is_agent_enabled("validator"):
                validator = ValidatorAgent(
                    llm=self.llm,
                    prompt_template=self._get_prompt_template("validator"),
                    linters=getattr(self.config, "validator", {}).get("linters") if hasattr(self.config, "validator") else None,
                    test_command=getattr(self.config, "validator", {}).get("test_command") if hasattr(self.config, "validator") else None,
                )
                # Convert proposal to RefactorProposal model if needed
                proposal = ref_result.output
                if isinstance(proposal, dict):
                    proposal = RefactorProposal.model_validate(proposal)
                
                val_result = validator.run(
                    cycle_spec,
                    description=description,
                    proposal=proposal,
                )
                results["validation"] = val_result.output

                # Persist validation
                try:
                    self.persistor.save_json(artifact_id, f"validation/report_iter{iteration}.json", val_result.output)
                except Exception as e:
                    logger.warning(f"Failed to persist validation: {e}")

                if val_result.status != "success":
                    logger.error(f"Validator failed: {val_result.logs}")
                    results["status"] = "error"
                    results["error"] = f"Validator failed: {val_result.logs}"
                    return results

                # Check if approved
                if val_result.output and val_result.output.get("approved"):
                    logger.info(f"Proposal approved on iteration {iteration}")
                    break
                else:
                    # Prepare feedback for next iteration
                    validator_feedback = val_result.output
                    logger.info(f"Proposal rejected on iteration {iteration}, will retry with feedback")

                    if iteration >= self.max_iterations:
                        logger.warning("Max iterations reached without approval")
                        results["status"] = "max_iterations_reached"
                        results["final_validation"] = val_result.output
            else:
                logger.info("Validator agent disabled, skipping validation")
                break

        # -------------------------------------------------------------------------
        # Stage 4: Explainer (only if approved)
        # -------------------------------------------------------------------------
        approved = val_result and val_result.output and val_result.output.get("approved", False)

        if approved and self._is_agent_enabled("explainer"):
            logger.info("Running Explainer agent")
            explainer = ExplainerAgent(
                llm=self.llm,
                prompt_template=self._get_prompt_template("explainer"),
            )
            # Convert validation to ValidationReport model if needed
            validation = val_result.output
            if isinstance(validation, dict):
                validation = ValidationReport.model_validate(validation)
            
            exp_result = explainer.run(
                cycle_spec,
                description=description,
                proposal=proposal,
                validation=validation,
            )
            results["explanation"] = exp_result.output

            # Persist explanation
            try:
                if exp_result.output:
                    self.persistor.save_json(artifact_id, "explanation/explanation.json", exp_result.output)
                    if isinstance(exp_result.output, dict) and "explanation" in exp_result.output:
                        self.persistor.save_text(artifact_id, "explanation/explanation.md", exp_result.output.get("explanation", ""))
            except Exception as e:
                logger.warning(f"Failed to persist explanation: {e}")

        # -------------------------------------------------------------------------
        # Finalize
        # -------------------------------------------------------------------------
        results["status"] = "approved" if approved else results.get("status", "not_approved")
        results["iterations"] = iteration

        # Save run metadata
        try:
            self.persistor.save_json(artifact_id, "run_metadata.json", {
                "artifact_id": artifact_id,
                "timestamp": int(time.time()),
                "iterations": iteration,
                "approved": approved,
                "agents_run": [a for a in ["describer", "refactor", "validator", "explainer"] if self._is_agent_enabled(a)],
            })
        except Exception as e:
            logger.warning(f"Failed to persist run metadata: {e}")

        logger.info(f"Pipeline complete: status={results['status']}, iterations={iteration}")
        return results
