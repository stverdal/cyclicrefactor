from typing import List, Dict, Any, Optional, Union
from agents.agent_base import AgentResult
from agents.describer import DescriberAgent
from agents.refactor_agent import RefactorAgent
from agents.validator import ValidatorAgent
from agents.explainer import ExplainerAgent
from utils.persistence import Persistor
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, UnbreakableReason
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

        logger.info("Initializing Orchestrator...")

        # Create shared LLM client from config
        try:
            self.llm = create_llm_from_config(self.config.llm)
            if self.llm:
                logger.info(f"LLM client initialized: {self.config.llm.provider}/{self.config.llm.model}")
            else:
                logger.warning("No LLM available - agents will use fallback logic")
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
            self.llm = None
        
        # Create RAG service from config
        try:
            self.rag_service = RAGService(self.config.retriever)
            if self.rag_service.is_available():
                logger.info(f"RAG service initialized: {self.config.retriever.persist_dir}")
            else:
                logger.warning("RAG service not available - run 'python -m rag.pdf_indexer' first")
                self.rag_service = None
        except Exception as e:
            logger.warning(f"Failed to create RAG service: {e}")
            self.rag_service = None

        # Pipeline settings
        self.max_iterations = getattr(self.config.pipeline, "max_iterations", 2)
        self.agents_order = getattr(self.config.pipeline, "agents_order", ["describer", "refactor", "validator"])
        self.enabled_agents = getattr(self.config.pipeline, "enable", {})
        
        # Unbreakable cycle detector
        self.query_builder = RAGQueryBuilder()

    def _is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in config."""
        return self.enabled_agents.get(agent_name, True)

    def _get_prompt_template(self, agent_name: str) -> Optional[str]:
        """Get prompt template path for an agent from config."""
        if hasattr(self.config, "prompts") and isinstance(self.config.prompts, dict):
            return self.config.prompts.get(agent_name)
        return None

    def _check_unbreakable_cycle(
        self, cycle_spec: CycleSpec, validation_history: List[Dict[str, Any]] = None
    ) -> tuple:
        """Check if the cycle is unbreakable.
        
        Returns:
            Tuple of (is_unbreakable, reason, explanation)
        """
        return self.query_builder.detect_unbreakable_cycle(
            cycle_spec.model_dump(), validation_history
        )

    def _generate_failure_explanation(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        validation_history: List[Dict[str, Any]],
        reason: UnbreakableReason = None,
    ) -> Dict[str, Any]:
        """Generate an explanation for why the refactoring failed.
        
        This runs when the pipeline fails after max_iterations or
        when an unbreakable cycle is detected.
        """
        logger.info("Generating failure explanation...")
        
        nodes = cycle_spec.graph.nodes
        node_list = ", ".join(nodes[:5])
        
        # Get detailed explanation for unbreakable reason
        if reason and reason != UnbreakableReason.NOT_UNBREAKABLE:
            detailed_explanation = self.query_builder.get_unbreakable_explanation(
                reason, cycle_spec.model_dump()
            )
            title = f"Cycle Cannot Be Broken: {reason.value.replace('_', ' ').title()}"
        else:
            # Generic failure after max iterations
            issues_summary = []
            for val in validation_history:
                for issue in val.get("issues", [])[:3]:
                    if isinstance(issue, dict):
                        issues_summary.append(issue.get("comment", ""))
            
            unique_issues = list(set(issues_summary))[:5]
            
            detailed_explanation = f"""
## Refactoring Failed After {len(validation_history)} Attempts

The cycle between **{node_list}** could not be automatically broken.

### Validation Issues Encountered
{chr(10).join(f"- {issue}" for issue in unique_issues) if unique_issues else "- No specific issues captured"}

### Possible Reasons
1. The LLM may need more context about the codebase
2. The cycle pattern may require a strategy not in the RAG knowledge base
3. Multiple interdependent changes may be needed simultaneously
4. The code may have constraints not visible in the provided files

### Recommended Actions
1. **Manual Review**: Examine the files and identify the specific import creating the cycle
2. **Provide More Context**: Include related files or configuration
3. **Try Different Strategy**: Manually specify a refactoring approach (interface extraction, shared module, etc.)
4. **Accept the Cycle**: If the coupling is intentional, document it as such
"""
            title = "Refactoring Failed: Max Iterations Reached"
        
        # Build the failure explanation
        explanation = {
            "title": title,
            "status": "failed",
            "cycle_id": cycle_spec.id,
            "nodes": nodes,
            "attempts": len(validation_history),
            "explanation": detailed_explanation,
            "description_summary": description.text[:500] if description.text else "",
            "validation_history": [
                {
                    "iteration": i + 1,
                    "approved": v.get("approved", False),
                    "issues_count": len(v.get("issues", [])),
                    "summary": v.get("summary", "")[:200],
                }
                for i, v in enumerate(validation_history)
            ],
            "recommendations": [
                "Review the cycle manually to understand the root cause",
                "Consider if the cycle is intentional (framework pattern, data model)",
                "Try a more targeted refactoring with explicit strategy hints",
                "Consult with code owners about the design intent",
            ],
        }
        
        return explanation

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
        pipeline_start = time.time()

        logger.info("="*50)
        logger.info(f"PIPELINE START: {artifact_id}")
        logger.info("="*50)
        logger.info(f"Cycle: {len(cycle_spec.graph.nodes)} nodes, {len(cycle_spec.graph.edges)} edges, {len(cycle_spec.files)} files")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"LLM available: {self.llm is not None}")
        logger.info(f"RAG available: {self.rag_service is not None}")

        # Persist input (serialize CycleSpec to dict for JSON storage)
        try:
            self.persistor.persist_cycle_input(artifact_id, cycle_spec.model_dump())
        except Exception as e:
            logger.warning(f"Failed to persist cycle input: {e}")

        # -------------------------------------------------------------------------
        # Early Check: Detect Obviously Unbreakable Cycles
        # -------------------------------------------------------------------------
        is_unbreakable, unbreakable_reason, unbreakable_msg = self._check_unbreakable_cycle(cycle_spec)
        if is_unbreakable:
            logger.warning(f"UNBREAKABLE CYCLE DETECTED: {unbreakable_reason.value}")
            logger.warning(f"Reason: {unbreakable_msg}")
            
            # Generate failure explanation
            failure_explanation = self._generate_failure_explanation(
                cycle_spec,
                CycleDescription(text="Cycle detected as unbreakable before analysis"),
                [],
                unbreakable_reason,
            )
            results["explanation"] = failure_explanation
            results["status"] = "unbreakable"
            results["unbreakable_reason"] = unbreakable_reason.value
            
            # Persist the explanation
            try:
                self.persistor.save_json(artifact_id, "explanation/failure_explanation.json", failure_explanation)
                self.persistor.save_text(artifact_id, "explanation/failure_explanation.md", failure_explanation.get("explanation", ""))
            except Exception as e:
                logger.warning(f"Failed to persist failure explanation: {e}")
            
            pipeline_elapsed = time.time() - pipeline_start
            logger.info("="*50)
            logger.info(f"PIPELINE COMPLETE: {artifact_id} (UNBREAKABLE)")
            logger.info("="*50)
            logger.info(f"Total time: {pipeline_elapsed:.1f}s")
            
            return results

        # Track validation history for pattern detection
        validation_history: List[Dict[str, Any]] = []

        # -------------------------------------------------------------------------
        # Stage 1: Describer
        # -------------------------------------------------------------------------
        desc_result = None
        if self._is_agent_enabled("describer"):
            stage_start = time.time()
            logger.info("-"*50)
            logger.info("STAGE 1: DESCRIBER")
            logger.info("-"*50)
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
            
            elapsed = time.time() - stage_start
            logger.info(f"Describer completed in {elapsed:.1f}s")
            if desc_result.output and isinstance(desc_result.output, dict):
                text = desc_result.output.get("text", "")
                logger.info(f"Description length: {len(text)} chars")
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

        logger.info("-"*50)
        logger.info("STAGE 2-3: REFACTOR + VALIDATE LOOP")
        logger.info("-"*50)

        while iteration < self.max_iterations:
            iteration += 1
            iter_start = time.time()
            logger.info(f"\n>>> Iteration {iteration}/{self.max_iterations}")

            # Run Refactor agent
            if self._is_agent_enabled("refactor"):
                logger.info("Running Refactor agent...")
                refactor_start = time.time()
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
                
                refactor_elapsed = time.time() - refactor_start
                if ref_result.output and isinstance(ref_result.output, dict):
                    patches = ref_result.output.get("patches", [])
                    changed = len([p for p in patches if p.get("diff")])
                    logger.info(f"Refactor completed in {refactor_elapsed:.1f}s: {changed}/{len(patches)} files changed")
                else:
                    logger.info(f"Refactor completed in {refactor_elapsed:.1f}s")
            else:
                logger.info("Refactor agent disabled, skipping")
                break

            # Run Validator agent
            if self._is_agent_enabled("validator"):
                logger.info("Running Validator agent...")
                validator_start = time.time()
                validator = ValidatorAgent(
                    llm=self.llm,
                    prompt_template=self._get_prompt_template("validator"),
                    linters=getattr(self.config, "validator", {}).get("linters") if hasattr(self.config, "validator") else None,
                    test_command=getattr(self.config, "validator", {}).get("test_command") if hasattr(self.config, "validator") else None,
                    rag_service=self.rag_service,
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
                validator_elapsed = time.time() - validator_start
                
                # Track validation history for unbreakable detection
                if val_result.output:
                    validation_history.append(val_result.output)
                
                if val_result.output and val_result.output.get("approved"):
                    logger.info(f"\n✓ PROPOSAL APPROVED on iteration {iteration} (validated in {validator_elapsed:.1f}s)")
                    break
                else:
                    # Prepare feedback for next iteration
                    validator_feedback = val_result.output
                    issues = validator_feedback.get("issues", []) if validator_feedback else []
                    suggestions = validator_feedback.get("suggestions", []) if validator_feedback else []
                    logger.warning(f"\n✗ PROPOSAL REJECTED on iteration {iteration}: {len(issues)} issue(s), {len(suggestions)} suggestion(s)")
                    for issue in issues[:3]:
                        if isinstance(issue, dict):
                            logger.warning(f"  - {issue.get('path', 'unknown')}: {issue.get('comment', 'no details')}")
                    if len(issues) > 3:
                        logger.warning(f"  ... and {len(issues) - 3} more issues")

                    if iteration >= self.max_iterations:
                        logger.warning(f"\n⚠ MAX ITERATIONS REACHED ({self.max_iterations}) without approval")
                        
                        # Check if cycle is unbreakable based on validation history
                        is_unbreakable, unbreakable_reason, _ = self._check_unbreakable_cycle(
                            cycle_spec, validation_history
                        )
                        
                        if is_unbreakable:
                            results["status"] = "unbreakable"
                            results["unbreakable_reason"] = unbreakable_reason.value
                            logger.warning(f"Cycle appears UNBREAKABLE: {unbreakable_reason.value}")
                        else:
                            results["status"] = "max_iterations_reached"
                        
                        results["final_validation"] = val_result.output
            else:
                logger.info("Validator agent disabled, skipping validation")
                break

        # -------------------------------------------------------------------------
        # Stage 4: Explainer (only if approved) OR Failure Explanation
        # -------------------------------------------------------------------------
        approved = val_result and val_result.output and val_result.output.get("approved", False)

        if approved and self._is_agent_enabled("explainer"):
            logger.info("-"*50)
            logger.info("STAGE 4: EXPLAINER")
            logger.info("-"*50)
            explainer_start = time.time()
            explainer = ExplainerAgent(
                llm=self.llm,
                prompt_template=self._get_prompt_template("explainer"),
                rag_service=self.rag_service,
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
            
            explainer_elapsed = time.time() - explainer_start
            if exp_result.output and isinstance(exp_result.output, dict):
                logger.info(f"Explainer completed in {explainer_elapsed:.1f}s: {exp_result.output.get('title', 'no title')}")
            else:
                logger.info(f"Explainer completed in {explainer_elapsed:.1f}s")
        elif approved:
            logger.info("Explainer agent disabled, skipping explanation generation")
        elif validation_history:
            # Generate failure explanation when not approved
            logger.info("-"*50)
            logger.info("STAGE 4: FAILURE EXPLANATION")
            logger.info("-"*50)
            
            unbreakable_reason = None
            if results.get("status") == "unbreakable":
                unbreakable_reason = UnbreakableReason(results.get("unbreakable_reason", "not_unbreakable"))
            
            failure_explanation = self._generate_failure_explanation(
                cycle_spec,
                description,
                validation_history,
                unbreakable_reason,
            )
            results["explanation"] = failure_explanation
            
            # Persist failure explanation
            try:
                self.persistor.save_json(artifact_id, "explanation/failure_explanation.json", failure_explanation)
                self.persistor.save_text(artifact_id, "explanation/failure_explanation.md", failure_explanation.get("explanation", ""))
            except Exception as e:
                logger.warning(f"Failed to persist failure explanation: {e}")
            
            logger.info(f"Failure explanation generated: {failure_explanation.get('title', 'Unknown')}")
        else:
            logger.info("Skipping Explainer (proposal not approved, no validation history)")

        # -------------------------------------------------------------------------
        # Finalize
        # -------------------------------------------------------------------------
        results["status"] = "approved" if approved else results.get("status", "not_approved")
        results["iterations"] = iteration
        
        pipeline_elapsed = time.time() - pipeline_start

        # Save run metadata
        try:
            self.persistor.save_json(artifact_id, "run_metadata.json", {
                "artifact_id": artifact_id,
                "timestamp": int(time.time()),
                "duration_seconds": round(pipeline_elapsed, 1),
                "iterations": iteration,
                "approved": approved,
                "agents_run": [a for a in ["describer", "refactor", "validator", "explainer"] if self._is_agent_enabled(a)],
            })
        except Exception as e:
            logger.warning(f"Failed to persist run metadata: {e}")

        logger.info("="*50)
        logger.info(f"PIPELINE COMPLETE: {artifact_id}")
        logger.info("="*50)
        logger.info(f"Status: {results['status']}")
        logger.info(f"Iterations: {iteration}")
        logger.info(f"Total time: {pipeline_elapsed:.1f}s")
        logger.info(f"Artifacts: {self.config.io.artifacts_dir}/{artifact_id}/")
        
        return results
