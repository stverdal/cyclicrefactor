from typing import List, Dict, Any, Optional, Union
from agents.agent_base import AgentResult
from agents.describer import DescriberAgent
from agents.refactor_agent import RefactorAgent
from agents.validator import ValidatorAgent
from agents.explainer import ExplainerAgent
from agents.failure_explainer import FailureExplainerAgent
from agents.dependency_analyzer import DependencyAnalyzerAgent
from agents.cycle_detector import CycleDetectorAgent
from utils.persistence import Persistor
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, UnbreakableReason
from utils.llm_logger import configure_from_config as configure_llm_logger
from agents.llm_utils import create_llm_from_config
from config import AppConfig
from models.schemas import (
    CycleSpec,
    CycleDescription,
    RefactorProposal,
    ValidationReport,
    Explanation,
    DependencyGraph,
    DetectedCycle,
)
from rag.rag_service import RAGService
import time
import json
from utils.failure_report import build_failure_report

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
        
        # Check for dry-run mode
        self.dry_run = getattr(self.config.pipeline, "dry_run", False)
        dry_run_log_writes = getattr(self.config.pipeline, "dry_run_log_writes", True)
        
        if self.dry_run:
            logger.info("=== DRY-RUN MODE: No files will be written ===")
        
        self.persistor = Persistor(
            base_dir=self.config.io.artifacts_dir,
            dry_run=self.dry_run,
            log_writes=dry_run_log_writes,
        )

        logger.info("Initializing Orchestrator...")
        
        # Configure LLM I/O logging from config
        if hasattr(self.config, 'logging') and hasattr(self.config.logging, 'log_llm_io'):
            try:
                configure_llm_logger({
                    "log_llm_io": self.config.logging.log_llm_io,
                    "llm_io_log_file": self.config.logging.llm_io_log_file,
                    "log_llm_prompts": self.config.logging.log_llm_prompts,
                    "log_llm_responses": self.config.logging.log_llm_responses,
                    "truncate_llm_logs": self.config.logging.truncate_llm_logs,
                    "log_llm_timestamps": self.config.logging.log_llm_timestamps,
                })
                if self.config.logging.log_llm_io:
                    logger.info(f"LLM I/O logging enabled: {self.config.logging.llm_io_log_file}")
            except Exception as e:
                logger.warning(f"Failed to configure LLM I/O logging: {e}")

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
        
        # Context settings (with fallbacks)
        self.context_window = getattr(self.config, "context", {}).get("window_size", 4096) if hasattr(self.config, "context") else 4096
        self.max_file_chars = getattr(self.config, "context", {}).get("max_file_chars", 4000) if hasattr(self.config, "context") else 4000
        
        # Also check LLM params for context size (may override)
        if hasattr(self.config, "llm") and hasattr(self.config.llm, "params"):
            llm_ctx = getattr(self.config.llm.params, "num_ctx", None)
            if llm_ctx:
                self.context_window = llm_ctx
                logger.info(f"Context window from LLM config: {self.context_window} tokens")
        
        # Unbreakable cycle detector
        self.query_builder = RAGQueryBuilder()

    def _is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in config."""
        return self.enabled_agents.get(agent_name, True)

    def _get_prompt_template(self, agent_name: str, compact: bool = False) -> Optional[str]:
        """Get prompt template path for an agent from config.
        
        Args:
            agent_name: Name of the agent (e.g., 'refactor', 'validator')
            compact: If True, try to get the compact variant
            
        Returns:
            Path to the prompt template file, or None if not configured
        """
        if hasattr(self.config, "prompts") and isinstance(self.config.prompts, dict):
            if compact:
                # Try compact variant first
                compact_key = f"{agent_name}_compact"
                if compact_key in self.config.prompts:
                    return self.config.prompts.get(compact_key)
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
                context_window=self.context_window,
                max_file_chars=self.max_file_chars,
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
        
        # Accumulate failure history across all iterations
        accumulated_failed_strategies = []
        accumulated_reverted_files = []
        attempt_summaries = []

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
                    prompt_template_compact=self._get_prompt_template("refactor", compact=True),
                    prompt_template_plan=self._get_prompt_template("refactor_plan"),
                    prompt_template_plan_compact=self._get_prompt_template("refactor_plan", compact=True),
                    prompt_template_file=self._get_prompt_template("refactor_file"),
                    prompt_template_file_compact=self._get_prompt_template("refactor_file", compact=True),
                    rag_service=self.rag_service,
                    context_window=self.context_window,
                    max_file_chars=self.max_file_chars,
                    refactor_config=self.config.refactor,
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
                
                # Check if this is a relaxed suggestion mode output (skip validation)
                is_relaxed_suggestion = (
                    isinstance(ref_result.output, dict) and 
                    ref_result.output.get("mode") == "relaxed_suggestion"
                )
                
                if is_relaxed_suggestion:
                    skip_validation = ref_result.output.get("skip_validation", True)
                    logger.info(f"Relaxed suggestion mode detected (skip_validation={skip_validation})")
                    
                    # Save the markdown report
                    try:
                        markdown_report = ref_result.output.get("markdown_report", "")
                        if markdown_report:
                            self.persistor.save_text(artifact_id, "suggestion/relaxed_suggestion.md", markdown_report)
                            logger.info(f"Saved relaxed suggestion report to suggestion/relaxed_suggestion.md")
                        
                        # Also save the raw suggestion
                        raw_suggestion = ref_result.output.get("suggestion", "")
                        if raw_suggestion:
                            self.persistor.save_text(artifact_id, "suggestion/raw_suggestion.txt", raw_suggestion)
                    except Exception as e:
                        logger.warning(f"Failed to persist relaxed suggestion: {e}")
                    
                    if skip_validation:
                        logger.info("Skipping validation for relaxed suggestion mode")
                        results["status"] = "relaxed_suggestion"
                        results["iterations"] = iteration
                        
                        pipeline_elapsed = time.time() - pipeline_start
                        logger.info("="*50)
                        logger.info(f"PIPELINE COMPLETE: {artifact_id} (RELAXED SUGGESTION)")
                        logger.info("="*50)
                        logger.info(f"Total time: {pipeline_elapsed:.1f}s")
                        logger.info(f"Output saved to: {self.config.io.artifacts_dir}/{artifact_id}/suggestion/")
                        
                        return results
                
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
                    refactor_config=self.config.refactor,  # Pass refactor config for validation settings
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
                    # Prepare enriched feedback for next iteration
                    validator_feedback = val_result.output
                    
                    # Enrich feedback with retry context
                    if validator_feedback:
                        # Add information about reverted files from this attempt
                        reverted_files = ref_result.output.get("reverted_files", []) if ref_result.output else []
                        
                        # Build attempt summary for this iteration
                        attempt_summary = (
                            f"Attempt {iteration}: {len(proposal.patches)} files processed"
                        )
                        if reverted_files:
                            attempt_summary += f", {len(reverted_files)} reverted"
                            reverted_paths = [rf.get('path', rf.path if hasattr(rf, 'path') else 'unknown') for rf in reverted_files[:3]]
                            attempt_summary += f" ({', '.join(reverted_paths)})"
                        
                        # Accumulate failure history across all iterations
                        attempt_summaries.append(attempt_summary)
                        accumulated_reverted_files.extend(reverted_files)
                        
                        # Track what strategies failed based on issue types
                        issue_types = set()
                        for issue in validator_feedback.get("issues", []):
                            if isinstance(issue, dict):
                                issue_types.add(issue.get("issue_type", "semantic"))
                            elif hasattr(issue, "issue_type"):
                                issue_types.add(issue.issue_type)
                        
                        # Add new failed strategies (avoid duplicates)
                        if "syntax" in issue_types:
                            strategy = "full file output (caused syntax errors)"
                            if strategy not in accumulated_failed_strategies:
                                accumulated_failed_strategies.append(strategy)
                        if "cycle" in issue_types:
                            strategy = "refactoring approach (cycle not broken)"
                            if strategy not in accumulated_failed_strategies:
                                accumulated_failed_strategies.append(strategy)
                        
                        # Provide ACCUMULATED history to the next iteration
                        validator_feedback["previous_reverted_files"] = accumulated_reverted_files
                        validator_feedback["previous_attempt_summary"] = " | ".join(attempt_summaries)
                        validator_feedback["failed_strategies"] = accumulated_failed_strategies.copy()
                        validator_feedback["iteration"] = iteration
                        validator_feedback["remaining_attempts"] = self.max_iterations - iteration
                    
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
            
            # Use FailureExplainerAgent for detailed failure analysis
            # This provides actionable guidance for human operators
            failure_explainer = FailureExplainerAgent(
                llm=self.llm,
                include_llm_suggestions=self.llm is not None,
            )
            
            # Get the last proposal for detailed analysis
            last_proposal = ref_result.output if ref_result else None
            last_validation = val_result.output if val_result else None
            
            failure_result = failure_explainer.run(
                cycle_spec=cycle_spec,
                description=description,
                proposal=last_proposal,
                validation=last_validation,
                validation_history=validation_history,
            )
            
            if failure_result.status == "success" and failure_result.output:
                failure_explanation = failure_result.output
                # Add unbreakable reason if applicable
                if unbreakable_reason:
                    failure_explanation["unbreakable_reason"] = unbreakable_reason.value
            else:
                # Fallback to basic explanation
                failure_explanation = self._generate_failure_explanation(
                    cycle_spec,
                    description,
                    validation_history,
                    unbreakable_reason,
                )

            # Augment failure explanation with log-based report (human readable)
            try:
                log_report = build_failure_report(
                    artifact_dir=self.config.io.artifacts_dir,
                    artifact_id=artifact_id,
                    cycle_id=cycle_spec.id,
                    last_proposal=last_proposal,
                    last_validation=last_validation,
                    validation_history=validation_history,
                    config=self.config.model_dump() if hasattr(self.config, 'model_dump') else None,
                )
                # Attach the log report and a short narrative to the explanation
                failure_explanation["log_report"] = log_report
                # If the failure_explanation has no markdown, add the narrative as markdown
                if isinstance(failure_explanation, dict):
                    if not failure_explanation.get("markdown_report"):
                        failure_explanation["markdown_report"] = "".join(["# Failure Narrative\n", log_report.get("narrative", "")])
            except Exception as e:
                logger.debug(f"Could not build log-based failure report: {e}")

            results["explanation"] = failure_explanation
            
            # Persist failure explanation
            try:
                self.persistor.save_json(artifact_id, "explanation/failure_explanation.json", failure_explanation)
                # Save markdown report for human review
                if isinstance(failure_explanation, dict):
                    markdown = failure_explanation.get("markdown_report") or failure_explanation.get("explanation", "")
                    self.persistor.save_text(artifact_id, "explanation/failure_explanation.md", markdown)
                    # Also save the raw LLM response separately for reference
                    if failure_explanation.get("raw_llm_response"):
                        self.persistor.save_text(artifact_id, "explanation/llm_raw_response.txt", failure_explanation["raw_llm_response"])
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

        # Persist failed patches for analysis (at DEBUG level in logs, always in artifacts)
        if accumulated_reverted_files:
            try:
                # Build detailed failure records for analysis
                failed_patches = []
                for rf in accumulated_reverted_files:
                    if isinstance(rf, dict):
                        failed_patches.append({
                            "path": rf.get("path", "unknown"),
                            "reason": rf.get("reason", "unknown"),
                            "warnings": rf.get("warnings", []),
                            "original_patched_preview": rf.get("original_patched", "")[:1000] if rf.get("original_patched") else None,
                        })
                    elif hasattr(rf, "path"):
                        failed_patches.append({
                            "path": getattr(rf, "path", "unknown"),
                            "reason": getattr(rf, "reason", "unknown"),
                            "warnings": getattr(rf, "warnings", []),
                            "original_patched_preview": getattr(rf, "original_patched", "")[:1000] if getattr(rf, "original_patched", None) else None,
                        })
                
                if failed_patches:
                    self.persistor.persist_failed_patches(artifact_id, failed_patches)
                    logger.debug(f"Persisted {len(failed_patches)} failed patch records for analysis")
            except Exception as e:
                logger.warning(f"Failed to persist failed patches: {e}")

        # Save run metadata
        try:
            self.persistor.save_json(artifact_id, "run_metadata.json", {
                "artifact_id": artifact_id,
                "timestamp": int(time.time()),
                "duration_seconds": round(pipeline_elapsed, 1),
                "iterations": iteration,
                "approved": approved,
                "failed_patches_count": len(accumulated_reverted_files),
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
        
        if self.dry_run:
            logger.info("="*50)
            logger.info("DRY-RUN SUMMARY")
            logger.info("="*50)
            write_log = self.persistor.get_write_log()
            logger.info(f"Files that would be written: {len(write_log)}")
            total_bytes = sum(w.get("size", 0) for w in write_log)
            logger.info(f"Total data: {total_bytes:,} bytes")
            results["dry_run_summary"] = {
                "files_count": len(write_log),
                "total_bytes": total_bytes,
                "writes": write_log,
            }
        else:
            logger.info(f"Artifacts: {self.config.io.artifacts_dir}/{artifact_id}/")
        
        return results

    # =========================================================================
    # Directory Analysis Mode (Optional Pre-Pipeline Step)
    # =========================================================================
    
    def analyze_directory(
        self,
        project_dir: str,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_cycles: int = 50,
    ) -> Dict[str, Any]:
        """Analyze a project directory to discover cycles.
        
        This is an optional pre-pipeline step that scans a TypeScript/JavaScript
        project to automatically detect cyclic dependencies, which can then be
        processed through the refactoring pipeline.
        
        Args:
            project_dir: Path to the project root directory
            extensions: File extensions to analyze (default: ts, tsx, js, jsx)
            exclude_patterns: Patterns to exclude (default: node_modules, dist, etc.)
            max_cycles: Maximum number of cycles to detect (default: 50)
            
        Returns:
            Dict containing:
                - success: Whether analysis succeeded
                - graph: DependencyGraph of the project
                - cycles: List of DetectedCycle objects
                - cycle_specs: List of CycleSpec objects ready for pipeline
                - summary: Cycle detection summary
                
        Example:
            orchestrator = Orchestrator()
            result = orchestrator.analyze_directory("/path/to/project")
            
            if result["success"]:
                for spec in result["cycle_specs"]:
                    orchestrator.run_pipeline(spec)
        """
        logger.info("="*50)
        logger.info("DIRECTORY ANALYSIS MODE")
        logger.info("="*50)
        logger.info(f"Project: {project_dir}")
        logger.info(f"Extensions: {extensions or ['ts', 'tsx', 'js', 'jsx']}")
        logger.info(f"Exclude patterns: {len(exclude_patterns) if exclude_patterns else 7} patterns")
        logger.info(f"Max cycles: {max_cycles}")
        
        start_time = time.time()
        
        # Step 1: Analyze dependencies
        logger.info("-"*50)
        logger.info("Step 1: Analyzing project dependencies")
        logger.info("-"*50)
        
        analyzer = DependencyAnalyzerAgent()
        analysis_result = analyzer.run({
            "project_dir": project_dir,
            "extensions": extensions or ["ts", "tsx", "js", "jsx"],
            "exclude_patterns": exclude_patterns or [
                "node_modules", "dist", "build", ".git", 
                "__tests__", "*.test.*", "*.spec.*"
            ],
        })
        
        if analysis_result.status != "success":
            error_msg = analysis_result.output.get("error", "unknown error")
            logger.error(f"Dependency analysis failed: {error_msg}")
            return {
                "success": False,
                "error": f"Dependency analysis failed: {error_msg}",
                "graph": None,
                "cycles": [],
                "cycle_specs": [],
            }
        
        graph: DependencyGraph = analysis_result.output["graph"]
        logger.info(f"Found {len(graph.nodes)} files, {len(graph.edges)} dependencies")
        
        # Log tool used and timing
        tools = analysis_result.output.get("tools", {})
        analysis_time = analysis_result.output.get("analysis_time_seconds", 0)
        logger.debug(f"Analysis tools: madge={tools.get('madge_available')}, node={tools.get('node_available')}")
        logger.debug(f"Dependency analysis took {analysis_time:.1f}s")
        
        # Step 2: Detect cycles
        logger.info("-"*50)
        logger.info("Step 2: Detecting cycles")
        logger.info("-"*50)
        
        detector = CycleDetectorAgent(config={"max_cycles": max_cycles})
        detection_result = detector.run({
            "graph": graph,
            "project_dir": project_dir,
        })
        
        if detection_result.status != "success":
            error_msg = detection_result.output.get("error", "unknown error")
            logger.error(f"Cycle detection failed: {error_msg}")
            return {
                "success": False,
                "error": f"Cycle detection failed: {error_msg}",
                "graph": graph,
                "cycles": [],
                "cycle_specs": [],
            }
        
        cycles: List[DetectedCycle] = detection_result.output.get("cycles", [])
        cycle_specs: List[CycleSpec] = detection_result.output.get("cycle_specs", [])
        summary = detection_result.output.get("summary", {})
        
        # Log cycle spec conversion stats
        if len(cycles) != len(cycle_specs):
            logger.warning(f"Cycle to CycleSpec conversion: {len(cycles)} cycles -> {len(cycle_specs)} specs ({len(cycles) - len(cycle_specs)} failed)")
        
        elapsed = time.time() - start_time
        
        logger.info("="*50)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*50)
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Files analyzed: {len(graph.nodes)}")
        logger.info(f"Cycles found: {len(cycles)}")
        if summary:
            logger.info(f"  - Critical: {summary.get('critical', 0)}")
            logger.info(f"  - Major: {summary.get('major', 0)}")
            logger.info(f"  - Minor: {summary.get('minor', 0)}")
        logger.info(f"Cycle specs ready for pipeline: {len(cycle_specs)}")
        
        return {
            "success": True,
            "graph": graph,
            "cycles": cycles,
            "cycle_specs": cycle_specs,
            "summary": summary,
            "project_info": analysis_result.output.get("project_info"),
            "tools": analysis_result.output.get("tools"),
            "analysis_time_seconds": elapsed,
        }
    
    def run_full_analysis(
        self,
        project_dir: str,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_cycles: int = 50,
        priority: str = "severity_first",
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze project, detect cycles, and run refactoring pipeline on all.
        
        This is the complete automated workflow:
        1. Scan project for dependencies
        2. Detect all cycles
        3. Prioritize cycles
        4. Run refactoring pipeline on each cycle
        
        Args:
            project_dir: Path to the project root directory
            extensions: File extensions to analyze
            exclude_patterns: Patterns to exclude
            max_cycles: Maximum cycles to process
            priority: Prioritization strategy ("severity_first", "size_first")
            prompt: Optional prompt to pass to each pipeline run
            
        Returns:
            Dict with overall results including per-cycle results
            
        Example:
            orchestrator = Orchestrator()
            results = orchestrator.run_full_analysis("/path/to/project")
            
            print(f"Processed {results['cycles_processed']} cycles")
            print(f"Approved: {results['approved_count']}")
        """
        logger.info("="*50)
        logger.info("FULL ANALYSIS & REFACTORING MODE")
        logger.info("="*50)
        
        overall_start = time.time()
        
        # Step 1: Analyze directory
        analysis = self.analyze_directory(
            project_dir,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            max_cycles=max_cycles,
        )
        
        if not analysis["success"]:
            return {
                "success": False,
                "error": analysis.get("error", "Analysis failed"),
                "analysis": analysis,
                "pipeline_results": [],
            }
        
        cycle_specs = analysis["cycle_specs"]
        
        if not cycle_specs:
            logger.info("No cycles found - project is cycle-free!")
            return {
                "success": True,
                "cycles_found": 0,
                "cycles_processed": 0,
                "approved_count": 0,
                "analysis": analysis,
                "pipeline_results": [],
            }
        
        # Step 2: Prioritize cycles
        logger.info("-"*50)
        logger.info("Step 2: Prioritizing cycles")
        logger.info("-"*50)
        logger.debug(f"Priority strategy: {priority}")
        
        detector = CycleDetectorAgent()
        cycles = analysis["cycles"]
        prioritized = detector.get_prioritized_cycles(cycles, priority)
        
        # Map back to specs (maintain priority order)
        cycle_id_to_spec = {s.id: s for s in cycle_specs}
        ordered_specs = []
        for cycle in prioritized:
            if cycle.id in cycle_id_to_spec:
                ordered_specs.append(cycle_id_to_spec[cycle.id])
        
        logger.info(f"Prioritized {len(ordered_specs)} cycles for processing")
        if ordered_specs:
            top3 = ordered_specs[:3]
            logger.debug(f"Top 3 cycles: {[s.id for s in top3]}")
        
        logger.info(f"\nProcessing {len(ordered_specs)} cycles (priority: {priority})")
        
        # Step 3: Run pipeline on each cycle
        logger.info("-"*50)
        logger.info("Step 3: Running refactoring pipeline")
        logger.info("-"*50)
        
        pipeline_results = []
        approved_count = 0
        
        for i, spec in enumerate(ordered_specs):
            logger.info(f"\n{'='*50}")
            logger.info(f"CYCLE {i+1}/{len(ordered_specs)}: {spec.id}")
            logger.info(f"{'='*50}")
            
            try:
                result = self.run_pipeline(spec, prompt=prompt)
                pipeline_results.append({
                    "cycle_id": spec.id,
                    "status": result.get("status", "unknown"),
                    "iterations": result.get("iterations", 0),
                    "result": result,
                })
                
                if result.get("status") == "approved":
                    approved_count += 1
                    logger.info(f"✓ Cycle {spec.id} APPROVED")
                else:
                    logger.info(f"✗ Cycle {spec.id}: {result.get('status', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"Pipeline failed for {spec.id}: {e}")
                pipeline_results.append({
                    "cycle_id": spec.id,
                    "status": "error",
                    "error": str(e),
                })
        
        overall_elapsed = time.time() - overall_start
        
        logger.info("\n" + "="*50)
        logger.info("FULL ANALYSIS COMPLETE")
        logger.info("="*50)
        logger.info(f"Total time: {overall_elapsed:.1f}s")
        logger.info(f"Cycles found: {len(cycles)}")
        logger.info(f"Cycles processed: {len(pipeline_results)}")
        logger.info(f"Approved: {approved_count}")
        logger.info(f"Failed: {len(pipeline_results) - approved_count}")
        
        return {
            "success": True,
            "cycles_found": len(cycles),
            "cycles_processed": len(pipeline_results),
            "approved_count": approved_count,
            "failed_count": len(pipeline_results) - approved_count,
            "analysis": analysis,
            "pipeline_results": pipeline_results,
            "total_time_seconds": overall_elapsed,
        }
