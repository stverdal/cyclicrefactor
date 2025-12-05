from typing import Dict, Any, Optional, Union
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from models.schemas import CycleSpec, CycleDescription
import json

logger = get_logger("describer")


class DescriberAgent(Agent):
    name = "describer"
    version = "0.5"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        max_file_chars: int = 4000,
        rag_service=None
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service

    def _get_rag_context(self, cycle: CycleSpec) -> str:
        """Retrieve relevant context from RAG for the cycle."""
        if self.rag_service is None:
            logger.debug("RAG service not available, skipping context retrieval")
            return ""
        
        try:
            # Build a query from the cycle context
            # We search for cyclic dependency patterns related to the specific nodes
            nodes = ", ".join(cycle.graph.nodes[:5])  # Limit to first 5 nodes
            query = f"cyclic dependency refactoring {nodes}"
            
            logger.info(f"RAG Query: '{query}'")
            logger.info("Purpose: Find patterns/examples for analyzing this type of cyclic dependency")
            
            # Use query_with_scores to get relevance information
            results = self.rag_service.query_with_scores(query, k=3)
            
            if results:
                logger.info(f"RAG Results: {len(results)} document(s) retrieved")
                for i, (doc, score) in enumerate(results, 1):
                    source = doc.metadata.get('source_file', 'unknown')
                    preview = doc.page_content[:100].replace('\n', ' ').strip()
                    logger.info(f"  [{i}] {source} (score: {score:.3f})")
                    logger.debug(f"      Preview: {preview}...")
                
                # Format context from documents
                context = self.rag_service.format_context(
                    [doc for doc, _ in results],
                    max_length=2000
                )
                logger.debug(f"RAG context formatted: {len(context)} chars")
                return context
            else:
                logger.info("RAG Results: No relevant documents found")
                
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")
        
        return ""

    def _build_prompt(self, cycle: CycleSpec) -> str:
        # Try to format the provided template, otherwise create a default prompt
        file_paths = cycle.get_file_paths()

        # Prepare file content snippets by selecting relevant regions (imports,
        # defs, and symbol mentions) rather than the file head so prompts stay
        # concise and focused. Each snippet is truncated to `max_file_chars`.
        snippets = []
        # Convert CycleSpec to dict for snippet selector compatibility
        cycle_dict = cycle.model_dump()
        for f in cycle.files:
            content = f.content or ""
            snippet = select_relevant_snippet(content, f.path, cycle_dict, self.max_file_chars)
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")

        file_snippets = "\n\n".join(snippets) if snippets else ""
        
        # Get RAG context for reference materials
        rag_context = self._get_rag_context(cycle)

        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                file_snippets=file_snippets,
                rag_context=rag_context,
            )
            # Append snippets if template didn't already include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            # Append RAG context if template didn't include it
            if rag_context and "{rag_context}" not in tpl:
                result = result + "\n\n--- REFERENCE MATERIALS ---\n" + rag_context
            return result

        base = f"Please describe the cyclic dependency for id={cycle.id}. Graph: {json.dumps(cycle.graph.model_dump())}. Affected files: {', '.join(file_paths)}."

        if file_snippets:
            base += "\n\n" + file_snippets
            
        if rag_context:
            base += "\n\n--- REFERENCE MATERIALS ---\n" + rag_context

        return base

    def run(self, cycle_spec: Union[CycleSpec, Dict[str, Any]], prompt: str = None) -> AgentResult:
        """Describe the cyclic dependency.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with CycleDescription output.
        """
        logger.info(f"DescriberAgent.run() starting for cycle_id={getattr(cycle_spec, 'id', cycle_spec.get('id', 'unknown')) if isinstance(cycle_spec, dict) else cycle_spec.id}")

        # Convert dict to CycleSpec if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
            logger.debug("Converted input dict to CycleSpec model")

        logger.debug(f"Cycle has {len(cycle_spec.files)} files, {len(cycle_spec.graph.nodes)} nodes")
        prompt_text = self._build_prompt(cycle_spec)
        logger.debug(f"Built prompt with {len(prompt_text)} chars")

        if self.llm is None:
            # Fallback: simple deterministic description
            logger.info("No LLM provided, using fallback deterministic description")
            nodes = cycle_spec.graph.nodes
            edges = cycle_spec.graph.edges
            text = f"Cyclic dependency involving {len(nodes)} node(s): {', '.join(nodes)}. Edges: {edges}"
            description = CycleDescription(text=text, highlights=[])
            logger.debug(f"Fallback description: {text[:100]}...")
            return AgentResult(status="success", output=description.model_dump())

        try:
            logger.info("Calling LLM for cycle description")
            response = call_llm(self.llm, prompt_text)
            text = response if isinstance(response, str) else json.dumps(response)
            logger.debug(f"LLM response length: {len(text)} chars")
            description = CycleDescription(text=text, highlights=[])
            logger.info("DescriberAgent completed successfully")
            return AgentResult(status="success", output=description.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
