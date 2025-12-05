from typing import Dict, Any, Optional, Union, List
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, QueryIntent, CycleAnalysis
from utils.context_budget import (
    ContextBudget, BudgetCategory, TokenEstimator,
    prioritize_cycle_files, get_file_budget, create_budget_for_agent,
    truncate_to_token_budget
)
from models.schemas import CycleSpec, CycleDescription
import json

logger = get_logger("describer")


class DescriberAgent(Agent):
    """Analyzes and describes cyclic dependencies with strategy recommendations.
    
    This agent:
    1. Classifies the cycle type (bidirectional, transitive, layer violation, etc.)
    2. Queries RAG for relevant architectural concepts (not code-specific)
    3. Provides actionable strategy recommendations for the Refactor agent
    """
    
    name = "describer"
    version = "0.7"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        max_file_chars: int = 4000,
        rag_service=None,
        context_window: int = 4096,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
        self.query_builder = RAGQueryBuilder()
        self.context_window = context_window

    def _analyze_cycle(self, cycle: CycleSpec) -> CycleAnalysis:
        """Analyze the cycle to classify its type and characteristics."""
        cycle_dict = cycle.model_dump()
        return self.query_builder.analyze_cycle(cycle_dict)

    def _get_rag_context(self, cycle: CycleSpec, analysis: CycleAnalysis) -> str:
        """Retrieve relevant context from RAG using conceptual queries."""
        if self.rag_service is None:
            logger.debug("RAG service not available, skipping context retrieval")
            return ""
        
        try:
            cycle_dict = cycle.model_dump()
            
            # Build conceptual queries (NOT including file names)
            understanding_queries = self.query_builder.build_queries_for_cycle(
                cycle_dict, QueryIntent.UNDERSTAND
            )
            strategy_queries = self.query_builder.build_queries_for_cycle(
                cycle_dict, QueryIntent.STRATEGIZE
            )
            
            # Combine and deduplicate
            all_queries = understanding_queries[:2] + strategy_queries[:2]
            
            all_results = []
            seen_content = set()
            
            for query in all_queries:
                logger.info(f"RAG Query: '{query}'")
                
                results = self.rag_service.query_with_scores(query, k=2)
                
                if results:
                    for doc, score in results:
                        # Deduplicate by content hash
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_results.append((doc, score, query))
                            source = doc.metadata.get('source_file', 'unknown')
                            logger.info(f"  Retrieved: {source} (score: {score:.3f})")
            
            if all_results:
                logger.info(f"RAG total: {len(all_results)} unique document(s) retrieved")
                
                # Sort by score (lower is better for cosine distance)
                all_results.sort(key=lambda x: x[1])
                
                # Format top results
                context_parts = []
                for doc, score, query in all_results[:4]:
                    source = doc.metadata.get('source_file', 'unknown')
                    content = doc.page_content[:600]
                    context_parts.append(f"[{source}]\n{content}")
                
                return "\n\n---\n\n".join(context_parts)
            else:
                logger.info("RAG: No relevant documents found")
                
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")
        
        return ""

    def _build_analysis_context(self, analysis: CycleAnalysis) -> str:
        """Build a context string from the cycle analysis."""
        parts = [
            f"Cycle Type: {analysis.cycle_type.value}",
            f"Node Count: {analysis.node_count}",
            f"Dominant Pattern: {analysis.dominant_pattern}",
        ]
        
        if analysis.layer_violation:
            parts.append(f"Layer Violation Detected: {analysis.layer_violation}")
        if analysis.has_inheritance:
            parts.append("Inheritance Relationships: Yes")
        if analysis.has_interface:
            parts.append("Existing Interfaces: Yes")
        
        return "\n".join(parts)

    def _build_prompt(self, cycle: CycleSpec, analysis: CycleAnalysis) -> str:
        """Build prompt with cycle analysis and strategy guidance.
        
        Uses context budget management to optimize for limited context window.
        """
        file_paths = cycle.get_file_paths()
        cycle_dict = cycle.model_dump()
        
        # Create context budget for this agent
        budget = create_budget_for_agent("describer", total_tokens=self.context_window)
        
        # Prioritize files based on cycle structure
        files_data = [{"path": f.path, "content": f.content or ""} for f in cycle.files]
        file_priorities = prioritize_cycle_files(
            files_data,
            cycle_dict.get("graph", {}),
            total_char_budget=budget.get_char_budget(BudgetCategory.FILE_CONTENT),
        )
        
        # Build file snippets with priority-based budgets
        snippets = []
        total_file_tokens = 0
        file_token_budget = budget.get_token_budget(BudgetCategory.FILE_CONTENT)
        
        for f in cycle.files:
            if total_file_tokens >= file_token_budget:
                break
                
            content = f.content or ""
            file_budget = get_file_budget(f.path, file_priorities, self.max_file_chars)
            snippet = select_relevant_snippet(content, f.path, cycle_dict, file_budget)
            snippet_tokens = TokenEstimator.estimate_tokens_for_file(snippet, f.path)
            
            if total_file_tokens + snippet_tokens <= file_token_budget:
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
                total_file_tokens += snippet_tokens
            else:
                remaining_tokens = file_token_budget - total_file_tokens
                truncated = truncate_to_token_budget(snippet, remaining_tokens)
                snippets.append(f"--- FILE: {f.path} ---\n{truncated}")
                break

        file_snippets = "\n\n".join(snippets) if snippets else ""
        logger.debug(f"File content: {total_file_tokens} tokens used across {len(snippets)} files")
        
        # Get RAG context with budget limit
        rag_token_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
        rag_context = self._get_rag_context(cycle, analysis)
        if rag_context:
            rag_context = truncate_to_token_budget(rag_context, rag_token_budget, "prose")
        
        # Build analysis context
        analysis_context = self._build_analysis_context(analysis)
        
        # Build recommended strategies based on cycle type
        strategy_hints = self._get_strategy_hints(analysis)

        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                file_snippets=file_snippets,
                rag_context=rag_context,
                cycle_analysis=analysis_context,
                strategy_hints=strategy_hints,
            )
            # Append snippets if template didn't already include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            # Append RAG context if template didn't include it
            if rag_context and "{rag_context}" not in tpl:
                result = result + "\n\n--- REFERENCE MATERIALS ---\n" + rag_context
            return result

        # Default prompt with analysis and strategies
        prompt = f"""You are analyzing a cyclic dependency to provide actionable recommendations.

## Cycle Information
- ID: {cycle.id}
- Graph: {json.dumps(cycle.graph.model_dump())}
- Affected files: {', '.join(file_paths)}

## Automated Cycle Analysis
{analysis_context}

## Recommended Strategies (based on cycle characteristics)
{strategy_hints}

## Source Code
{file_snippets}
"""

        if rag_context:
            prompt += f"""
## Reference Materials (from architecture literature)
{rag_context}
"""

        prompt += """
## Your Task
Analyze this cycle and provide:
1. A clear description of what creates the cycle
2. Why this cycle is problematic (maintainability, testability, etc.)
3. Specific strategy recommendation for breaking it
4. Concrete first steps the Refactor agent should take

## Output Format (JSON)
{
  "text": "<description of the cycle and its problems>",
  "cycle_type": "<bidirectional|transitive|layer_violation|inheritance>",
  "recommended_strategy": "<interface_extraction|dependency_inversion|shared_module|mediator>",
  "strategy_rationale": "<why this strategy fits>",
  "first_steps": ["<specific actionable steps>"],
  "highlights": [{"path": "...", "explanation": "..."}]
}
"""
        return prompt

    def _get_strategy_hints(self, analysis: CycleAnalysis) -> str:
        """Generate strategy hints based on cycle analysis."""
        hints = []
        
        if analysis.cycle_type.value == "bidirectional":
            hints.append("""
**Interface Extraction** (Recommended for bidirectional cycles):
- Extract an interface from one of the two classes
- Have the other class depend on the interface, not the concrete class
- Example: If A↔B, create IA, have A implement IA, B depends on IA only
""")
        
        if analysis.layer_violation:
            hints.append(f"""
**Dependency Inversion** (Recommended for layer violations like {analysis.layer_violation}):
- Lower layers should not depend on higher layers
- Introduce an interface in the lower layer
- Higher layer provides implementation, lower layer uses abstraction
""")
        
        if analysis.has_inheritance:
            hints.append("""
**Composition Over Inheritance**:
- If the cycle involves inheritance, consider using composition instead
- Extract shared behavior into a separate class that both can use
""")
        
        if analysis.node_count > 3:
            hints.append("""
**Shared Module Extraction** (Recommended for multi-node cycles):
- Identify common functionality used by multiple modules
- Extract to a new shared module with no dependencies on the cycle members
- Update all cycle members to depend on shared module instead of each other
""")
        
        if not hints:
            hints.append("""
**General Strategies**:
1. Interface Extraction: Create abstraction between tightly coupled modules
2. Dependency Inversion: Invert the direction of problematic dependencies
3. Mediator Pattern: Introduce a coordinator that manages interactions
""")
        
        return "\n".join(hints)

    def run(self, cycle_spec: Union[CycleSpec, Dict[str, Any]], prompt: str = None) -> AgentResult:
        """Describe the cyclic dependency with classification and strategy recommendations.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with CycleDescription output including strategy recommendations.
        """
        logger.info(f"DescriberAgent.run() starting for cycle_id={getattr(cycle_spec, 'id', cycle_spec.get('id', 'unknown')) if isinstance(cycle_spec, dict) else cycle_spec.id}")

        # Convert dict to CycleSpec if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
            logger.debug("Converted input dict to CycleSpec model")

        logger.debug(f"Cycle has {len(cycle_spec.files)} files, {len(cycle_spec.graph.nodes)} nodes")
        
        # Analyze the cycle first
        analysis = self._analyze_cycle(cycle_spec)
        logger.info(f"Cycle classified as: {analysis.cycle_type.value}, pattern: {analysis.dominant_pattern}")
        
        # Build prompt with analysis
        prompt_text = self._build_prompt(cycle_spec, analysis)
        logger.debug(f"Built prompt with {len(prompt_text)} chars")

        if self.llm is None:
            # Fallback: use analysis for deterministic description
            logger.info("No LLM provided, using analysis-based fallback description")
            nodes = cycle_spec.graph.nodes
            edges = cycle_spec.graph.edges
            
            strategy_hints = self._get_strategy_hints(analysis)
            
            text = f"""Cyclic dependency analysis:
            
Type: {analysis.cycle_type.value}
Nodes: {', '.join(nodes)}
Edges: {edges}
Pattern: {analysis.dominant_pattern}
{"Layer Violation: " + analysis.layer_violation if analysis.layer_violation else ""}

{strategy_hints}
"""
            description = CycleDescription(text=text, highlights=[])
            logger.debug(f"Fallback description generated")
            return AgentResult(status="success", output=description.model_dump())

        try:
            logger.info("Calling LLM for cycle description")
            response = call_llm(self.llm, prompt_text)
            text = response if isinstance(response, str) else json.dumps(response)
            logger.debug(f"LLM response length: {len(text)} chars")
            
            # Try to parse structured response
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    # Enrich with strategy if present
                    if "recommended_strategy" in parsed:
                        logger.info(f"LLM recommended strategy: {parsed.get('recommended_strategy')}")
                    text = parsed.get("text", text)
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Could not parse structured response: {e}")
            
            description = CycleDescription(text=text, highlights=[])
            logger.info("DescriberAgent completed successfully")
            return AgentResult(status="success", output=description.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
