from typing import Dict, Any, List, Optional, Union
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, QueryIntent, CycleAnalysis
from models.schemas import CycleSpec, CycleDescription, RefactorProposal, Patch, ValidationReport
import json
import difflib
import re

logger = get_logger("refactor")


class RefactorAgent(Agent):
    """Proposes code patches to break cyclic dependencies.
    
    This agent:
    1. Extracts strategy hints from the Describer's output
    2. Queries RAG for implementation guidance (using conceptual terms)
    3. Includes mini-examples of refactoring patterns in prompts
    4. Uses chain-of-thought reasoning for complex cycles
    """
    
    name = "refactor"
    version = "0.7"

    # Mini-examples of refactoring patterns (language-agnostic concepts)
    PATTERN_EXAMPLES = {
        "interface_extraction": """
**Interface Extraction Pattern**
Before:
  ClassA imports ClassB (uses methods directly)
  ClassB imports ClassA (uses methods directly)
  
After:
  Create IClassB interface with methods ClassA needs
  ClassB implements IClassB
  ClassA depends on IClassB (not ClassB)
  Cycle broken: ClassB no longer needs to import ClassA
""",
        "dependency_inversion": """
**Dependency Inversion Pattern**
Before:
  HighLevelModule imports LowLevelModule directly
  
After:
  Create IService interface in HighLevelModule's layer
  LowLevelModule implements IService
  HighLevelModule depends on IService abstraction
  Dependency direction inverted
""",
        "shared_module": """
**Shared Module Extraction Pattern**
Before:
  ModuleA imports ModuleB (for shared utility)
  ModuleB imports ModuleA (for shared utility)
  
After:
  Create SharedModule with common functionality
  ModuleA imports SharedModule
  ModuleB imports SharedModule
  ModuleA and ModuleB no longer import each other
""",
        "mediator": """
**Mediator Pattern**
Before:
  ComponentA imports ComponentB (direct communication)
  ComponentB imports ComponentA (direct communication)
  
After:
  Create Mediator class
  ComponentA and ComponentB both depend on Mediator
  Mediator coordinates communication between them
  No direct dependencies between A and B
""",
    }

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
        self.query_builder = RAGQueryBuilder()

    def _extract_strategy_from_description(self, description: CycleDescription) -> Optional[str]:
        """Extract the recommended strategy from the Describer's output."""
        text = description.text.lower()
        
        strategy_keywords = {
            "interface_extraction": ["interface extraction", "extract interface", "create interface", "iservice"],
            "dependency_inversion": ["dependency inversion", "invert dependency", "inversion of control"],
            "shared_module": ["shared module", "common module", "extract shared", "move to common"],
            "mediator": ["mediator", "coordinator", "event bus"],
        }
        
        for strategy, keywords in strategy_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    logger.info(f"Detected strategy hint from description: {strategy}")
                    return strategy
        
        # Default based on common patterns in text
        if "bidirectional" in text or "mutual" in text:
            return "interface_extraction"
        if "layer" in text or "violation" in text:
            return "dependency_inversion"
        
        return None
    
    def _get_rag_context(self, cycle: CycleSpec, strategy: Optional[str], description: CycleDescription) -> str:
        """Retrieve implementation-focused context from RAG."""
        if self.rag_service is None:
            logger.debug("RAG service not available, skipping context retrieval")
            return ""
        
        try:
            cycle_dict = cycle.model_dump()
            
            # Build implementation-focused queries
            queries = self.query_builder.build_queries_for_cycle(
                cycle_dict, 
                QueryIntent.IMPLEMENT,
                description.text  # Pass description as hints
            )
            
            # Add strategy-specific query if we know the strategy
            if strategy:
                strategy_queries = {
                    "interface_extraction": "how to extract interface break dependency example",
                    "dependency_inversion": "implementing dependency inversion principle example",
                    "shared_module": "extract common module reduce coupling example",
                    "mediator": "mediator pattern implementation example",
                }
                if strategy in strategy_queries:
                    queries.insert(0, strategy_queries[strategy])
            
            all_results = []
            seen_content = set()
            
            for query in queries[:3]:  # Limit queries
                logger.info(f"RAG Query: '{query}'")
                logger.info("Purpose: Find implementation guidance for refactoring")
                
                results = self.rag_service.query_with_scores(query, k=2)
                
                if results:
                    for doc, score in results:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_results.append((doc, score))
                            source = doc.metadata.get('source_file', 'unknown')
                            logger.info(f"  Retrieved: {source} (score: {score:.3f})")
            
            if all_results:
                logger.info(f"RAG total: {len(all_results)} unique document(s) for implementation")
                all_results.sort(key=lambda x: x[1])
                
                context_parts = []
                for doc, score in all_results[:3]:
                    source = doc.metadata.get('source_file', 'unknown')
                    context_parts.append(f"[{source}]\n{doc.page_content[:500]}")
                
                return "\n\n---\n\n".join(context_parts)
            else:
                logger.info("RAG: No implementation guidance found")
                
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")
        
        return ""

    def _get_pattern_example(self, strategy: Optional[str]) -> str:
        """Get a mini-example for the recommended strategy."""
        if strategy and strategy in self.PATTERN_EXAMPLES:
            return self.PATTERN_EXAMPLES[strategy]
        
        # Return general examples if no specific strategy
        return """
**Common Refactoring Patterns:**

1. Interface Extraction: Create an interface that one module implements,
   and the other module depends on the interface instead of the concrete class.

2. Dependency Inversion: Ensure high-level modules don't depend on low-level modules.
   Both should depend on abstractions.

3. Extract Shared Module: Move common code to a new module that both can import,
   eliminating direct dependencies between the original modules.
"""

    def _build_prompt(self, cycle: CycleSpec, description: CycleDescription) -> str:
        """Basic prompt builder (kept for compatibility)."""
        return self._build_prompt_with_strategy(cycle, description, None, None)

    def _build_prompt_with_strategy(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        feedback: Optional[ValidationReport] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """Build prompt with strategy guidance, examples, and chain-of-thought."""
        file_paths = cycle.get_file_paths()

        # Prepare file content snippets
        snippets = []
        cycle_dict = cycle.model_dump()
        for f in cycle.files:
            content = f.content or ""
            snippet = select_relevant_snippet(content, f.path, cycle_dict, self.max_file_chars)
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")

        file_snippets = "\n\n".join(snippets) if snippets else ""
        
        # Get RAG context with strategy focus
        rag_context = self._get_rag_context(cycle, strategy, description)
        
        # Get pattern example
        pattern_example = self._get_pattern_example(strategy)

        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                file_snippets=file_snippets,
                rag_context=rag_context,
                pattern_example=pattern_example,
                strategy=strategy or "not specified",
            )
            # Append snippets if template didn't include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            
            # Append feedback if provided
            if feedback:
                result += self._format_feedback(feedback)
            return result

        # Default prompt with chain-of-thought structure
        prompt = f"""You are a refactoring expert. Break the cyclic dependency described below.

## Cycle Information
- ID: {cycle.id}
- Nodes: {', '.join(cycle.graph.nodes)}
- Edges: {json.dumps(cycle.graph.edges)}
- Files: {', '.join(file_paths)}

## Problem Description (from analysis)
{description.text[:2000]}

## Recommended Strategy: {strategy or "Choose the most appropriate"}

{pattern_example}

## Source Code
{file_snippets}
"""

        if rag_context:
            prompt += f"""
## Reference (from architecture literature)
{rag_context[:1500]}
"""

        if feedback:
            prompt += self._format_feedback(feedback)

        prompt += """
## IMPORTANT: Think Step-by-Step

Before writing code, reason through:
1. IDENTIFY: Which specific import/reference creates the problematic edge?
2. STRATEGY: Which pattern applies? (interface extraction, dependency inversion, shared module)
3. PLAN: What minimal changes break the cycle without breaking functionality?
4. IMPLEMENT: Write the patches

## Output Format
Return a JSON object with your patches:
```json
{
  "reasoning": "<brief explanation of your approach>",
  "strategy_used": "<interface_extraction|dependency_inversion|shared_module|other>",
  "patches": [
    {
      "path": "path/to/file.ext",
      "patched": "<COMPLETE file content after refactoring>"
    }
  ],
  "notes": "<any additional notes>"
}
```

OR use file markers:
```
--- FILE: path/to/file.ext ---
<COMPLETE file content after refactoring>
```

## Critical Requirements
1. Include COMPLETE file content (not just changes)
2. Ensure all imports are correct after changes
3. Preserve existing functionality
4. Remove or invert the problematic dependency
"""
        return prompt

    def _format_feedback(self, feedback: ValidationReport) -> str:
        """Format validator feedback for retry prompt."""
        result = "\n\n## ⚠️ Previous Attempt Failed - Address These Issues:\n"
        
        for issue in feedback.issues:
            line_info = f" (line {issue.line})" if issue.line else ""
            result += f"- **{issue.path}**{line_info}: {issue.comment}\n"
        
        if feedback.suggestions:
            result += "\n**Suggestions:**\n"
            for suggestion in feedback.suggestions:
                result += f"- {suggestion}\n"
        
        result += "\nPlease fix these issues in your new proposal.\n"
        return result

    def _parse_json_patches(self, text: str) -> List[Dict[str, str]]:
        try:
            data = json.loads(text)
            patches = []
            for p in data.get("patches", []):
                patches.append({
                    "path": p.get("path"),
                    "patched": p.get("patched")
                })
            return patches
        except Exception:
            return []

    def _parse_marker_patches(self, text: str) -> List[Dict[str, str]]:
        # Split by marker lines like '--- FILE: path ---'
        pattern = r"--- FILE: (.+?) ---\n"
        parts = re.split(pattern, text)
        # re.split will produce [pre, path1, content1, path2, content2, ...]
        patches = []
        if len(parts) < 3:
            return patches
        # drop leading preamble
        it = iter(parts[1:])
        for path, content in zip(it, it):
            patches.append({"path": path.strip(), "patched": content.strip()})
        return patches

    def _infer_patches(self, llm_response: Any, cycle_files: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # Try structured JSON first
        text = llm_response if isinstance(llm_response, str) else json.dumps(llm_response)

        patches = self._parse_json_patches(text)
        if patches:
            return patches

        patches = self._parse_marker_patches(text)
        if patches:
            return patches

        # Nothing parsed: return empty => no-op
        return []

    def _make_unified_diff(self, original: str, patched: str, path: str) -> str:
        orig_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)
        diff = difflib.unified_diff(orig_lines, patched_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
        return "".join(diff)

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        validator_feedback: Optional[Union[ValidationReport, Dict[str, Any]]] = None,
        prompt: str = None,
    ) -> AgentResult:
        """Propose patches to break the cycle using strategy-aware refactoring.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            validator_feedback: Optional ValidationReport for retry loop.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with RefactorProposal output.
        """
        logger.info(f"RefactorAgent.run() starting, has_feedback={validator_feedback is not None}")

        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            logger.error("Missing description input")
            return AgentResult(status="error", output=None, logs="Missing description")
        if isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if validator_feedback is not None and isinstance(validator_feedback, dict):
            validator_feedback = ValidationReport.model_validate(validator_feedback)
            logger.info(f"Retry with feedback: {len(validator_feedback.issues)} issues, {len(validator_feedback.suggestions)} suggestions")

        # Extract strategy hint from description
        strategy = self._extract_strategy_from_description(description)
        if strategy:
            logger.info(f"Using strategy: {strategy}")
        else:
            logger.info("No specific strategy detected, will use general approach")

        # Build strategy-aware prompt
        prompt_text = self._build_prompt_with_strategy(
            cycle_spec, description, validator_feedback, strategy
        )
        logger.debug(f"Built refactor prompt with {len(prompt_text)} chars")

        # If no LLM available just return original files as no-op patches
        if self.llm is None:
            logger.info("No LLM provided, returning no-op patches (original files unchanged)")
            patches = [
                Patch(path=f.path, original=f.content, patched=f.content, diff="")
                for f in cycle_spec.files
            ]
            proposal = RefactorProposal(patches=patches, rationale="No-op (no LLM provided)", llm_response=None)
            return AgentResult(status="success", output=proposal.model_dump())

        try:
            logger.info("Calling LLM for refactor proposal")
            llm_response = call_llm(self.llm, prompt_text)
            logger.debug(f"LLM response received, length: {len(str(llm_response))} chars")

            # Convert files to dict for _infer_patches compatibility
            files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
            inferred = self._infer_patches(llm_response, files_dict)
            logger.debug(f"Inferred {len(inferred)} patches from LLM response")

            # Try to extract strategy from response
            try:
                text = llm_response if isinstance(llm_response, str) else json.dumps(llm_response)
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "strategy_used" in parsed:
                        logger.info(f"LLM used strategy: {parsed['strategy_used']}")
                    if "reasoning" in parsed:
                        logger.debug(f"LLM reasoning: {parsed['reasoning'][:200]}...")
            except Exception:
                pass

            # Build final patches list merging originals with patched content
            patches_out = []
            files_changed = 0
            for f in cycle_spec.files:
                path = f.path
                original = f.content or ""
                patched_entry = next((p for p in inferred if p.get("path") == path), None)
                if patched_entry is None:
                    # try basename match
                    basename = path.split("/")[-1]
                    patched_entry = next((p for p in inferred if p.get("path") == basename), None)

                patched = patched_entry.get("patched") if patched_entry else original

                diff = self._make_unified_diff(original, patched, path) if patched != original else ""
                if diff:
                    files_changed += 1
                    logger.debug(f"File changed: {path} ({len(diff)} chars diff)")

                patches_out.append(Patch(path=path, original=original, patched=patched, diff=diff))

            logger.info(f"RefactorAgent completed: {files_changed}/{len(patches_out)} files changed")
            proposal = RefactorProposal(
                patches=patches_out,
                rationale=f"LLM proposal using strategy: {strategy or 'auto-selected'}",
                llm_response=llm_response if isinstance(llm_response, str) else json.dumps(llm_response),
            )
            return AgentResult(status="success", output=proposal.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
