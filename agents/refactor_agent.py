from typing import Dict, Any, List, Optional, Union
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
        rag_service=None,
        context_window: int = 4096,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
        self.query_builder = RAGQueryBuilder()
        self.context_window = context_window

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
        """Build prompt with strategy guidance, examples, and chain-of-thought.
        
        Uses context budget management to optimize for limited context window.
        """
        file_paths = cycle.get_file_paths()
        cycle_dict = cycle.model_dump()
        has_feedback = feedback is not None
        
        # Create context budget for this agent
        budget = create_budget_for_agent(
            "refactor", 
            total_tokens=self.context_window,
            has_feedback=has_feedback
        )
        
        # Get validation issues for file prioritization (if retry)
        validation_issues = None
        if feedback and feedback.issues:
            validation_issues = [{"path": i.path, "comment": i.comment} for i in feedback.issues]
        
        # Prioritize files based on cycle structure and validation issues
        files_data = [{"path": f.path, "content": f.content or ""} for f in cycle.files]
        file_priorities = prioritize_cycle_files(
            files_data,
            cycle_dict.get("graph", {}),
            validation_issues=validation_issues,
            total_char_budget=budget.get_char_budget(BudgetCategory.FILE_CONTENT),
        )
        
        # Build file snippets with priority-based budgets
        snippets = []
        total_file_tokens = 0
        file_token_budget = budget.get_token_budget(BudgetCategory.FILE_CONTENT)
        
        for f in cycle.files:
            if total_file_tokens >= file_token_budget:
                logger.debug(f"File budget exhausted, skipping remaining files")
                break
                
            content = f.content or ""
            file_budget = get_file_budget(f.path, file_priorities, self.max_file_chars)
            
            # Select snippet with file-specific budget
            snippet = select_relevant_snippet(content, f.path, cycle_dict, file_budget)
            snippet_tokens = TokenEstimator.estimate_tokens_for_file(snippet, f.path)
            
            # Check if we have room
            if total_file_tokens + snippet_tokens <= file_token_budget:
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
                total_file_tokens += snippet_tokens
            else:
                # Truncate to fit remaining budget
                remaining_tokens = file_token_budget - total_file_tokens
                truncated = truncate_to_token_budget(snippet, remaining_tokens)
                snippets.append(f"--- FILE: {f.path} ---\n{truncated}")
                total_file_tokens = file_token_budget
                break

        file_snippets = "\n\n".join(snippets) if snippets else ""
        budget.use_budget(BudgetCategory.FILE_CONTENT, total_file_tokens)
        logger.debug(f"File content: {total_file_tokens} tokens used across {len(snippets)} files")
        
        # Get RAG context with budget limit
        rag_token_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
        rag_context = self._get_rag_context(cycle, strategy, description)
        if rag_context:
            rag_context = truncate_to_token_budget(rag_context, rag_token_budget, "prose")
        
        # Get pattern example (fits in examples budget)
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

## Output Format - CHOOSE ONE:

### Option A: For SMALL files (< 100 lines) - Full content
```json
{
  "reasoning": "<brief explanation>",
  "strategy_used": "<interface_extraction|dependency_inversion|shared_module|other>",
  "patches": [
    {"path": "path/to/file.ext", "patched": "<COMPLETE file content>"}
  ]
}
```

### Option B: For LARGE files - Use SEARCH/REPLACE blocks (PREFERRED)
```json
{
  "reasoning": "<brief explanation>",
  "strategy_used": "<interface_extraction|dependency_inversion|shared_module|other>",
  "changes": [
    {
      "path": "path/to/file.ext",
      "search_replace": [
        {
          "search": "<exact text to find, 3-5 lines of context>",
          "replace": "<replacement text>"
        }
      ]
    }
  ]
}
```

### Option C: Use file markers with SEARCH/REPLACE
```
--- FILE: path/to/file.ext ---
<<<<<<< SEARCH
using OldNamespace;
=======
using NewNamespace;
>>>>>>> REPLACE

<<<<<<< SEARCH
public class Foo : IOldInterface
=======
public class Foo : INewInterface
>>>>>>> REPLACE
```

## Critical Requirements
1. For large files, use SEARCH/REPLACE blocks (Option B or C) - this prevents truncation
2. Include enough context in SEARCH blocks to be unique (3-5 lines before/after)
3. Ensure all brackets/braces are balanced in your output
4. Preserve existing functionality
5. DO NOT truncate your output - if file is too large, use SEARCH/REPLACE
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
        """Parse patches from JSON format in LLM response.
        
        Handles various LLM output formats:
        1. Clean JSON
        2. JSON wrapped in markdown code blocks
        3. JSON with embedded code blocks for file content
        """
        # Step 1: Strip markdown code block wrapper if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```
        text = text.strip()
        
        # Step 2: Try direct JSON parsing first
        try:
            data = json.loads(text)
            return self._extract_patches_from_data(data)
        except json.JSONDecodeError:
            pass
        
        # Step 3: Try to find JSON object with regex
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._extract_patches_from_data(data)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed: {e}")
        
        # Step 4: Try to extract patches with embedded code blocks
        # LLM sometimes outputs: "patched": "```csharp\n...code...\n```"
        # or even uses actual newlines in the JSON string
        patches = self._parse_json_with_code_blocks(text)
        if patches:
            return patches
        
        # Step 5: Try a more lenient extraction - find patches array directly
        patches = self._extract_patches_lenient(text)
        if patches:
            return patches
        
        logger.debug("All JSON parsing strategies failed")
        return []
    
    def _extract_patches_from_data(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract patches from parsed JSON data."""
        patches = []
        raw_patches = data.get("patches", [])
        
        if not raw_patches:
            logger.debug("JSON parsed but no 'patches' array found")
            return []
        
        for p in raw_patches:
            path = p.get("path")
            patched = p.get("patched")
            
            if not path:
                logger.warning("Patch missing 'path' field")
                continue
            if not patched:
                logger.warning(f"Patch for {path} missing 'patched' content")
                continue
            
            # Clean up the patched content (remove code block markers if present)
            patched = self._clean_code_content(patched)
                
            patches.append({
                "path": path,
                "patched": patched
            })
        
        return patches
    
    def _clean_code_content(self, content: str) -> str:
        """Remove code block markers from content."""
        content = content.strip()
        # Remove leading code block marker with optional language
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            else:
                content = content[3:]
        # Remove trailing code block marker
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    
    def _parse_json_with_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Parse JSON where patched content might have embedded code blocks.
        
        Some LLMs output:
        {
          "patches": [
            {
              "path": "file.cs",
              "patched": "```csharp
              ...actual code...
              ```"
            }
          ]
        }
        """
        patches = []
        
        # Find all path/patched pairs using a more flexible pattern
        # Look for "path": "..." followed eventually by "patched": "..."
        path_pattern = r'"path"\s*:\s*"([^"]+)"'
        
        # Find all paths first
        path_matches = list(re.finditer(path_pattern, text))
        
        for i, path_match in enumerate(path_matches):
            path = path_match.group(1)
            
            # Find the "patched" field after this path
            start_search = path_match.end()
            end_search = path_matches[i + 1].start() if i + 1 < len(path_matches) else len(text)
            
            section = text[start_search:end_search]
            
            # Look for patched content - might be a code block
            patched_match = re.search(r'"patched"\s*:\s*"', section)
            if patched_match:
                # Find where the value starts
                value_start = patched_match.end()
                # Now find the end - this is tricky because the content might have quotes
                # Look for the pattern that ends the JSON string
                # Could end with ", or "} or "] 
                content_section = section[value_start:]
                
                # Try to find the end by looking for unescaped quote followed by comma/bracket
                patched_content = self._extract_json_string_value(content_section)
                if patched_content:
                    patched_content = self._clean_code_content(patched_content)
                    patches.append({"path": path, "patched": patched_content})
        
        return patches
    
    def _extract_json_string_value(self, text: str) -> Optional[str]:
        """Extract a JSON string value, handling escaped characters."""
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            if char == '\\' and i + 1 < len(text):
                # Escaped character
                next_char = text[i + 1]
                if next_char == 'n':
                    result.append('\n')
                elif next_char == 't':
                    result.append('\t')
                elif next_char == 'r':
                    result.append('\r')
                elif next_char == '"':
                    result.append('"')
                elif next_char == '\\':
                    result.append('\\')
                else:
                    result.append(next_char)
                i += 2
            elif char == '"':
                # End of string
                return ''.join(result)
            else:
                result.append(char)
                i += 1
        
        # Didn't find closing quote - return what we have
        return ''.join(result) if result else None
    
    def _extract_patches_lenient(self, text: str) -> List[Dict[str, str]]:
        """Lenient extraction when JSON parsing fails.
        
        Looks for file paths and code content directly.
        """
        patches = []
        
        # Pattern: "path": "/some/path/file.cs" followed by code
        # Then look for code blocks or "patched": content
        
        # Find patterns like: "path": "...", "patched": <content>
        # where content might be malformed JSON but contains the code
        
        path_matches = re.findall(r'"path"\s*:\s*"([^"]+)"', text)
        
        for path in path_matches:
            # Try to find associated code block after the path mention
            path_pos = text.find(f'"path": "{path}"') 
            if path_pos == -1:
                path_pos = text.find(f'"path":"{path}"')
            if path_pos == -1:
                continue
            
            # Look for code block after this path
            remaining = text[path_pos:]
            
            # Look for code block markers
            code_block_match = re.search(r'```(?:csharp|cs|python|java|javascript|typescript)?\n([\s\S]*?)```', remaining)
            if code_block_match:
                patches.append({
                    "path": path,
                    "patched": code_block_match.group(1).strip()
                })
        
        return patches

    def _parse_marker_patches(self, text: str) -> List[Dict[str, str]]:
        """Parse file marker format, handling both full content and SEARCH/REPLACE blocks."""
        pattern = r"--- FILE: (.+?) ---\n"
        parts = re.split(pattern, text)
        patches = []
        if len(parts) < 3:
            return patches
        
        # drop leading preamble
        it = iter(parts[1:])
        for path, content in zip(it, it):
            path = path.strip()
            content = content.strip()
            
            # Check if content contains SEARCH/REPLACE blocks
            if "<<<<<<< SEARCH" in content and ">>>>>>> REPLACE" in content:
                patches.append({
                    "path": path,
                    "search_replace_blocks": content,  # Will be processed later
                    "patched": None  # Indicates we need to apply search/replace
                })
            else:
                patches.append({"path": path, "patched": content})
        
        return patches

    def _parse_search_replace_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse JSON format with search_replace changes instead of full patches."""
        try:
            # Strip markdown wrapper
            text = text.strip()
            if text.startswith("```"):
                first_newline = text.find("\n")
                text = text[first_newline + 1:] if first_newline != -1 else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            # Try to parse JSON
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            changes = data.get("changes", [])
            
            if not changes:
                return []
            
            result = []
            for change in changes:
                path = change.get("path")
                search_replace = change.get("search_replace", [])
                if path and search_replace:
                    result.append({
                        "path": path,
                        "search_replace": search_replace
                    })
            
            return result
        except Exception as e:
            logger.debug(f"Failed to parse search_replace JSON: {e}")
            return []

    def _apply_search_replace(self, original: str, search_replace_blocks: str) -> str:
        """Apply SEARCH/REPLACE blocks to original content.
        
        Format:
        <<<<<<< SEARCH
        old text
        =======
        new text
        >>>>>>> REPLACE
        """
        result = original
        
        # Normalize line endings for consistent matching
        result = result.replace('\r\n', '\n')
        search_replace_blocks = search_replace_blocks.replace('\r\n', '\n')
        
        # Parse SEARCH/REPLACE blocks
        pattern = r'<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE'
        matches = re.findall(pattern, search_replace_blocks)
        
        if not matches:
            logger.warning("No SEARCH/REPLACE blocks found in content")
            return original
        
        applied_count = 0
        for search_text, replace_text in matches:
            # Strategy 1: Exact match
            if search_text in result:
                result = result.replace(search_text, replace_text, 1)
                applied_count += 1
                logger.debug(f"Applied exact search/replace: {len(search_text)} chars -> {len(replace_text)} chars")
                continue
            
            # Strategy 2: Strip trailing whitespace from each line and try again
            search_lines = [line.rstrip() for line in search_text.split('\n')]
            result_lines = result.split('\n')
            result_lines_stripped = [line.rstrip() for line in result_lines]
            
            # Find the starting line
            found = False
            for i in range(len(result_lines_stripped) - len(search_lines) + 1):
                if result_lines_stripped[i:i + len(search_lines)] == search_lines:
                    # Found match - replace these lines
                    new_lines = result_lines[:i] + replace_text.split('\n') + result_lines[i + len(search_lines):]
                    result = '\n'.join(new_lines)
                    applied_count += 1
                    found = True
                    logger.debug(f"Applied whitespace-normalized search/replace at line {i}")
                    break
            
            if found:
                continue
            
            # Strategy 3: Anchor-based matching (first and last non-empty lines)
            non_empty_search = [l.strip() for l in search_lines if l.strip()]
            if len(non_empty_search) >= 2:
                first_anchor = non_empty_search[0]
                last_anchor = non_empty_search[-1]
                
                start_idx = None
                end_idx = None
                
                for i, line in enumerate(result_lines):
                    stripped = line.strip()
                    if stripped == first_anchor and start_idx is None:
                        start_idx = i
                    elif start_idx is not None and stripped == last_anchor:
                        end_idx = i
                        break
                
                if start_idx is not None and end_idx is not None and end_idx > start_idx:
                    new_lines = result_lines[:start_idx] + replace_text.split('\n') + result_lines[end_idx + 1:]
                    result = '\n'.join(new_lines)
                    applied_count += 1
                    logger.debug(f"Applied anchor-based search/replace at lines {start_idx}-{end_idx}")
                    continue
            
            logger.warning(f"Search text not found in file: {search_text[:100]}...")
        
        logger.info(f"Applied {applied_count}/{len(matches)} search/replace blocks")
        return result

    def _apply_search_replace_list(self, original: str, search_replace_list: List[Dict[str, str]]) -> str:
        """Apply a list of search/replace operations from JSON format."""
        result = original.replace('\r\n', '\n')  # Normalize line endings
        applied_count = 0
        
        for sr in search_replace_list:
            search_text = sr.get("search", "").replace('\r\n', '\n')
            replace_text = sr.get("replace", "").replace('\r\n', '\n')
            
            if not search_text:
                continue
            
            # Strategy 1: Exact match
            if search_text in result:
                result = result.replace(search_text, replace_text, 1)
                applied_count += 1
                logger.debug(f"Applied exact search/replace: {len(search_text)} chars -> {len(replace_text)} chars")
                continue
            
            # Strategy 2: Line-by-line with stripped whitespace
            search_lines = [line.rstrip() for line in search_text.split('\n')]
            result_lines = result.split('\n')
            result_lines_stripped = [line.rstrip() for line in result_lines]
            
            found = False
            for i in range(len(result_lines_stripped) - len(search_lines) + 1):
                if result_lines_stripped[i:i + len(search_lines)] == search_lines:
                    new_lines = result_lines[:i] + replace_text.split('\n') + result_lines[i + len(search_lines):]
                    result = '\n'.join(new_lines)
                    applied_count += 1
                    found = True
                    logger.debug(f"Applied whitespace-normalized search/replace at line {i}")
                    break
            
            if found:
                continue
            
            # Strategy 3: Anchor-based (first and last non-empty lines)
            non_empty_search = [l.strip() for l in search_lines if l.strip()]
            if len(non_empty_search) >= 2:
                first_anchor = non_empty_search[0]
                last_anchor = non_empty_search[-1]
                
                start_idx = None
                end_idx = None
                
                for i, line in enumerate(result_lines):
                    stripped = line.strip()
                    if stripped == first_anchor and start_idx is None:
                        start_idx = i
                    elif start_idx is not None and stripped == last_anchor:
                        end_idx = i
                        break
                
                if start_idx is not None and end_idx is not None and end_idx > start_idx:
                    new_lines = result_lines[:start_idx] + replace_text.split('\n') + result_lines[end_idx + 1:]
                    result = '\n'.join(new_lines)
                    applied_count += 1
                    logger.debug(f"Applied anchor-based search/replace at lines {start_idx}-{end_idx}")
                    continue
            
            logger.warning(f"Search text not found: {search_text[:80]}...")
        
        logger.info(f"Applied {applied_count}/{len(search_replace_list)} search/replace operations")
        return result

    def _infer_patches(self, llm_response: Any, cycle_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse patches from LLM response using multiple strategies.
        
        Returns list of dicts with either:
        - {"path": str, "patched": str} for full content patches
        - {"path": str, "search_replace_blocks": str} for marker-style S/R
        - {"path": str, "search_replace": List[Dict]} for JSON-style S/R
        """
        text = llm_response if isinstance(llm_response, str) else json.dumps(llm_response)

        # Try structured JSON first (handles both full patches and search_replace format)
        patches = self._parse_json_patches(text)
        if patches:
            logger.debug(f"Parsed {len(patches)} patches from JSON format")
            return patches

        # Try JSON with search_replace changes
        patches = self._parse_search_replace_json(text)
        if patches:
            logger.debug(f"Parsed {len(patches)} search/replace patches from JSON format")
            return patches

        # Try marker format (handles both full content and SEARCH/REPLACE blocks)
        patches = self._parse_marker_patches(text)
        if patches:
            logger.debug(f"Parsed {len(patches)} patches from marker format")
            return patches

        # Log the first part of response to help debug parsing issues
        logger.warning(f"Could not parse patches from LLM response. First 500 chars: {text[:500]}")
        
        # Nothing parsed: return empty => no-op
        return []

    def _make_unified_diff(self, original: str, patched: str, path: str) -> str:
        orig_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)
        diff = difflib.unified_diff(orig_lines, patched_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
        return "".join(diff)

    def _check_for_truncation(self, content: str, path: str) -> bool:
        """Check if content appears to be truncated (unbalanced brackets).
        
        Returns True if truncation is detected.
        """
        if not content:
            return False
        
        # Count brackets
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_parens = content.count('(')
        close_parens = content.count(')')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # Check for significant imbalance
        brace_diff = open_braces - close_braces
        paren_diff = open_parens - close_parens
        bracket_diff = open_brackets - close_brackets
        
        if abs(brace_diff) > 1:
            logger.warning(f"Possible truncation in {path}: unbalanced braces {{}} ({open_braces} open, {close_braces} close)")
            return True
        if abs(paren_diff) > 2:  # Allow more leeway for parentheses
            logger.warning(f"Possible truncation in {path}: unbalanced parentheses () ({open_parens} open, {close_parens} close)")
            return True
        if abs(bracket_diff) > 1:
            logger.warning(f"Possible truncation in {path}: unbalanced brackets [] ({open_brackets} open, {close_brackets} close)")
            return True
        
        # Check if file ends abruptly (common truncation patterns)
        content_stripped = content.rstrip()
        truncation_indicators = [
            # Incomplete statements
            content_stripped.endswith(','),
            content_stripped.endswith('('),
            content_stripped.endswith('{'),
            content_stripped.endswith('['),
            content_stripped.endswith(':'),
            # Incomplete strings
            content_stripped.count('"') % 2 != 0,
            content_stripped.count("'") % 2 != 0,
        ]
        
        if any(truncation_indicators):
            logger.warning(f"Possible truncation in {path}: file ends with incomplete construct")
            return True
        
        return False

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
            
            # Log what patches were inferred for debugging
            if inferred:
                inferred_paths = [p.get("path", "unknown") for p in inferred]
                logger.info(f"Inferred patches for paths: {inferred_paths}")
            else:
                logger.warning("No patches inferred from LLM response - check if LLM returned valid output")
            
            for f in cycle_spec.files:
                path = f.path
                original = f.content or ""
                
                # Try multiple matching strategies
                patched_entry = None
                
                # 1. Exact path match
                patched_entry = next((p for p in inferred if p.get("path") == path), None)
                
                if patched_entry is None:
                    # 2. Basename match (just the filename)
                    basename = path.split("/")[-1]
                    patched_entry = next((p for p in inferred if p.get("path", "").endswith(basename)), None)
                
                if patched_entry is None:
                    # 3. Partial path match (e.g., "Stores/ISensorStore.cs" matches "/full/path/Stores/ISensorStore.cs")
                    for p in inferred:
                        inferred_path = p.get("path", "")
                        if inferred_path and (path.endswith(inferred_path) or inferred_path.endswith(basename)):
                            patched_entry = p
                            break
                
                if patched_entry is None:
                    # 4. Case-insensitive basename match (for Windows paths vs Linux paths)
                    basename_lower = basename.lower()
                    patched_entry = next(
                        (p for p in inferred if p.get("path", "").lower().endswith(basename_lower)), 
                        None
                    )
                
                if patched_entry is None:
                    logger.warning(f"No patch found for {path} (basename: {basename})")

                # Determine patched content - handle both full patches and search/replace
                if patched_entry:
                    if patched_entry.get("patched"):
                        # Full patched content provided
                        patched = patched_entry.get("patched")
                    elif patched_entry.get("search_replace_blocks"):
                        # Marker-style SEARCH/REPLACE blocks
                        patched = self._apply_search_replace(
                            original, patched_entry.get("search_replace_blocks")
                        )
                        logger.debug(f"Applied marker-style search/replace to {path}")
                    elif patched_entry.get("search_replace"):
                        # JSON-style search_replace list
                        patched = self._apply_search_replace_list(
                            original, patched_entry.get("search_replace")
                        )
                        logger.debug(f"Applied JSON-style search/replace to {path}")
                    else:
                        # Entry exists but no content - keep original
                        patched = original
                else:
                    patched = original

                # Check for truncation - if detected, revert to original
                if patched != original and self._check_for_truncation(patched, path):
                    logger.warning(f"Truncation detected in {path}, reverting to original")
                    patched = original

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
