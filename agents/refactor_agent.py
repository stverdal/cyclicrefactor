from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
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
from utils.syntax_checker import (
    validate_code_block, check_truncation, check_introduced_issues,
    get_common_indent, normalize_line_endings, SyntaxIssue
)
from utils.compile_checker import CompileChecker, CompileResult
from utils.patch_parser import (
    parse_json_patches, parse_marker_patches, parse_search_replace_json,
    infer_patches, clean_code_content, extract_patches_from_data
)
from utils.patch_applier import (
    SearchReplaceResult, apply_search_replace_atomic, apply_search_replace_list_atomic,
    apply_with_partial_rollback, try_find_search_text, extract_line_hint
)
from utils.diff_utils import (
    make_unified_diff, looks_truncated, validate_patched_content,
    check_for_truncation, get_common_indent as diff_get_common_indent
)
from models.schemas import CycleSpec, CycleDescription, RefactorProposal, Patch, ValidationReport, RevertedFile
from config import RefactorConfig
import json
import difflib
import re

logger = get_logger("refactor")


@dataclass
class CompileCheckInfo:
    """Information about compile/lint check results."""
    checked: bool = False
    success: bool = True
    tool_used: str = ""
    error_count: int = 0
    warning_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class PatchProcessingResult:
    """Result of processing a single file patch."""
    path: str
    original: str
    patched: str
    diff: str
    status: str  # "unchanged", "applied", "partial", "failed", "reverted"
    warnings: List[str]
    confidence: float
    applied_blocks: int
    total_blocks: int
    revert_reason: str
    pre_validated: bool
    validation_issues: List[str]
    has_critical_error: bool  # True if critical syntax/validation error found
    original_patched: Optional[str]  # Content before revert (if reverted)
    compile_info: Optional[CompileCheckInfo] = None  # Compile/lint check results


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
        prompt_template_compact: str = None,
        max_file_chars: int = 4000,
        rag_service=None,
        context_window: int = 4096,
        refactor_config: Optional[RefactorConfig] = None,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.prompt_template_compact = prompt_template_compact
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
        self.query_builder = RAGQueryBuilder()
        self.context_window = context_window
        self.refactor_config = refactor_config or RefactorConfig()
        
        # Initialize compile checker based on config
        self.compile_checker = CompileChecker(
            enabled=self.refactor_config.compile_check,
            timeout=self.refactor_config.compile_check_timeout,
        )
        
        # Determine if compact prompts should be used
        self.use_compact_prompts = self._should_use_compact_prompts()
        if self.use_compact_prompts:
            logger.info(f"Using compact prompts (context_window={context_window}, compact_enabled={self.refactor_config.compact_prompts})")
    
    def _should_use_compact_prompts(self) -> bool:
        """Determine whether to use compact prompts based on config and context window."""
        # Explicit config takes precedence
        if self.refactor_config.compact_prompts:
            return True
        
        # Auto-detect based on context window size
        threshold = getattr(self.refactor_config, 'auto_compact_threshold', 8192)
        if self.context_window < threshold:
            logger.debug(f"Auto-enabling compact prompts: context_window={self.context_window} < threshold={threshold}")
            return True
        
        return False
    
    def _get_prompt_template(self) -> Optional[str]:
        """Get the appropriate prompt template based on compact mode."""
        if self.use_compact_prompts and self.prompt_template_compact:
            logger.debug("Using compact prompt template")
            return self.prompt_template_compact
        return self.prompt_template

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
        Adapts budget based on feedback type (syntax vs semantic errors).
        """
        file_paths = cycle.get_file_paths()
        cycle_dict = cycle.model_dump()
        has_feedback = feedback is not None
        
        # Analyze feedback to determine issue types
        has_syntax_errors = False
        has_cycle_issues = False
        syntax_error_files = set()
        
        if feedback and feedback.issues:
            for issue in feedback.issues:
                issue_type = getattr(issue, 'issue_type', 'semantic')
                if issue_type == 'syntax':
                    has_syntax_errors = True
                    if issue.path:
                        syntax_error_files.add(issue.path)
                elif issue_type == 'cycle':
                    has_cycle_issues = True
        
        # Log the feedback classification
        if has_feedback:
            logger.info(f"Feedback analysis: syntax_errors={has_syntax_errors}, cycle_issues={has_cycle_issues}, syntax_files={len(syntax_error_files)}")
        
        # Create context budget - adjust based on feedback type
        # If only syntax errors, skip RAG and give more room to files
        skip_rag = has_syntax_errors and not has_cycle_issues
        
        budget = create_budget_for_agent(
            "refactor", 
            total_tokens=self.context_window,
            has_feedback=has_feedback
        )
        
        # If we have syntax errors, reallocate RAG budget to file content
        if skip_rag:
            rag_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
            logger.info(f"Syntax errors detected - reallocating {rag_budget} RAG tokens to file content")
            # Zero out RAG and add to file content
            budget.allocations[BudgetCategory.RAG_CONTEXT] = 0
            budget.allocations[BudgetCategory.FILE_CONTENT] += rag_budget / self.context_window
        
        # Get validation issues for file prioritization (if retry)
        validation_issues = None
        if feedback and feedback.issues:
            validation_issues = [{"path": i.path, "comment": i.comment} for i in feedback.issues]
        
        # Prioritize files based on cycle structure, validation issues, and strategy
        files_data = [{"path": f.path, "content": f.content or ""} for f in cycle.files]
        file_priorities = prioritize_cycle_files(
            files_data,
            cycle_dict.get("graph", {}),
            validation_issues=validation_issues,
            total_char_budget=budget.get_char_budget(BudgetCategory.FILE_CONTENT),
            strategy_hint=strategy,
        )
        
        # If we have syntax errors, boost priority of files with errors
        if syntax_error_files:
            for fp in file_priorities:
                # Check if this file has syntax errors (match by basename)
                basename = fp.path.split('/')[-1].split('\\')[-1]
                for err_path in syntax_error_files:
                    err_basename = err_path.split('/')[-1].split('\\')[-1]
                    if basename == err_basename:
                        fp.priority_score = 1.0  # Maximum priority
                        fp.reasons = ["syntax_error_file"]  # Replace reasons list
                        logger.debug(f"Boosted priority for syntax error file: {fp.path}")
                        break
        
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
        
        # Get RAG context with budget limit (skip if we have syntax errors)
        rag_context = ""
        if skip_rag:
            logger.info("Skipping RAG context due to syntax errors - focusing on file content")
        else:
            rag_token_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
            rag_context = self._get_rag_context(cycle, strategy, description)
            if rag_context:
                rag_context = truncate_to_token_budget(rag_context, rag_token_budget, "prose")
        
        # Get pattern example (skip if we have syntax errors - focus on fixing the code)
        # Also skip for compact prompts to save tokens
        pattern_example = ""
        if not skip_rag and not self.use_compact_prompts:
            pattern_example = self._get_pattern_example(strategy)

        # Use compact template if available and appropriate
        template_path = self._get_prompt_template()
        
        if template_path:
            tpl = load_template(template_path)
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
        """Format validator feedback for retry prompt.
        
        Categorizes issues by type and provides focused instructions.
        Includes enriched retry context when available.
        Detects anti-patterns and provides specific "do not" guidance.
        """
        # Access enriched retry context directly from model fields
        iteration = getattr(feedback, 'iteration', 1)
        remaining_attempts = getattr(feedback, 'remaining_attempts', 0)
        previous_reverted = getattr(feedback, 'previous_reverted_files', [])
        previous_summary = getattr(feedback, 'previous_attempt_summary', '')
        failed_strategies = getattr(feedback, 'failed_strategies', [])
        
        # Categorize issues
        syntax_issues = []
        cycle_issues = []
        semantic_issues = []
        
        for issue in feedback.issues:
            issue_type = getattr(issue, 'issue_type', 'semantic')
            if issue_type == 'syntax':
                syntax_issues.append(issue)
            elif issue_type == 'cycle':
                cycle_issues.append(issue)
            else:
                semantic_issues.append(issue)
        
        # Detect anti-patterns from failure history
        anti_patterns = self._detect_anti_patterns(
            syntax_issues, cycle_issues, semantic_issues, 
            previous_reverted, failed_strategies
        )
        
        result = f"\n\n## ⚠️ Attempt {iteration} Failed - {remaining_attempts} attempt(s) remaining\n"
        
        # Show anti-patterns as prominent warnings (highest priority)
        if anti_patterns:
            result += "\n### 🚫 DO NOT REPEAT THESE MISTAKES:\n"
            for ap in anti_patterns:
                result += f"- ❌ {ap}\n"
            result += "\n"
        
        # Show previous attempt summary if available
        if previous_summary:
            result += f"\n**Previous attempt:** {previous_summary}\n"
        
        # Show what was reverted (important context for LLM)
        if previous_reverted:
            result += "\n### 🔄 Files Reverted in Previous Attempt:\n"
            result += "These files had errors and were reverted to original. Try a DIFFERENT approach:\n"
            for rf in previous_reverted[:5]:
                path = rf.get('path', rf.path if hasattr(rf, 'path') else 'unknown')
                reason = rf.get('reason', rf.reason if hasattr(rf, 'reason') else 'unknown')
                result += f"- **{path}**: {reason[:100]}\n"
        
        # Show failed strategies so LLM tries something different
        if failed_strategies:
            result += f"\n**Strategies already tried:** {', '.join(failed_strategies)}\n"
            result += "Please try a DIFFERENT approach.\n"
        
        # Syntax errors first (highest priority)
        if syntax_issues:
            result += "\n### 🔴 CRITICAL - Syntax Errors (Fix First!):\n"
            result += "Your previous output had truncated or malformed code. Use SEARCH/REPLACE format for large files.\n\n"
            for issue in syntax_issues[:5]:  # Limit to avoid overwhelming
                line_info = f" (line {issue.line})" if issue.line else ""
                result += f"- **{issue.path}**{line_info}: {issue.comment}\n"
            if len(syntax_issues) > 5:
                result += f"- ... and {len(syntax_issues) - 5} more syntax issues\n"
            result += "\n**To fix:** Use targeted SEARCH/REPLACE blocks instead of outputting entire files.\n"
        
        # Cycle issues
        if cycle_issues:
            result += "\n### 🟠 Cycle Not Broken:\n"
            for issue in cycle_issues[:3]:
                result += f"- {issue.comment}\n"
            if len(cycle_issues) > 3:
                result += f"- ... and {len(cycle_issues) - 3} more cycle issues\n"
        
        # Other semantic issues
        if semantic_issues:
            result += "\n### 🟡 Other Issues:\n"
            for issue in semantic_issues[:5]:
                line_info = f" (line {issue.line})" if issue.line else ""
                result += f"- **{issue.path}**{line_info}: {issue.comment}\n"
        
        if feedback.suggestions:
            result += "\n**Suggestions:**\n"
            for suggestion in feedback.suggestions[:3]:
                result += f"- {suggestion}\n"
        
        result += "\nPlease fix these issues in your new proposal. Use a DIFFERENT strategy if previous attempts failed.\n"
        return result
    
    def _detect_anti_patterns(
        self,
        syntax_issues: List,
        cycle_issues: List,
        semantic_issues: List,
        previous_reverted: List[Dict],
        failed_strategies: List[str]
    ) -> List[str]:
        """Detect anti-patterns from failure history and return specific warnings.
        
        Analyzes the patterns in failures to provide actionable "do not" guidance.
        """
        anti_patterns = []
        
        # Pattern 1: Full file output causing syntax errors
        if syntax_issues:
            has_truncation = any(
                "truncat" in str(getattr(i, 'comment', '')).lower() or
                "incomplete" in str(getattr(i, 'comment', '')).lower() or
                "unexpected EOF" in str(getattr(i, 'comment', '')) or
                "unterminated" in str(getattr(i, 'comment', '')).lower()
                for i in syntax_issues
            )
            if has_truncation:
                anti_patterns.append(
                    "DO NOT output full file contents - the output gets truncated. "
                    "Use SEARCH/REPLACE blocks to make targeted changes."
                )
        
        # Pattern 2: Same files keep failing
        if previous_reverted:
            reverted_paths = {rf.get('path', '') for rf in previous_reverted if rf.get('path')}
            current_failed_paths = {
                getattr(i, 'path', '') 
                for i in (syntax_issues + semantic_issues) 
                if getattr(i, 'path', '')
            }
            repeated_failures = reverted_paths & current_failed_paths
            if repeated_failures:
                files = ', '.join(list(repeated_failures)[:3])
                anti_patterns.append(
                    f"DO NOT use the same approach for {files} - it failed before. "
                    "Try a different file or a different type of change."
                )
        
        # Pattern 3: Cycle not broken despite changes
        if cycle_issues and "refactoring approach (cycle not broken)" in failed_strategies:
            anti_patterns.append(
                "DO NOT move code without breaking the dependency. "
                "Consider: 1) Extract an interface, 2) Use dependency injection, "
                "3) Create a mediator/event system, 4) Move the shared code to a third module."
            )
        
        # Pattern 4: Multiple syntax errors suggests wrong output format
        if len(syntax_issues) >= 3:
            anti_patterns.append(
                "DO NOT output file contents directly - always use the SEARCH/REPLACE format: "
                "<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE"
            )
        
        # Pattern 5: Same edge keeps appearing in cycle issues
        cycle_edges = []
        for issue in cycle_issues:
            comment = str(getattr(issue, 'comment', ''))
            # Try to extract edge info like "A -> B still exists"
            if "->" in comment:
                cycle_edges.append(comment)
        if len(cycle_edges) >= 2:
            anti_patterns.append(
                "The same dependency edge keeps appearing. "
                "Focus on REMOVING or INVERTING this specific dependency, "
                "not just moving code around."
            )
        
        return anti_patterns
    
    # -------------------------------------------------------------------------
    # Patch Parsing Methods (delegated to utils.patch_parser)
    # -------------------------------------------------------------------------
    
    def _parse_json_patches(self, text: str) -> List[Dict[str, str]]:
        """Parse patches from JSON format. Delegates to utils.patch_parser."""
        return parse_json_patches(text)
    
    def _extract_patches_from_data(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract patches from parsed JSON data. Delegates to utils.patch_parser."""
        return extract_patches_from_data(data)
    
    def _clean_code_content(self, content: str) -> str:
        """Remove code block markers. Delegates to utils.patch_parser."""
        return clean_code_content(content)
    
    def _parse_marker_patches(self, text: str) -> List[Dict[str, str]]:
        """Parse file marker format. Delegates to utils.patch_parser."""
        return parse_marker_patches(text)
    
    def _parse_search_replace_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse JSON format with search_replace changes. Delegates to utils.patch_parser."""
        return parse_search_replace_json(text)
    
    # -------------------------------------------------------------------------
    # Patch Application Methods (delegated to utils.patch_applier)
    # -------------------------------------------------------------------------
    
    def _extract_line_hint(self, text: str) -> Optional[int]:
        """Extract line number hint from text. Delegates to utils.patch_applier."""
        return extract_line_hint(text)
    
    def _try_find_search_text(
        self, 
        search_text: str, 
        content_lines: List[str],
        content_lines_stripped: List[str],
        line_hint: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int], str, float]:
        """Try to find search text in content. Delegates to utils.patch_applier."""
        result = try_find_search_text(search_text, content_lines, content_lines_stripped, line_hint)
        return result.start, result.end, result.strategy, result.confidence
    
    def _apply_search_replace_atomic(
        self, 
        original: str, 
        search_replace_blocks: str,
        min_confidence: Optional[float] = None
    ) -> SearchReplaceResult:
        """Apply SEARCH/REPLACE blocks atomically. Delegates to utils.patch_applier."""
        if min_confidence is None:
            min_confidence = self.refactor_config.min_match_confidence
        return apply_search_replace_atomic(
            original, 
            search_replace_blocks,
            min_confidence=min_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_search_replace_list_atomic(
        self, 
        original: str, 
        search_replace_list: List[Dict[str, str]],
        min_confidence: Optional[float] = None
    ) -> SearchReplaceResult:
        """Apply a list of search/replace operations atomically. Delegates to utils.patch_applier."""
        if min_confidence is None:
            min_confidence = self.refactor_config.min_match_confidence
        return apply_search_replace_list_atomic(
            original, 
            search_replace_list,
            min_confidence=min_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_with_partial_rollback(
        self,
        original: str,
        search_replace_blocks: str,
        path: str,
        original_content: str = None
    ) -> SearchReplaceResult:
        """Apply with partial rollback on failure. Delegates to utils.patch_applier."""
        return apply_with_partial_rollback(
            original,
            search_replace_blocks,
            path,
            original_content,
            min_confidence=self.refactor_config.min_match_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_search_replace(self, original: str, search_replace_blocks: str) -> Tuple[str, List[str]]:
        """Apply SEARCH/REPLACE blocks (simple interface). Delegates to utils.patch_applier."""
        result = self._apply_search_replace_atomic(original, search_replace_blocks)
        return result.content, result.warnings
    
    def _apply_search_replace_list(self, original: str, search_replace_list: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        """Apply a list of search/replace operations (simple interface). Delegates to utils.patch_applier."""
        result = self._apply_search_replace_list_atomic(original, search_replace_list)
        return result.content, result.warnings
    
    # -------------------------------------------------------------------------
    # Diff and Validation Methods (delegated to utils.diff_utils)
    # -------------------------------------------------------------------------
    
    def _looks_truncated(self, text: str) -> bool:
        """Check if text appears truncated. Delegates to utils.diff_utils."""
        return looks_truncated(text)
    
    def _get_common_indent(self, lines: List[str]) -> str:
        """Get common leading whitespace. Delegates to utils.diff_utils."""
        return diff_get_common_indent(lines)
    
    def _validate_patched_content(self, original: str, patched: str, path: str) -> List[str]:
        """Validate patched content. Delegates to utils.diff_utils."""
        return validate_patched_content(original, patched, path)

    def _infer_patches(self, llm_response: Any, cycle_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse patches from LLM response using multiple strategies.
        
        Delegates to utils.patch_parser.infer_patches.
        """
        return infer_patches(llm_response, cycle_files)

    def _make_unified_diff(self, original: str, patched: str, path: str) -> str:
        """Generate a unified diff. Delegates to utils.diff_utils."""
        return make_unified_diff(original, patched, path)

    def _check_for_truncation(self, content: str, path: str) -> bool:
        """Check if content appears to be truncated. Delegates to utils.diff_utils."""
        return check_for_truncation(content, path)
    
    def _log_failed_patch_details(
        self,
        path: str,
        original: str,
        patched_entry: Optional[Dict[str, Any]],
        warnings: List[str],
        reason: str,
    ) -> None:
        """Log detailed information about a failed patch for debugging.
        
        This logs at DEBUG level to help diagnose why patches fail to apply.
        Information is logged in a structured way for later analysis.
        """
        logger.debug("=" * 60)
        logger.debug(f"FAILED PATCH DETAILS: {path}")
        logger.debug("=" * 60)
        logger.debug(f"Failure reason: {reason}")
        
        # Log original file info
        original_lines = original.split('\n') if original else []
        logger.debug(f"Original file: {len(original_lines)} lines, {len(original)} chars")
        
        # Log first/last lines of original for context
        if original_lines:
            logger.debug(f"Original first 3 lines:")
            for i, line in enumerate(original_lines[:3]):
                logger.debug(f"  {i+1}: {line[:100]}")
            if len(original_lines) > 6:
                logger.debug(f"  ... ({len(original_lines) - 6} lines omitted) ...")
            logger.debug(f"Original last 3 lines:")
            for i, line in enumerate(original_lines[-3:]):
                logger.debug(f"  {len(original_lines) - 2 + i}: {line[:100]}")
        
        # Log what the LLM tried to do
        if patched_entry:
            logger.debug("-" * 40)
            logger.debug("LLM patch entry:")
            
            if patched_entry.get("patched"):
                patched_content = patched_entry.get("patched", "")
                patched_lines = patched_content.split('\n')
                logger.debug(f"  Type: full patched content ({len(patched_lines)} lines)")
                logger.debug(f"  First 3 lines:")
                for line in patched_lines[:3]:
                    logger.debug(f"    {line[:100]}")
            
            elif patched_entry.get("search_replace"):
                sr_list = patched_entry.get("search_replace", [])
                logger.debug(f"  Type: JSON search_replace ({len(sr_list)} operations)")
                for i, sr in enumerate(sr_list[:5]):  # Log up to 5 operations
                    search_text = sr.get("search", "")
                    replace_text = sr.get("replace", "")
                    search_lines = search_text.split('\n')
                    replace_lines = replace_text.split('\n')
                    logger.debug(f"  Operation {i+1}:")
                    logger.debug(f"    Search ({len(search_lines)} lines):")
                    for line in search_lines[:3]:
                        logger.debug(f"      | {line[:80]}")
                    if len(search_lines) > 3:
                        logger.debug(f"      ... ({len(search_lines) - 3} more lines)")
                    logger.debug(f"    Replace ({len(replace_lines)} lines):")
                    for line in replace_lines[:3]:
                        logger.debug(f"      | {line[:80]}")
                    if len(replace_lines) > 3:
                        logger.debug(f"      ... ({len(replace_lines) - 3} more lines)")
                if len(sr_list) > 5:
                    logger.debug(f"  ... and {len(sr_list) - 5} more operations")
            
            elif patched_entry.get("search_replace_blocks"):
                blocks = patched_entry.get("search_replace_blocks", "")
                logger.debug(f"  Type: marker-style SEARCH/REPLACE ({len(blocks)} chars)")
                # Log first 500 chars of blocks
                logger.debug(f"  Content preview:")
                for line in blocks[:500].split('\n'):
                    logger.debug(f"    {line[:100]}")
                if len(blocks) > 500:
                    logger.debug(f"    ... ({len(blocks) - 500} more chars)")
        else:
            logger.debug("No patch entry found for this file")
        
        # Log warnings
        if warnings:
            logger.debug("-" * 40)
            logger.debug(f"Warnings ({len(warnings)}):")
            for w in warnings[:10]:
                logger.debug(f"  - {w}")
            if len(warnings) > 10:
                logger.debug(f"  ... and {len(warnings) - 10} more warnings")
        
        logger.debug("=" * 60)

    def _process_single_file_patch(
        self,
        path: str,
        original: str,
        inferred: List[Dict[str, Any]],
    ) -> PatchProcessingResult:
        """Process a single file's patch and return the result.
        
        This method handles:
        - Finding the matching patch from inferred list
        - Applying SEARCH/REPLACE operations
        - Pre-validating patched content for critical errors
        
        Args:
            path: File path being patched
            original: Original file content
            inferred: List of inferred patches from LLM
            
        Returns:
            PatchProcessingResult with all patch details
        """
        # Try multiple matching strategies to find the patch
        patched_entry = None
        basename = path.split("/")[-1]
        
        # 1. Exact path match
        patched_entry = next((p for p in inferred if p.get("path") == path), None)
        
        if patched_entry is None:
            # 2. Basename match
            patched_entry = next((p for p in inferred if p.get("path", "").endswith(basename)), None)
        
        if patched_entry is None:
            # 3. Partial path match
            for p in inferred:
                inferred_path = p.get("path", "")
                if inferred_path and (path.endswith(inferred_path) or inferred_path.endswith(basename)):
                    patched_entry = p
                    break
        
        if patched_entry is None:
            # 4. Case-insensitive basename match
            basename_lower = basename.lower()
            patched_entry = next(
                (p for p in inferred if p.get("path", "").lower().endswith(basename_lower)), 
                None
            )
        
        if patched_entry is None:
            logger.warning(f"No patch found for {path} (basename: {basename})")
        
        # Initialize tracking variables
        patch_status = "unchanged"
        patch_warnings: List[str] = []
        patch_confidence = 1.0
        applied_blocks = 0
        total_blocks = 0
        revert_reason = ""
        has_critical_error = False
        original_patched = None
        patched = original
        
        # Determine patched content
        if patched_entry:
            if patched_entry.get("patched"):
                # Full patched content provided
                patched = patched_entry.get("patched")
                patch_status = "applied"
            elif patched_entry.get("search_replace_blocks"):
                # Marker-style SEARCH/REPLACE blocks
                sr_result = self._apply_with_partial_rollback(
                    original, 
                    patched_entry.get("search_replace_blocks"),
                    path,
                    original
                )
                patched = sr_result.content
                patch_warnings = sr_result.warnings
                patch_confidence = sr_result.confidence
                applied_blocks = sr_result.applied_count
                total_blocks = sr_result.total_count
                
                if sr_result.applied_count == 0:
                    patch_status = "failed"
                    revert_reason = "No SEARCH/REPLACE blocks could be applied"
                    has_critical_error = True
                    # Log detailed failure info at DEBUG level
                    self._log_failed_patch_details(
                        path, original, patched_entry, sr_result.warnings, revert_reason
                    )
                elif not sr_result.is_atomic:
                    patch_status = "partial"
                else:
                    patch_status = "applied"
                
                logger.debug(f"Applied marker-style search/replace to {path}: {applied_blocks}/{total_blocks}")
            elif patched_entry.get("search_replace"):
                # JSON-style search_replace list
                sr_result = self._apply_search_replace_list_atomic(
                    original, patched_entry.get("search_replace")
                )
                patched = sr_result.content
                patch_warnings = sr_result.warnings
                patch_confidence = sr_result.confidence
                applied_blocks = sr_result.applied_count
                total_blocks = sr_result.total_count
                
                if sr_result.applied_count == 0:
                    patch_status = "failed"
                    revert_reason = "No search/replace operations could be applied"
                    has_critical_error = True
                    # Log detailed failure info at DEBUG level
                    self._log_failed_patch_details(
                        path, original, patched_entry, sr_result.warnings, revert_reason
                    )
                else:
                    patch_status = "applied"
                
                logger.debug(f"Applied JSON-style search/replace to {path}: {applied_blocks}/{total_blocks}")
        
        # Log warnings
        for warning in patch_warnings:
            logger.warning(f"S/R Warning [{path}]: {warning}")
        
        # Pre-validate patched content
        pre_validated = False
        validation_issues: List[str] = []
        
        if patched != original:
            validation_result = validate_code_block(patched, path, original)
            pre_validated = True
            validation_issues = [i.message for i in validation_result.issues]
            
            for issue in validation_result.issues:
                if issue.severity == "critical":
                    logger.error(f"Patch Validation [{path}]: {issue.message}")
                else:
                    logger.warning(f"Patch Validation [{path}]: {issue.message}")
            
            # Check for critical issues
            if validation_result.has_critical:
                has_critical_error = True
                revert_reason = "; ".join(i.message for i in validation_result.issues if i.severity == "critical")
                original_patched = patched  # Save what we tried
                # Log detailed failure info at DEBUG level
                self._log_failed_patch_details(
                    path, original, patched_entry, patch_warnings + validation_issues, revert_reason
                )
        
        # Compile/lint check (if enabled and no critical errors yet)
        compile_info = CompileCheckInfo()
        
        if patched != original and self.refactor_config.compile_check and not has_critical_error:
            compile_result = self.compile_checker.check_file(path, patched)
            compile_info = CompileCheckInfo(
                checked=True,
                success=compile_result.success,
                tool_used=compile_result.tool_used,
                error_count=compile_result.error_count,
                warning_count=compile_result.warning_count,
                errors=[str(e) for e in compile_result.errors],
            )
            
            if compile_result.tool_available:
                logger.info(f"Compile check [{path}]: {compile_result.summary()}")
                
                if not compile_result.success:
                    for error in compile_result.errors:
                        logger.error(f"Compile Error [{path}]: {error}")
                        validation_issues.append(f"Compile: {error}")
                    
                    # Mark as critical if configured to revert on compile errors
                    if self.refactor_config.revert_on_compile_error:
                        has_critical_error = True
                        revert_reason = f"Compile check failed: {compile_result.error_count} errors"
                        original_patched = patched
                
                for warning in compile_result.warnings[:5]:  # Limit warnings
                    logger.warning(f"Compile Warning [{path}]: {warning}")
                    patch_warnings.append(f"Compile: {warning}")
            else:
                logger.debug(f"Compile check skipped for {path}: {compile_result.tool_used} not available")
        
        # Generate diff
        diff = self._make_unified_diff(original, patched, path) if patched != original else ""
        
        return PatchProcessingResult(
            path=path,
            original=original,
            patched=patched,
            diff=diff,
            status=patch_status,
            warnings=patch_warnings,
            confidence=patch_confidence,
            applied_blocks=applied_blocks,
            total_blocks=total_blocks,
            revert_reason=revert_reason,
            pre_validated=pre_validated,
            validation_issues=validation_issues,
            has_critical_error=has_critical_error,
            original_patched=original_patched,
            compile_info=compile_info,
        )

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
            # Log what patches were inferred for debugging
            if inferred:
                inferred_paths = [p.get("path", "unknown") for p in inferred]
                logger.info(f"Inferred patches for paths: {inferred_paths}")
            else:
                logger.warning("No patches inferred from LLM response - check if LLM returned valid output")
            
            # Phase 1: Process all files and collect results (without committing)
            processing_results: List[PatchProcessingResult] = []
            
            for f in cycle_spec.files:
                result = self._process_single_file_patch(
                    path=f.path,
                    original=f.content or "",
                    inferred=inferred,
                )
                processing_results.append(result)
            
            # Phase 2: Check if atomic proposal mode should revert everything
            atomic_proposal = self.refactor_config.atomic_proposal
            revert_on_critical = self.refactor_config.revert_on_any_critical
            
            has_any_critical = any(r.has_critical_error for r in processing_results)
            has_any_failed = any(r.status == "failed" for r in processing_results)
            
            revert_all = False
            revert_all_reason = ""
            
            if atomic_proposal:
                if has_any_critical and revert_on_critical:
                    critical_files = [r.path for r in processing_results if r.has_critical_error]
                    revert_all = True
                    revert_all_reason = f"Atomic proposal mode: critical errors in {critical_files}"
                    logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to critical errors in: {critical_files}")
                elif has_any_failed:
                    failed_files = [r.path for r in processing_results if r.status == "failed"]
                    revert_all = True
                    revert_all_reason = f"Atomic proposal mode: patch application failed in {failed_files}"
                    logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to failures in: {failed_files}")
            
            # Phase 3: Build final patches list
            patches_out = []
            reverted_files = []
            files_changed = 0
            
            for result in processing_results:
                if revert_all:
                    # Revert everything - use original content
                    if result.patched != result.original:
                        reverted_files.append(RevertedFile(
                            path=result.path,
                            reason=revert_all_reason,
                            warnings=result.warnings,
                            original_patched=result.patched[:2000] if result.patched else None,
                        ))
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.original,  # Reverted to original
                        diff="",
                        status="reverted" if result.patched != result.original else "unchanged",
                        warnings=result.warnings + [revert_all_reason] if result.warnings else [revert_all_reason],
                        confidence=result.confidence,
                        applied_blocks=0,
                        total_blocks=result.total_blocks,
                        revert_reason=revert_all_reason if result.patched != result.original else "",
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))
                elif result.has_critical_error:
                    # Individual file revert (non-atomic mode)
                    logger.error(f"Reverting {result.path} due to critical errors")
                    reverted_files.append(RevertedFile(
                        path=result.path,
                        reason=result.revert_reason,
                        warnings=result.warnings,
                        original_patched=result.original_patched[:2000] if result.original_patched else None,
                    ))
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.original,  # Reverted
                        diff="",
                        status="reverted",
                        warnings=result.warnings,
                        confidence=result.confidence,
                        applied_blocks=0,
                        total_blocks=result.total_blocks,
                        revert_reason=result.revert_reason,
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))
                else:
                    # Normal case - apply the patch
                    if result.diff:
                        files_changed += 1
                        logger.debug(f"File changed: {result.path} ({len(result.diff)} chars diff)")
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.patched,
                        diff=result.diff,
                        status=result.status,
                        warnings=result.warnings,
                        confidence=result.confidence,
                        applied_blocks=result.applied_blocks,
                        total_blocks=result.total_blocks,
                        revert_reason=result.revert_reason,
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))

            logger.info(f"RefactorAgent completed: {files_changed}/{len(patches_out)} files changed, {len(reverted_files)} reverted (atomic={atomic_proposal})")
            proposal = RefactorProposal(
                patches=patches_out,
                reverted_files=reverted_files,
                rationale=f"LLM proposal using strategy: {strategy or 'auto-selected'}",
                llm_response=llm_response if isinstance(llm_response, str) else json.dumps(llm_response),
            )
            return AgentResult(status="success", output=proposal.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
