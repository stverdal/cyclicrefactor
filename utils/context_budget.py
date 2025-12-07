"""Context budget management for LLM prompts.

This module provides utilities for managing context window limits when
building prompts for LLMs with limited context (e.g., 4096 tokens).

Key features:
1. Token estimation (character-based heuristic, no external dependencies)
2. Global budget allocation across prompt components
3. Priority-based content selection
4. Cycle-aware file prioritization
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

from utils.logging import get_logger

logger = get_logger("context_budget")


# =============================================================================
# Token Estimation
# =============================================================================

class TokenEstimator:
    """Estimate token counts without external dependencies.
    
    Uses character-based heuristics tuned for code content.
    Different languages have different token densities.
    """
    
    # Approximate chars per token by content type
    # Based on empirical observations with common LLMs
    CHARS_PER_TOKEN = {
        "code_python": 3.5,      # Python is relatively verbose
        "code_csharp": 3.2,      # C# has longer keywords
        "code_java": 3.2,        # Similar to C#
        "code_javascript": 3.4,  # JS/TS
        "code_generic": 3.3,     # Default for code
        "prose": 4.0,            # English prose
        "json": 3.0,             # JSON is token-heavy (brackets, quotes)
        "diff": 3.5,             # Unified diffs
    }
    
    @classmethod
    def estimate_tokens(cls, text: str, content_type: str = "code_generic") -> int:
        """Estimate token count for text.
        
        Args:
            text: The text to estimate tokens for
            content_type: Type of content (affects estimation ratio)
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        chars_per_token = cls.CHARS_PER_TOKEN.get(content_type, 3.5)
        
        # Adjust for whitespace (whitespace is often merged with adjacent tokens)
        # Count significant characters more heavily
        whitespace_count = len(re.findall(r'\s', text))
        non_whitespace = len(text) - whitespace_count
        
        # Whitespace contributes less to token count
        effective_chars = non_whitespace + (whitespace_count * 0.3)
        
        return int(effective_chars / chars_per_token)
    
    @classmethod
    def estimate_tokens_for_file(cls, content: str, path: str) -> int:
        """Estimate tokens for a file based on its extension."""
        path_lower = path.lower()
        
        if path_lower.endswith('.py'):
            content_type = "code_python"
        elif path_lower.endswith('.cs'):
            content_type = "code_csharp"
        elif path_lower.endswith(('.java', '.kt')):
            content_type = "code_java"
        elif path_lower.endswith(('.js', '.ts', '.jsx', '.tsx')):
            content_type = "code_javascript"
        elif path_lower.endswith('.json'):
            content_type = "json"
        elif path_lower.endswith(('.md', '.txt', '.rst')):
            content_type = "prose"
        else:
            content_type = "code_generic"
        
        return cls.estimate_tokens(content, content_type)
    
    @classmethod
    def tokens_to_chars(cls, tokens: int, content_type: str = "code_generic") -> int:
        """Convert token budget to approximate character budget."""
        chars_per_token = cls.CHARS_PER_TOKEN.get(content_type, 3.5)
        return int(tokens * chars_per_token)


# =============================================================================
# Budget Allocation
# =============================================================================

class BudgetCategory(Enum):
    """Categories for context budget allocation."""
    SYSTEM_PROMPT = "system_prompt"      # Fixed instructions
    CYCLE_INFO = "cycle_info"            # Graph, nodes, edges
    FILE_CONTENT = "file_content"        # Source code snippets
    RAG_CONTEXT = "rag_context"          # Retrieved documents
    FEEDBACK = "feedback"                # Validator feedback on retry
    EXAMPLES = "examples"                # Pattern examples
    OUTPUT_RESERVE = "output_reserve"    # Reserved for LLM output


@dataclass
class BudgetAllocation:
    """Represents allocated budget for each category."""
    category: BudgetCategory
    tokens: int
    priority: int  # Higher = more important, gets budget first
    flexible: bool = True  # Can give up unused budget
    
    def to_chars(self, content_type: str = "code_generic") -> int:
        """Convert token budget to character budget."""
        return TokenEstimator.tokens_to_chars(self.tokens, content_type)


@dataclass
class ContextBudget:
    """Manages context budget allocation for a prompt.
    
    Usage:
        budget = ContextBudget(total_tokens=4096)
        budget.allocate_for_refactor()
        
        file_chars = budget.get_char_budget(BudgetCategory.FILE_CONTENT)
        rag_chars = budget.get_char_budget(BudgetCategory.RAG_CONTEXT)
    """
    total_tokens: int = 4096
    allocations: Dict[BudgetCategory, BudgetAllocation] = field(default_factory=dict)
    used: Dict[BudgetCategory, int] = field(default_factory=dict)
    
    def allocate_for_describer(self):
        """Allocation optimized for the Describer agent.
        
        Describer needs:
        - Good file content to understand the cycle
        - RAG context for architectural guidance
        - Less output reserve (produces description, not code)
        """
        # Reserve for output first
        output_reserve = int(self.total_tokens * 0.15)  # 15% for output
        available = self.total_tokens - output_reserve
        
        self.allocations = {
            BudgetCategory.OUTPUT_RESERVE: BudgetAllocation(
                BudgetCategory.OUTPUT_RESERVE, output_reserve, priority=100, flexible=False
            ),
            BudgetCategory.SYSTEM_PROMPT: BudgetAllocation(
                BudgetCategory.SYSTEM_PROMPT, int(available * 0.10), priority=90, flexible=False
            ),
            BudgetCategory.CYCLE_INFO: BudgetAllocation(
                BudgetCategory.CYCLE_INFO, int(available * 0.10), priority=85, flexible=True
            ),
            BudgetCategory.FILE_CONTENT: BudgetAllocation(
                BudgetCategory.FILE_CONTENT, int(available * 0.50), priority=80, flexible=True
            ),
            BudgetCategory.RAG_CONTEXT: BudgetAllocation(
                BudgetCategory.RAG_CONTEXT, int(available * 0.25), priority=70, flexible=True
            ),
            BudgetCategory.EXAMPLES: BudgetAllocation(
                BudgetCategory.EXAMPLES, int(available * 0.05), priority=50, flexible=True
            ),
        }
        logger.debug(f"Describer budget allocated: {self._summarize()}")
    
    def allocate_for_refactor(self, has_feedback: bool = False):
        """Allocation optimized for the Refactor agent.
        
        Refactor needs:
        - Maximum file content (must output complete files)
        - Pattern examples for guidance
        - More output reserve (produces full code)
        - Feedback if retrying
        """
        # Reserve more for output (full file content)
        output_reserve = int(self.total_tokens * 0.30)  # 30% for output
        available = self.total_tokens - output_reserve
        
        if has_feedback:
            # On retry, allocate budget for feedback
            self.allocations = {
                BudgetCategory.OUTPUT_RESERVE: BudgetAllocation(
                    BudgetCategory.OUTPUT_RESERVE, output_reserve, priority=100, flexible=False
                ),
                BudgetCategory.SYSTEM_PROMPT: BudgetAllocation(
                    BudgetCategory.SYSTEM_PROMPT, int(available * 0.08), priority=90, flexible=False
                ),
                BudgetCategory.FEEDBACK: BudgetAllocation(
                    BudgetCategory.FEEDBACK, int(available * 0.15), priority=88, flexible=False
                ),
                BudgetCategory.CYCLE_INFO: BudgetAllocation(
                    BudgetCategory.CYCLE_INFO, int(available * 0.07), priority=85, flexible=True
                ),
                BudgetCategory.FILE_CONTENT: BudgetAllocation(
                    BudgetCategory.FILE_CONTENT, int(available * 0.50), priority=80, flexible=True
                ),
                BudgetCategory.EXAMPLES: BudgetAllocation(
                    BudgetCategory.EXAMPLES, int(available * 0.10), priority=75, flexible=True
                ),
                BudgetCategory.RAG_CONTEXT: BudgetAllocation(
                    BudgetCategory.RAG_CONTEXT, int(available * 0.10), priority=60, flexible=True
                ),
            }
        else:
            self.allocations = {
                BudgetCategory.OUTPUT_RESERVE: BudgetAllocation(
                    BudgetCategory.OUTPUT_RESERVE, output_reserve, priority=100, flexible=False
                ),
                BudgetCategory.SYSTEM_PROMPT: BudgetAllocation(
                    BudgetCategory.SYSTEM_PROMPT, int(available * 0.08), priority=90, flexible=False
                ),
                BudgetCategory.CYCLE_INFO: BudgetAllocation(
                    BudgetCategory.CYCLE_INFO, int(available * 0.07), priority=85, flexible=True
                ),
                BudgetCategory.FILE_CONTENT: BudgetAllocation(
                    BudgetCategory.FILE_CONTENT, int(available * 0.55), priority=80, flexible=True
                ),
                BudgetCategory.EXAMPLES: BudgetAllocation(
                    BudgetCategory.EXAMPLES, int(available * 0.15), priority=75, flexible=True
                ),
                BudgetCategory.RAG_CONTEXT: BudgetAllocation(
                    BudgetCategory.RAG_CONTEXT, int(available * 0.15), priority=60, flexible=True
                ),
            }
        logger.debug(f"Refactor budget allocated (feedback={has_feedback}): {self._summarize()}")
    
    def allocate_for_validator(self):
        """Allocation optimized for the Validator agent.
        
        Validator needs:
        - Diffs (compact representation of changes)
        - Original description for context
        - Less RAG (validation is more mechanical)
        """
        output_reserve = int(self.total_tokens * 0.15)
        available = self.total_tokens - output_reserve
        
        self.allocations = {
            BudgetCategory.OUTPUT_RESERVE: BudgetAllocation(
                BudgetCategory.OUTPUT_RESERVE, output_reserve, priority=100, flexible=False
            ),
            BudgetCategory.SYSTEM_PROMPT: BudgetAllocation(
                BudgetCategory.SYSTEM_PROMPT, int(available * 0.10), priority=90, flexible=False
            ),
            BudgetCategory.CYCLE_INFO: BudgetAllocation(
                BudgetCategory.CYCLE_INFO, int(available * 0.15), priority=85, flexible=True
            ),
            BudgetCategory.FILE_CONTENT: BudgetAllocation(
                BudgetCategory.FILE_CONTENT, int(available * 0.60), priority=80, flexible=True
            ),
            BudgetCategory.RAG_CONTEXT: BudgetAllocation(
                BudgetCategory.RAG_CONTEXT, int(available * 0.10), priority=50, flexible=True
            ),
            BudgetCategory.EXAMPLES: BudgetAllocation(
                BudgetCategory.EXAMPLES, int(available * 0.05), priority=40, flexible=True
            ),
        }
        logger.debug(f"Validator budget allocated: {self._summarize()}")
    
    def get_token_budget(self, category: BudgetCategory) -> int:
        """Get token budget for a category."""
        if category in self.allocations:
            return self.allocations[category].tokens
        return 0
    
    def get_char_budget(self, category: BudgetCategory, content_type: str = "code_generic") -> int:
        """Get character budget for a category."""
        if category in self.allocations:
            return self.allocations[category].to_chars(content_type)
        return 0
    
    def use_budget(self, category: BudgetCategory, tokens_used: int):
        """Record budget usage for a category."""
        self.used[category] = self.used.get(category, 0) + tokens_used
    
    def get_remaining(self, category: BudgetCategory) -> int:
        """Get remaining token budget for a category."""
        allocated = self.get_token_budget(category)
        used = self.used.get(category, 0)
        return max(0, allocated - used)
    
    def redistribute_unused(self, from_category: BudgetCategory, to_category: BudgetCategory):
        """Move unused budget from one category to another."""
        if from_category not in self.allocations or to_category not in self.allocations:
            return
        
        from_alloc = self.allocations[from_category]
        if not from_alloc.flexible:
            return
        
        unused = self.get_remaining(from_category)
        if unused > 0:
            self.allocations[to_category].tokens += unused
            self.allocations[from_category].tokens -= unused
            logger.debug(f"Redistributed {unused} tokens from {from_category.value} to {to_category.value}")
    
    def _summarize(self) -> str:
        """Create a summary of allocations."""
        parts = []
        for cat, alloc in sorted(self.allocations.items(), key=lambda x: -x[1].priority):
            parts.append(f"{cat.value}={alloc.tokens}")
        return ", ".join(parts)


# =============================================================================
# File Prioritization for Cycles
# =============================================================================

@dataclass
class FilePriority:
    """Priority information for a file in the cycle."""
    path: str
    priority_score: float
    reasons: List[str] = field(default_factory=list)
    char_budget: int = 0


def prioritize_cycle_files(
    files: List[Dict[str, Any]],
    graph: Dict[str, Any],
    validation_issues: Optional[List[Dict[str, Any]]] = None,
    total_char_budget: int = 12000,
    strategy_hint: Optional[str] = None,
    edge_focus: Optional[List[List[str]]] = None,
) -> List[FilePriority]:
    """Prioritize files in a cycle for context inclusion.
    
    Prioritization heuristics for cycle refactoring:
    1. Files mentioned in validation issues (if retry)
    2. Files at cycle "joints" (have both incoming and outgoing edges)
    3. Files with more connections in the cycle
    4. Smaller files (easier to include completely)
    5. Strategy-aware prioritization (e.g., interfaces for interface_extraction)
    6. Edge focus (prioritize files involved in specific problematic edges)
    
    Args:
        files: List of file dicts with 'path' and 'content'
        graph: Graph dict with 'nodes' and 'edges'
        validation_issues: List of issues from failed validation
        total_char_budget: Total character budget to distribute
        strategy_hint: Suggested refactoring strategy (e.g., "interface_extraction")
        edge_focus: Specific edges to focus on [[from, to], ...]
        
    Returns:
        List of FilePriority objects sorted by priority (highest first)
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Build adjacency info
    incoming: Dict[str, int] = {n: 0 for n in nodes}
    outgoing: Dict[str, int] = {n: 0 for n in nodes}
    
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = edge[0], edge[1]
            if src in outgoing:
                outgoing[src] += 1
            if dst in incoming:
                incoming[dst] += 1
    
    # Build edge focus set for quick lookup
    focus_nodes = set()
    if edge_focus:
        for edge in edge_focus:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                focus_nodes.add(edge[0])
                focus_nodes.add(edge[1])
        logger.info(f"Edge focus: prioritizing nodes {focus_nodes}")
    
    # Map file paths to node names (heuristic: basename or path contains node)
    def file_matches_node(path: str, node: str) -> bool:
        path_lower = path.lower()
        node_lower = node.lower()
        # Check if node name appears in path
        return node_lower in path_lower or node_lower.replace(".", "/") in path_lower
    
    priorities: List[FilePriority] = []
    
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        content_len = len(content)
        
        score = 0.0
        reasons = []
        
        # Find matching node(s)
        matching_nodes = [n for n in nodes if file_matches_node(path, n)]
        
        # 1. Validation issues (highest priority on retry)
        if validation_issues:
            for issue in validation_issues:
                issue_path = issue.get("path", "")
                if issue_path and (path.endswith(issue_path) or issue_path.endswith(path.split("/")[-1])):
                    score += 50
                    reasons.append("mentioned in validation issue")
                    break
        
        # 2. Edge focus (specific edges to break)
        for node in matching_nodes:
            if node in focus_nodes:
                score += 40
                reasons.append("focus edge node")
                break
        
        # 3. Cycle joints (both incoming and outgoing)
        for node in matching_nodes:
            if incoming.get(node, 0) > 0 and outgoing.get(node, 0) > 0:
                score += 30
                reasons.append(f"cycle joint ({node})")
                break
        
        # 4. Connection count
        for node in matching_nodes:
            connections = incoming.get(node, 0) + outgoing.get(node, 0)
            if connections > 0:
                score += connections * 10
                reasons.append(f"{connections} connections")
        
        # 5. File size (prefer smaller files - easier to include completely)
        if content_len < 1000:
            score += 15
            reasons.append("small file")
        elif content_len < 3000:
            score += 10
            reasons.append("medium file")
        elif content_len > 10000:
            score -= 10
            reasons.append("large file (may truncate)")
        
        # 6. Interface files (often key to breaking cycles)
        basename = path.split("/")[-1].lower()
        if basename.startswith("i") and basename[1:2].isupper():
            score += 20
            reasons.append("interface file")
        elif "interface" in basename:
            score += 20
            reasons.append("interface file")
        
        # 7. Model/Entity files (often involved in cycles)
        if any(x in basename for x in ["model", "entity", "dto"]):
            score += 5
            reasons.append("model/entity file")
        
        # 8. Strategy-aware prioritization
        if strategy_hint:
            if strategy_hint == "interface_extraction":
                # Prioritize files that could use interfaces (concrete classes)
                if not basename.startswith("i") and not "interface" in basename:
                    # Prioritize files with high outgoing deps (depend on many things)
                    for node in matching_nodes:
                        if outgoing.get(node, 0) >= 2:
                            score += 25
                            reasons.append("interface extraction candidate")
                            break
            elif strategy_hint == "dependency_inversion":
                # Prioritize high-level modules (few outgoing, many incoming)
                for node in matching_nodes:
                    if incoming.get(node, 0) > outgoing.get(node, 0):
                        score += 20
                        reasons.append("high-level module (DI candidate)")
                        break
            elif strategy_hint == "shared_module":
                # Prioritize files that are depended on by multiple nodes
                for node in matching_nodes:
                    if incoming.get(node, 0) >= 2:
                        score += 20
                        reasons.append("shared module candidate")
                        break
            elif strategy_hint == "mediator":
                # Prioritize files with bidirectional dependencies
                for node in matching_nodes:
                    if incoming.get(node, 0) > 0 and outgoing.get(node, 0) > 0:
                        score += 25
                        reasons.append("mediator pattern candidate")
                        break
        
        priorities.append(FilePriority(
            path=path,
            priority_score=score,
            reasons=reasons,
        ))
    
    # Sort by priority (highest first)
    priorities.sort(key=lambda x: -x.priority_score)
    
    # Distribute character budget based on priority
    if priorities:
        total_score = sum(max(p.priority_score, 1) for p in priorities)  # Ensure min score of 1
        
        for p in priorities:
            # Proportional allocation based on score
            p.char_budget = int(total_char_budget * max(p.priority_score, 1) / total_score)
            
            # Ensure minimum budget for each file
            p.char_budget = max(p.char_budget, 500)
    
    # Log prioritization
    logger.info(f"File prioritization for {len(priorities)} files (strategy={strategy_hint}):")
    for p in priorities[:5]:  # Log top 5
        logger.info(f"  {p.path.split('/')[-1]}: score={p.priority_score:.1f}, budget={p.char_budget}, reasons={p.reasons}")
    
    return priorities


def get_file_budget(
    path: str,
    priorities: List[FilePriority],
    default_budget: int = 4000
) -> int:
    """Get the character budget for a specific file."""
    for p in priorities:
        if p.path == path or p.path.endswith(path.split("/")[-1]):
            return p.char_budget
    return default_budget


# =============================================================================
# Convenience Functions
# =============================================================================

def create_budget_for_agent(
    agent_name: str,
    total_tokens: int = 4096,
    has_feedback: bool = False
) -> ContextBudget:
    """Create a context budget for a specific agent.
    
    Args:
        agent_name: One of 'describer', 'refactor', 'validator', 'explainer'
        total_tokens: Total context window size
        has_feedback: Whether this is a retry with feedback
        
    Returns:
        Configured ContextBudget
    """
    budget = ContextBudget(total_tokens=total_tokens)
    
    if agent_name == "describer":
        budget.allocate_for_describer()
    elif agent_name == "refactor":
        budget.allocate_for_refactor(has_feedback=has_feedback)
    elif agent_name == "validator":
        budget.allocate_for_validator()
    else:
        # Default allocation
        budget.allocate_for_describer()
    
    return budget


def estimate_prompt_tokens(prompt: str) -> int:
    """Estimate tokens in a complete prompt."""
    return TokenEstimator.estimate_tokens(prompt, "code_generic")


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    content_type: str = "code_generic",
    preserve_end: bool = False
) -> str:
    """Truncate text to fit within a token budget.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        content_type: Type of content for estimation
        preserve_end: If True, keep the end instead of the beginning
        
    Returns:
        Truncated text with marker if truncated
    """
    current_tokens = TokenEstimator.estimate_tokens(text, content_type)
    
    if current_tokens <= max_tokens:
        return text
    
    # Estimate chars to keep
    target_chars = TokenEstimator.tokens_to_chars(max_tokens, content_type)
    target_chars = int(target_chars * 0.95)  # Leave some margin
    
    if preserve_end:
        truncated = "...[truncated]...\n" + text[-target_chars:]
    else:
        truncated = text[:target_chars] + "\n...[truncated]"
    
    return truncated
