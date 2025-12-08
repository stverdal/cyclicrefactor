"""Validation Memory - Learn from past validation attempts.

This module provides persistent memory of validation outcomes to help
the LLM learn from what worked and what didn't across multiple runs.

Key concepts:
- ValidationAttempt: A single refactoring attempt with its outcome
- PatternMemory: Aggregated learnings about what works for different patterns
- CycleMemory: History specific to a cycle (for retries within a session)
- GlobalMemory: Cross-session learnings stored on disk
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from utils.logging import get_logger

logger = get_logger("validation_memory")


@dataclass
class ValidationAttempt:
    """Record of a single validation attempt."""
    timestamp: str
    cycle_id: str
    strategy: str  # e.g., "interface_extraction", "lazy_import", etc.
    files_changed: List[str]
    
    # Outcome
    approved: bool
    issues: List[Dict[str, Any]]  # List of issue dicts
    suggestions: List[str]
    
    # What was tried
    approach_summary: str  # Brief description of what was attempted
    diff_summary: str  # Condensed diff info
    
    # Learning signals
    issue_types: List[str]  # e.g., ["syntax", "missing_import", "cycle_not_broken"]
    severity_counts: Dict[str, int] = field(default_factory=dict)  # {"critical": 1, "major": 2}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationAttempt":
        return cls(**data)


@dataclass
class PatternLearning:
    """Aggregated learning about a specific pattern/strategy."""
    pattern_name: str
    success_count: int = 0
    failure_count: int = 0
    common_issues: Dict[str, int] = field(default_factory=dict)  # issue_type -> count
    successful_approaches: List[str] = field(default_factory=list)
    failed_approaches: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)  # Generated tips based on history
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternLearning":
        return cls(**data)


class ValidationMemory:
    """Manages validation memory across sessions.
    
    Memory is stored in two places:
    1. Session memory (in-memory): Current session's attempts
    2. Persistent memory (JSON file): Cross-session learnings
    
    The memory helps by:
    - Tracking what strategies work for what patterns
    - Recording common failure modes
    - Providing context to the LLM about past attempts
    - Suggesting alternative approaches when current ones fail
    """
    
    def __init__(
        self,
        memory_file: str = "cache/validation_memory.json",
        max_attempts_per_cycle: int = 10,
        max_pattern_history: int = 50,
    ):
        self.memory_file = Path(memory_file)
        self.max_attempts_per_cycle = max_attempts_per_cycle
        self.max_pattern_history = max_pattern_history
        
        # Session memory (current run)
        self.session_attempts: Dict[str, List[ValidationAttempt]] = {}  # cycle_id -> attempts
        
        # Persistent memory (loaded from disk)
        self.pattern_learnings: Dict[str, PatternLearning] = {}
        self.global_attempts: List[ValidationAttempt] = []
        
        # Load existing memory
        self._load_memory()
    
    def _load_memory(self):
        """Load persistent memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load pattern learnings
                for name, learning_data in data.get("patterns", {}).items():
                    self.pattern_learnings[name] = PatternLearning.from_dict(learning_data)
                
                # Load recent global attempts (limited)
                for attempt_data in data.get("recent_attempts", [])[-self.max_pattern_history:]:
                    self.global_attempts.append(ValidationAttempt.from_dict(attempt_data))
                
                logger.info(f"Loaded validation memory: {len(self.pattern_learnings)} patterns, "
                           f"{len(self.global_attempts)} recent attempts")
            except Exception as e:
                logger.warning(f"Failed to load validation memory: {e}")
    
    def _save_memory(self):
        """Save persistent memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "last_updated": datetime.now().isoformat(),
                "patterns": {name: learning.to_dict() 
                            for name, learning in self.pattern_learnings.items()},
                "recent_attempts": [a.to_dict() for a in self.global_attempts[-self.max_pattern_history:]],
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved validation memory to {self.memory_file}")
        except Exception as e:
            logger.warning(f"Failed to save validation memory: {e}")
    
    def record_attempt(
        self,
        cycle_id: str,
        strategy: str,
        files_changed: List[str],
        approved: bool,
        issues: List[Dict[str, Any]],
        suggestions: List[str],
        approach_summary: str,
        diff_summary: str = "",
    ):
        """Record a validation attempt for learning.
        
        Args:
            cycle_id: ID of the cycle being refactored
            strategy: Strategy used (e.g., "interface_extraction")
            files_changed: List of file paths that were modified
            approved: Whether validation approved the changes
            issues: List of issue dicts from validation
            suggestions: List of suggestion strings
            approach_summary: Brief description of the approach taken
            diff_summary: Optional condensed diff info
        """
        # Extract issue types and severity counts
        issue_types = []
        severity_counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}
        
        for issue in issues:
            if isinstance(issue, dict):
                issue_type = issue.get("issue_type", issue.get("type", "unknown"))
                severity = issue.get("severity", "minor")
            else:
                issue_type = getattr(issue, "issue_type", "unknown")
                severity = getattr(issue, "severity", "minor")
            
            if issue_type not in issue_types:
                issue_types.append(issue_type)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        attempt = ValidationAttempt(
            timestamp=datetime.now().isoformat(),
            cycle_id=cycle_id,
            strategy=strategy,
            files_changed=files_changed,
            approved=approved,
            issues=[i if isinstance(i, dict) else asdict(i) if hasattr(i, '__dataclass_fields__') else {"comment": str(i)} for i in issues],
            suggestions=suggestions,
            approach_summary=approach_summary,
            diff_summary=diff_summary[:500] if diff_summary else "",
            issue_types=issue_types,
            severity_counts=severity_counts,
        )
        
        # Add to session memory
        if cycle_id not in self.session_attempts:
            self.session_attempts[cycle_id] = []
        self.session_attempts[cycle_id].append(attempt)
        
        # Trim session memory if needed
        if len(self.session_attempts[cycle_id]) > self.max_attempts_per_cycle:
            self.session_attempts[cycle_id] = self.session_attempts[cycle_id][-self.max_attempts_per_cycle:]
        
        # Add to global attempts
        self.global_attempts.append(attempt)
        
        # Update pattern learnings
        self._update_pattern_learning(attempt)
        
        # Save to disk
        self._save_memory()
        
        logger.info(f"Recorded validation attempt for {cycle_id}: approved={approved}, "
                   f"strategy={strategy}, issues={len(issues)}")
    
    def _update_pattern_learning(self, attempt: ValidationAttempt):
        """Update pattern learning based on an attempt."""
        strategy = attempt.strategy or "unknown"
        
        if strategy not in self.pattern_learnings:
            self.pattern_learnings[strategy] = PatternLearning(pattern_name=strategy)
        
        learning = self.pattern_learnings[strategy]
        
        if attempt.approved:
            learning.success_count += 1
            if attempt.approach_summary and attempt.approach_summary not in learning.successful_approaches:
                learning.successful_approaches.append(attempt.approach_summary)
                # Keep only recent successful approaches
                learning.successful_approaches = learning.successful_approaches[-10:]
        else:
            learning.failure_count += 1
            if attempt.approach_summary and attempt.approach_summary not in learning.failed_approaches:
                learning.failed_approaches.append(attempt.approach_summary)
                learning.failed_approaches = learning.failed_approaches[-10:]
            
            # Track common issues
            for issue_type in attempt.issue_types:
                learning.common_issues[issue_type] = learning.common_issues.get(issue_type, 0) + 1
        
        # Generate tips based on history
        self._generate_tips(learning)
    
    def _generate_tips(self, learning: PatternLearning):
        """Generate tips based on accumulated learning."""
        tips = []
        
        # Tip based on success rate
        if learning.failure_count > 3 and learning.success_rate < 0.3:
            tips.append(f"Strategy '{learning.pattern_name}' has low success rate ({learning.success_rate:.0%}). Consider alternative approaches.")
        
        # Tips based on common issues
        if learning.common_issues:
            sorted_issues = sorted(learning.common_issues.items(), key=lambda x: x[1], reverse=True)
            top_issue = sorted_issues[0]
            if top_issue[1] >= 3:
                tips.append(f"Common issue with '{learning.pattern_name}': {top_issue[0]} (occurred {top_issue[1]} times)")
        
        # Tips from successful approaches
        if learning.successful_approaches:
            tips.append(f"Previously successful: {learning.successful_approaches[-1]}")
        
        learning.tips = tips[-5:]  # Keep only recent tips
    
    def get_session_history(self, cycle_id: str) -> List[ValidationAttempt]:
        """Get validation history for current session."""
        return self.session_attempts.get(cycle_id, [])
    
    def get_failed_attempts(self, cycle_id: str) -> List[ValidationAttempt]:
        """Get only failed attempts for a cycle."""
        return [a for a in self.get_session_history(cycle_id) if not a.approved]
    
    def get_pattern_learning(self, strategy: str) -> Optional[PatternLearning]:
        """Get learning for a specific strategy/pattern."""
        return self.pattern_learnings.get(strategy)
    
    def get_similar_failures(self, issue_types: List[str], limit: int = 5) -> List[ValidationAttempt]:
        """Find past attempts with similar issue types."""
        similar = []
        for attempt in reversed(self.global_attempts):
            if not attempt.approved:
                overlap = set(attempt.issue_types) & set(issue_types)
                if overlap:
                    similar.append(attempt)
                    if len(similar) >= limit:
                        break
        return similar
    
    def build_memory_context(
        self,
        cycle_id: str,
        strategy: str = None,
        max_chars: int = 2000,
    ) -> str:
        """Build context string for LLM prompt based on memory.
        
        This creates a summary of relevant past attempts and learnings
        that can be included in the prompt to help the LLM avoid
        repeating mistakes.
        
        Args:
            cycle_id: Current cycle being processed
            strategy: Current strategy being attempted
            max_chars: Maximum characters for the context
            
        Returns:
            Formatted string with memory context
        """
        lines = []
        
        # Session history for this cycle
        session_history = self.get_session_history(cycle_id)
        if session_history:
            lines.append("## Previous Attempts (This Session)")
            for i, attempt in enumerate(session_history[-3:], 1):  # Last 3 attempts
                status = "✓ APPROVED" if attempt.approved else "✗ REJECTED"
                lines.append(f"\n### Attempt {i}: {status}")
                lines.append(f"Strategy: {attempt.strategy}")
                lines.append(f"Approach: {attempt.approach_summary}")
                if not attempt.approved and attempt.issue_types:
                    lines.append(f"Issues: {', '.join(attempt.issue_types)}")
                if attempt.suggestions:
                    lines.append(f"Suggestions: {attempt.suggestions[0][:100]}...")
            lines.append("")
        
        # Pattern learning
        if strategy:
            learning = self.get_pattern_learning(strategy)
            if learning and (learning.success_count + learning.failure_count) > 0:
                lines.append(f"## Learning for '{strategy}' Strategy")
                lines.append(f"Success rate: {learning.success_rate:.0%} ({learning.success_count}/{learning.success_count + learning.failure_count})")
                
                if learning.common_issues:
                    top_issues = sorted(learning.common_issues.items(), key=lambda x: x[1], reverse=True)[:3]
                    lines.append(f"Common issues: {', '.join(f'{k}({v})' for k, v in top_issues)}")
                
                if learning.tips:
                    lines.append("Tips:")
                    for tip in learning.tips[:2]:
                        lines.append(f"  - {tip}")
                
                if learning.successful_approaches:
                    lines.append(f"What worked before: {learning.successful_approaches[-1][:150]}")
                
                lines.append("")
        
        # What to avoid (from failed attempts)
        failed = self.get_failed_attempts(cycle_id)
        if failed:
            lines.append("## What to Avoid")
            avoided = set()
            for attempt in failed[-3:]:
                if attempt.approach_summary and attempt.approach_summary not in avoided:
                    avoided.add(attempt.approach_summary)
                    lines.append(f"- Don't: {attempt.approach_summary[:100]}")
            lines.append("")
        
        # Truncate if too long
        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars-50] + "\n\n[Memory truncated...]"
        
        return result if lines else ""
    
    def get_retry_guidance(self, cycle_id: str, current_issues: List[str]) -> str:
        """Get specific guidance for retry based on current issues and history.
        
        Args:
            cycle_id: Current cycle
            current_issues: Issue types from current validation
            
        Returns:
            Guidance string for the next attempt
        """
        guidance = []
        
        # Check session history
        session_history = self.get_session_history(cycle_id)
        attempts_count = len(session_history)
        
        if attempts_count == 0:
            return ""
        
        # Find patterns in failures
        all_issue_types = []
        for attempt in session_history:
            if not attempt.approved:
                all_issue_types.extend(attempt.issue_types)
        
        # Count recurring issues
        from collections import Counter
        issue_counts = Counter(all_issue_types)
        recurring = [issue for issue, count in issue_counts.items() if count >= 2]
        
        if recurring:
            guidance.append(f"RECURRING ISSUES (failed {len(recurring)} times): {', '.join(recurring)}")
            guidance.append("These issues keep appearing - try a fundamentally different approach.")
        
        # Check if same strategy keeps failing
        strategy_failures = Counter(a.strategy for a in session_history if not a.approved)
        for strategy, count in strategy_failures.items():
            if count >= 2:
                guidance.append(f"Strategy '{strategy}' has failed {count} times. Consider switching strategies.")
        
        # Suggest based on what worked globally
        if "syntax" in current_issues:
            guidance.append("SYNTAX ISSUES: Focus on generating complete, valid code. Don't use placeholders like '...'")
        
        if "cycle_not_broken" in current_issues or "cycle" in current_issues:
            guidance.append("CYCLE NOT BROKEN: The import/dependency still exists. Must remove or invert the specific import.")
        
        if "missing_import" in current_issues:
            guidance.append("MISSING IMPORTS: Ensure all new types/interfaces are properly imported where used.")
        
        return "\n".join(guidance) if guidance else ""
    
    def clear_session(self, cycle_id: str = None):
        """Clear session memory (optionally for specific cycle)."""
        if cycle_id:
            self.session_attempts.pop(cycle_id, None)
        else:
            self.session_attempts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_attempts = len(self.global_attempts)
        approved = sum(1 for a in self.global_attempts if a.approved)
        
        return {
            "total_attempts": total_attempts,
            "approved_count": approved,
            "rejected_count": total_attempts - approved,
            "success_rate": approved / total_attempts if total_attempts > 0 else 0,
            "patterns_tracked": len(self.pattern_learnings),
            "session_cycles": len(self.session_attempts),
        }


# Singleton instance for global access
_memory_instance: Optional[ValidationMemory] = None


def get_validation_memory(memory_file: str = "cache/validation_memory.json") -> ValidationMemory:
    """Get or create the global validation memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ValidationMemory(memory_file=memory_file)
    return _memory_instance


def reset_validation_memory():
    """Reset the global validation memory instance."""
    global _memory_instance
    _memory_instance = None
