"""
RAG Query Builder - Creates effective queries for academic literature.

The key insight is that academic papers discuss CONCEPTS and PATTERNS,
not specific file names or module names from your codebase. This utility
helps agents formulate queries that will actually match indexed documents.

Usage:
    from utils.rag_query_builder import RAGQueryBuilder
    
    builder = RAGQueryBuilder()
    queries = builder.build_queries_for_cycle(cycle_spec, intent="describe")
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re

from utils.logging import get_logger

logger = get_logger("rag_query_builder")


class CycleType(Enum):
    """Classification of cyclic dependency types."""
    BIDIRECTIONAL = "bidirectional"      # A ↔ B (two nodes)
    TRANSITIVE = "transitive"            # A → B → C → A (chain)
    STAR = "star"                        # Multiple nodes all depend on each other
    INHERITANCE = "inheritance"          # Involves class inheritance
    LAYER_VIOLATION = "layer_violation"  # Lower layer depends on higher layer
    UNKNOWN = "unknown"


class QueryIntent(Enum):
    """What the agent wants to learn from RAG."""
    UNDERSTAND = "understand"      # Describer: understand why cycles are bad
    STRATEGIZE = "strategize"      # Describer: what strategies exist
    IMPLEMENT = "implement"        # Refactor: how to implement a fix
    VALIDATE = "validate"          # Validator: what makes a good refactor
    EXPLAIN = "explain"            # Explainer: how to explain the change


@dataclass
class CycleAnalysis:
    """Analysis of a cycle's characteristics."""
    cycle_type: CycleType
    node_count: int
    has_inheritance: bool
    has_interface: bool
    layer_violation: Optional[str]  # e.g., "UI → DataAccess"
    dominant_pattern: str           # e.g., "tight coupling", "shared state"
    keywords: List[str]             # Extracted relevant keywords


class RAGQueryBuilder:
    """
    Builds effective RAG queries that match academic literature.
    
    Key principles:
    1. Never include file names, class names, or code-specific identifiers
    2. Use architectural concepts and design pattern terminology
    3. Match the vocabulary of software engineering literature
    4. Build multiple complementary queries for better coverage
    """
    
    # Conceptual vocabulary that matches academic papers
    CONCEPT_VOCABULARY = {
        # Problem concepts
        "cycle": [
            "cyclic dependency", "circular dependency", "dependency cycle",
            "mutual dependency", "bidirectional coupling"
        ],
        "coupling": [
            "tight coupling", "high coupling", "strong coupling",
            "module coupling", "component coupling"
        ],
        "problem": [
            "code smell", "architectural debt", "technical debt",
            "maintainability issue", "testability problem"
        ],
        
        # Solution concepts
        "interface": [
            "interface extraction", "abstract interface", "dependency inversion",
            "program to interface", "interface segregation"
        ],
        "inversion": [
            "dependency inversion principle", "inversion of control",
            "dependency injection", "IoC container"
        ],
        "extraction": [
            "extract class", "extract module", "extract component",
            "shared module", "common abstraction"
        ],
        "patterns": [
            "design pattern", "mediator pattern", "observer pattern",
            "facade pattern", "adapter pattern"
        ],
        
        # Principles
        "solid": [
            "SOLID principles", "single responsibility", "open closed principle",
            "dependency inversion principle", "interface segregation"
        ],
        "architecture": [
            "clean architecture", "layered architecture", "hexagonal architecture",
            "acyclic dependencies principle", "stable dependencies principle"
        ],
    }
    
    # Layer patterns for detecting layer violations
    LAYER_PATTERNS = {
        "ui": ["ui", "view", "controller", "presentation", "web", "api", "endpoint"],
        "business": ["service", "business", "domain", "logic", "manager", "handler"],
        "data": ["repository", "dao", "data", "database", "storage", "persistence", "entity"],
        "infrastructure": ["infrastructure", "external", "integration", "client"],
    }
    
    # Layer hierarchy (higher index = lower layer)
    LAYER_HIERARCHY = ["ui", "business", "data", "infrastructure"]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._inheritance_pattern = re.compile(
            r'\b(extends|inherits|:.*class|<.*>)\b', re.IGNORECASE
        )
        self._interface_pattern = re.compile(
            r'\b(interface|implements|IDisposable|IEnumerable)\b', re.IGNORECASE
        )
    
    def analyze_cycle(self, cycle_spec: Dict[str, Any]) -> CycleAnalysis:
        """
        Analyze a cycle to understand its characteristics.
        
        Args:
            cycle_spec: Cycle specification dict with graph and files
            
        Returns:
            CycleAnalysis with classified information
        """
        graph = cycle_spec.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        files = cycle_spec.get("files", [])
        
        # Determine cycle type
        cycle_type = self._classify_cycle_type(nodes, edges)
        
        # Analyze code content
        all_content = " ".join(f.get("content", "") for f in files if f.get("content"))
        has_inheritance = bool(self._inheritance_pattern.search(all_content))
        has_interface = bool(self._interface_pattern.search(all_content))
        
        # Detect layer violations
        layer_violation = self._detect_layer_violation(nodes, files)
        
        # Determine dominant pattern
        dominant_pattern = self._infer_dominant_pattern(
            cycle_type, has_inheritance, has_interface, layer_violation
        )
        
        # Extract keywords (conceptual, not code-specific)
        keywords = self._extract_conceptual_keywords(
            cycle_type, has_inheritance, has_interface, layer_violation
        )
        
        analysis = CycleAnalysis(
            cycle_type=cycle_type,
            node_count=len(nodes),
            has_inheritance=has_inheritance,
            has_interface=has_interface,
            layer_violation=layer_violation,
            dominant_pattern=dominant_pattern,
            keywords=keywords,
        )
        
        logger.info(f"Cycle analysis: type={cycle_type.value}, nodes={len(nodes)}, "
                   f"inheritance={has_inheritance}, layer_violation={layer_violation}")
        
        return analysis
    
    def _classify_cycle_type(self, nodes: List[str], edges: List[Any]) -> CycleType:
        """Classify the type of cycle based on structure."""
        node_count = len(nodes)
        
        if node_count == 2:
            return CycleType.BIDIRECTIONAL
        elif node_count <= 4:
            return CycleType.TRANSITIVE
        else:
            # Check if it's a star pattern (everything connects to everything)
            if len(edges) > node_count * 1.5:
                return CycleType.STAR
            return CycleType.TRANSITIVE
    
    def _detect_layer_violation(
        self, nodes: List[str], files: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Detect if the cycle involves a layer violation."""
        node_layers = {}
        
        for node in nodes:
            node_lower = node.lower()
            for layer, patterns in self.LAYER_PATTERNS.items():
                if any(p in node_lower for p in patterns):
                    node_layers[node] = layer
                    break
        
        if len(node_layers) < 2:
            return None
        
        # Check for violations (lower layer depending on higher layer)
        layers_involved = list(set(node_layers.values()))
        if len(layers_involved) >= 2:
            # Find the highest and lowest layers
            layer_indices = [self.LAYER_HIERARCHY.index(l) for l in layers_involved 
                           if l in self.LAYER_HIERARCHY]
            if layer_indices:
                min_idx, max_idx = min(layer_indices), max(layer_indices)
                if max_idx - min_idx >= 1:
                    high_layer = self.LAYER_HIERARCHY[min_idx]
                    low_layer = self.LAYER_HIERARCHY[max_idx]
                    return f"{low_layer} → {high_layer}"
        
        return None
    
    def _infer_dominant_pattern(
        self,
        cycle_type: CycleType,
        has_inheritance: bool,
        has_interface: bool,
        layer_violation: Optional[str],
    ) -> str:
        """Infer the dominant problematic pattern."""
        if layer_violation:
            return "layer violation"
        if has_inheritance:
            return "inheritance coupling"
        if cycle_type == CycleType.BIDIRECTIONAL:
            return "mutual dependency"
        if cycle_type == CycleType.STAR:
            return "highly coupled cluster"
        return "transitive dependency chain"
    
    def _extract_conceptual_keywords(
        self,
        cycle_type: CycleType,
        has_inheritance: bool,
        has_interface: bool,
        layer_violation: Optional[str],
    ) -> List[str]:
        """Extract conceptual keywords for RAG queries."""
        keywords = ["cyclic dependency", "refactoring"]
        
        if cycle_type == CycleType.BIDIRECTIONAL:
            keywords.extend(["bidirectional", "mutual dependency"])
        elif cycle_type == CycleType.TRANSITIVE:
            keywords.extend(["transitive", "chain"])
        
        if has_inheritance:
            keywords.extend(["inheritance", "subclass", "base class"])
        
        if has_interface:
            keywords.extend(["interface", "abstraction"])
        
        if layer_violation:
            keywords.extend(["layer", "architecture", "separation of concerns"])
        
        return keywords
    
    def build_queries_for_cycle(
        self,
        cycle_spec: Dict[str, Any],
        intent: QueryIntent,
        description_hints: Optional[str] = None,
    ) -> List[str]:
        """
        Build effective RAG queries for a cycle.
        
        Args:
            cycle_spec: The cycle specification
            intent: What the agent wants to learn
            description_hints: Optional hints from previous agent output
            
        Returns:
            List of query strings (use all for comprehensive retrieval)
        """
        analysis = self.analyze_cycle(cycle_spec)
        queries = []
        
        if intent == QueryIntent.UNDERSTAND:
            queries.extend(self._build_understanding_queries(analysis))
        elif intent == QueryIntent.STRATEGIZE:
            queries.extend(self._build_strategy_queries(analysis))
        elif intent == QueryIntent.IMPLEMENT:
            queries.extend(self._build_implementation_queries(analysis, description_hints))
        elif intent == QueryIntent.VALIDATE:
            queries.extend(self._build_validation_queries(analysis))
        elif intent == QueryIntent.EXPLAIN:
            queries.extend(self._build_explanation_queries(analysis))
        
        # Log queries for debugging
        logger.info(f"Built {len(queries)} RAG queries for intent={intent.value}")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  Query {i}: {q}")
        
        return queries
    
    def _build_understanding_queries(self, analysis: CycleAnalysis) -> List[str]:
        """Queries for understanding why the cycle is problematic."""
        queries = [
            # General understanding
            "why cyclic dependencies are problematic software design",
            "impact of circular dependencies on maintainability testability",
        ]
        
        # Specific to cycle type
        if analysis.cycle_type == CycleType.BIDIRECTIONAL:
            queries.append("bidirectional dependency coupling problems solutions")
        
        if analysis.layer_violation:
            queries.append("layered architecture violations consequences")
            queries.append("acyclic dependencies principle clean architecture")
        
        if analysis.has_inheritance:
            queries.append("inheritance dependency problems composition over inheritance")
        
        return queries
    
    def _build_strategy_queries(self, analysis: CycleAnalysis) -> List[str]:
        """Queries for learning about available strategies."""
        queries = [
            # Core strategies
            "breaking cyclic dependencies strategies techniques",
            "dependency inversion principle interface extraction",
        ]
        
        # Pattern-specific strategies
        if analysis.cycle_type == CycleType.BIDIRECTIONAL:
            queries.append("extract interface break bidirectional dependency")
            queries.append("dependency inversion two modules tightly coupled")
        elif analysis.cycle_type == CycleType.TRANSITIVE:
            queries.append("break dependency chain introduce abstraction layer")
        
        if analysis.layer_violation:
            queries.append("fix layer violation dependency inversion")
            queries.append("clean architecture dependency rule")
        
        if analysis.has_inheritance:
            queries.append("replace inheritance with composition delegation")
        
        return queries
    
    def _build_implementation_queries(
        self, analysis: CycleAnalysis, hints: Optional[str]
    ) -> List[str]:
        """Queries for implementation guidance."""
        queries = [
            # General implementation
            "refactoring cyclic dependency step by step",
            "extract interface refactoring example",
        ]
        
        # If we have hints from describer, use them
        if hints:
            hints_lower = hints.lower()
            if "interface" in hints_lower:
                queries.append("implementing interface extraction refactoring")
                queries.append("dependency inversion with interfaces example")
            if "shared" in hints_lower or "common" in hints_lower:
                queries.append("extract shared module common functionality")
            if "event" in hints_lower or "callback" in hints_lower:
                queries.append("event driven decoupling observer pattern implementation")
            if "mediator" in hints_lower:
                queries.append("mediator pattern implementation decoupling")
        
        # Pattern-specific implementation
        if analysis.cycle_type == CycleType.BIDIRECTIONAL:
            queries.append("implement dependency inversion two classes")
        
        if analysis.layer_violation:
            queries.append("implementing ports and adapters hexagonal architecture")
        
        return queries
    
    def _build_validation_queries(self, analysis: CycleAnalysis) -> List[str]:
        """Queries for validation criteria."""
        return [
            "validating refactoring success criteria",
            "checking dependency cycle is broken",
            "refactoring quality attributes maintainability",
            "code review refactoring checklist",
        ]
    
    def _build_explanation_queries(self, analysis: CycleAnalysis) -> List[str]:
        """Queries for explaining the refactoring."""
        queries = [
            "explaining refactoring benefits stakeholders",
            "dependency inversion benefits explained",
        ]
        
        if analysis.has_interface:
            queries.append("benefits of interface extraction explained")
        
        if analysis.layer_violation:
            queries.append("clean architecture benefits explained")
        
        return queries
    
    def get_strategy_keywords(self, analysis: CycleAnalysis) -> Dict[str, List[str]]:
        """
        Get keywords associated with different refactoring strategies.
        Useful for detecting which strategy was attempted.
        """
        return {
            "interface_extraction": [
                "interface", "abstraction", "IService", "extract interface",
                "program to interface"
            ],
            "dependency_inversion": [
                "inversion", "inject", "IoC", "constructor injection",
                "dependency injection"
            ],
            "shared_module": [
                "shared", "common", "utility", "extract module",
                "move to common"
            ],
            "mediator": [
                "mediator", "coordinator", "hub", "event bus",
                "message broker"
            ],
            "facade": [
                "facade", "wrapper", "adapter", "gateway"
            ],
            "lazy_loading": [
                "lazy", "deferred", "on demand", "late binding"
            ],
        }


# Convenience functions
def build_rag_queries(
    cycle_spec: Dict[str, Any],
    intent: str,
    hints: Optional[str] = None
) -> List[str]:
    """
    Convenience function to build RAG queries.
    
    Args:
        cycle_spec: Cycle specification dict
        intent: One of "understand", "strategize", "implement", "validate", "explain"
        hints: Optional hints from previous agent
        
    Returns:
        List of query strings
    """
    builder = RAGQueryBuilder()
    query_intent = QueryIntent(intent)
    return builder.build_queries_for_cycle(cycle_spec, query_intent, hints)


def analyze_cycle(cycle_spec: Dict[str, Any]) -> CycleAnalysis:
    """
    Convenience function to analyze a cycle.
    
    Args:
        cycle_spec: Cycle specification dict
        
    Returns:
        CycleAnalysis object
    """
    builder = RAGQueryBuilder()
    return builder.analyze_cycle(cycle_spec)
