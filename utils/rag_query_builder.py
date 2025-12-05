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

from typing import List, Dict, Any, Optional, Set, Tuple
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


class UnbreakableReason(Enum):
    """Reasons why a cycle might be unbreakable or impractical to break."""
    NOT_UNBREAKABLE = "not_unbreakable"  # Cycle can likely be broken
    MUTUAL_RECURSION = "mutual_recursion"  # Fundamental algorithmic recursion
    CIRCULAR_DATA_STRUCTURE = "circular_data"  # Tree/graph with parent-child refs
    FRAMEWORK_REQUIREMENT = "framework"  # Framework pattern requires cycle
    INCOMPLETE_CONTEXT = "incomplete"  # Not enough file content to refactor
    EXTERNAL_DEPENDENCY = "external"  # Cycle involves unmodifiable code
    TIGHTLY_COUPLED_BY_DESIGN = "by_design"  # Intentional tight coupling


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

    def detect_unbreakable_cycle(
        self, cycle_spec: Dict[str, Any], validation_history: List[Dict[str, Any]] = None
    ) -> Tuple[bool, UnbreakableReason, str]:
        """
        Detect if a cycle is likely unbreakable or impractical to break.
        
        Args:
            cycle_spec: Cycle specification dict
            validation_history: List of previous validation results (for pattern detection)
            
        Returns:
            Tuple of (is_unbreakable, reason, explanation)
        """
        graph = cycle_spec.get("graph", {})
        nodes = graph.get("nodes", [])
        files = cycle_spec.get("files", [])
        
        # Combine all file content for analysis
        all_content = " ".join(f.get("content", "") for f in files if f.get("content"))
        content_lower = all_content.lower()
        
        # Check 1: Incomplete context (not enough content to refactor)
        total_content_len = sum(len(f.get("content", "")) for f in files)
        if total_content_len < 100:  # Very little content
            return (
                True, 
                UnbreakableReason.INCOMPLETE_CONTEXT,
                f"Insufficient file content provided ({total_content_len} chars). "
                "Cannot analyze dependencies without seeing the actual code."
            )
        
        # Check for truncated files
        truncated_markers = ["...", "[truncated]", "/* lines", "// lines"]
        truncated_files = [
            f.get("path", "unknown") for f in files 
            if any(m in f.get("content", "").lower() for m in truncated_markers)
        ]
        if len(truncated_files) >= len(files) * 0.5:  # More than half truncated
            return (
                True,
                UnbreakableReason.INCOMPLETE_CONTEXT,
                f"Most files appear to be truncated: {', '.join(truncated_files[:3])}. "
                "Cannot safely refactor without seeing complete file content."
            )
        
        # Check 2: Mutual recursion patterns
        mutual_recursion_patterns = [
            (r'\bParser\b.*\bLexer\b|\bLexer\b.*\bParser\b', "parser-lexer recursion"),
            (r'\bvisitor\b.*\bnode\b|\bnode\b.*\bvisitor\b', "visitor pattern"),
            (r'\bExpr.*\beval.*\bExpr\b', "expression evaluation recursion"),
            (r'\bserializ.*\bdeserializ', "serialization pair"),
        ]
        for pattern, description in mutual_recursion_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return (
                    True,
                    UnbreakableReason.MUTUAL_RECURSION,
                    f"Detected fundamental mutual recursion pattern ({description}). "
                    "This is an algorithmic design that inherently requires bidirectional references. "
                    "Consider documenting this as an intentional architectural decision rather than tech debt."
                )
        
        # Check 3: Circular data structures
        circular_data_patterns = [
            (r'\.parent\b.*\.children\b|\.children\b.*\.parent\b', "tree structure"),
            (r'\.next\b.*\.prev\b|\.prev\b.*\.next\b', "doubly-linked list"),
            (r'\.owner\b.*\.owned\b|\.owned\b.*\.owner\b', "ownership graph"),
        ]
        for pattern, description in circular_data_patterns:
            if re.search(pattern, content_lower):
                return (
                    True,
                    UnbreakableReason.CIRCULAR_DATA_STRUCTURE,
                    f"Detected circular data structure ({description}). "
                    "Bidirectional references in data models are often intentional and necessary. "
                    "Consider if the cycle is in the data model (acceptable) vs. the module imports (problematic)."
                )
        
        # Check 4: Framework requirements (common patterns that require cycles)
        framework_patterns = [
            (r'\bINotifyPropertyChanged\b', "WPF/MVVM data binding"),
            (r'\bOnPropertyChanged\b', "WPF/MVVM data binding"),
            (r'\bViewModelBase\b.*\bView\b', "MVVM pattern"),
            (r'@Component.*@Autowired|@Autowired.*@Component', "Spring circular injection"),
            (r'\bDbContext\b.*\bDbSet\b', "Entity Framework navigation"),
        ]
        for pattern, description in framework_patterns:
            if re.search(pattern, all_content, re.IGNORECASE):
                return (
                    True,
                    UnbreakableReason.FRAMEWORK_REQUIREMENT,
                    f"Detected framework pattern that may require this cycle ({description}). "
                    "Some frameworks intentionally create bidirectional relationships. "
                    "Check if this is a framework convention before attempting to break it."
                )
        
        # Check 5: Repeated validation failures with same issues (after multiple attempts)
        if validation_history and len(validation_history) >= 2:
            # Check if the same issues keep appearing
            issues_per_run = []
            for val in validation_history:
                issues = val.get("issues", [])
                issue_comments = frozenset(
                    i.get("comment", "")[:50] for i in issues if isinstance(i, dict)
                )
                issues_per_run.append(issue_comments)
            
            # If the same issues appear in all runs, it's likely unbreakable
            if issues_per_run:
                common_issues = issues_per_run[0]
                for issues in issues_per_run[1:]:
                    common_issues = common_issues.intersection(issues)
                
                if len(common_issues) > 0 and any("cycle" in i.lower() or "dependency" in i.lower() for i in common_issues):
                    return (
                        True,
                        UnbreakableReason.TIGHTLY_COUPLED_BY_DESIGN,
                        f"After {len(validation_history)} attempts, the same core issues remain. "
                        f"Common issues: {', '.join(list(common_issues)[:2])}. "
                        "This suggests the cycle may be intentional or require a more fundamental redesign "
                        "that is beyond the scope of automated refactoring."
                    )
        
        # Check 6: External dependencies (if files reference things we can't modify)
        external_patterns = [
            r'from\s+(System|Microsoft|Google|Amazon)\.',
            r'using\s+(System|Microsoft|Google|Amazon)\.',
            r'import\s+(java\.lang|javax\.|org\.springframework)',
        ]
        external_refs = set()
        for pattern in external_patterns:
            matches = re.findall(pattern, all_content)
            external_refs.update(matches)
        
        # If cycle nodes include terms that look like external packages
        for node in nodes:
            if any(ext in node for ext in ["System.", "Microsoft.", "javax.", "org."]):
                return (
                    True,
                    UnbreakableReason.EXTERNAL_DEPENDENCY,
                    f"Cycle includes external/framework code that cannot be modified: {node}. "
                    "Only internal code can be refactored. Consider wrapping the external dependency."
                )
        
        # No unbreakable pattern detected
        return (False, UnbreakableReason.NOT_UNBREAKABLE, "")

    def get_unbreakable_explanation(
        self, reason: UnbreakableReason, cycle_spec: Dict[str, Any]
    ) -> str:
        """Generate a detailed explanation for why a cycle cannot be broken."""
        nodes = cycle_spec.get("graph", {}).get("nodes", [])
        node_list = ", ".join(nodes[:5])
        
        explanations = {
            UnbreakableReason.MUTUAL_RECURSION: f"""
## Cycle Cannot Be Broken: Mutual Recursion

The cycle between **{node_list}** represents a fundamental algorithmic pattern 
where components must reference each other to function correctly.

### Why This Is Not Technical Debt
- Mutual recursion is a valid design pattern for certain algorithms
- Parser/Lexer, Visitor/Node, and Expression evaluators require bidirectional references
- Attempting to break this would fundamentally change the algorithm's behavior

### Recommended Actions
1. **Document** this as an intentional architectural decision
2. **Annotate** the code to suppress cycle detection warnings
3. **Consider** if the cycle detection scope should exclude these patterns
""",
            UnbreakableReason.CIRCULAR_DATA_STRUCTURE: f"""
## Cycle Cannot Be Broken: Circular Data Structure

The cycle between **{node_list}** is part of a data model that inherently 
requires bidirectional references (like parent-child in trees).

### Why This Is Acceptable
- Data structures often need navigation in both directions
- This is at the data model level, not the module/package level
- ORM frameworks and serializers handle these patterns correctly

### Recommended Actions
1. **Verify** the cycle is in the data model, not imports
2. **Document** the data structure's intentional circularity
3. **Consider** if the analysis should focus on package-level cycles only
""",
            UnbreakableReason.FRAMEWORK_REQUIREMENT: f"""
## Cycle Cannot Be Broken: Framework Requirement

The cycle between **{node_list}** appears to be required by the 
framework or library being used.

### Common Framework Patterns That Create Cycles
- MVVM: View ↔ ViewModel data binding
- Entity Framework: Navigation properties
- Spring: Circular dependency injection
- Event-driven: Publisher ↔ Subscriber

### Recommended Actions
1. **Verify** this is indeed a framework pattern
2. **Check** framework documentation for best practices
3. **Consider** if the framework provides alternatives (e.g., lazy injection)
""",
            UnbreakableReason.INCOMPLETE_CONTEXT: f"""
## Cannot Analyze: Incomplete Context

The file content provided for **{node_list}** is insufficient 
to safely generate a refactoring.

### What's Missing
- File content may be truncated or empty
- Key imports or class definitions are not visible
- Cannot determine the full dependency structure

### Recommended Actions
1. **Provide** complete file content (not truncated)
2. **Include** all files in the cycle, not just some
3. **Retry** with the full source files
""",
            UnbreakableReason.EXTERNAL_DEPENDENCY: f"""
## Cycle Cannot Be Broken: External Dependency

The cycle includes **{node_list}**, which references external 
or framework code that cannot be modified.

### Why This Cannot Be Automated
- External libraries are not under your control
- Framework patterns may require specific relationships
- Modifying external code would break updates/compatibility

### Recommended Actions
1. **Wrap** the external dependency with an internal abstraction
2. **Create** an anti-corruption layer if needed
3. **Accept** this as a boundary condition of the architecture
""",
            UnbreakableReason.TIGHTLY_COUPLED_BY_DESIGN: f"""
## Cycle May Be Intentional: Tight Coupling By Design

After multiple refactoring attempts, the cycle between **{node_list}** 
persists. This suggests it may be an intentional design decision.

### Possible Reasons
- The components are intentionally cohesive
- Breaking the cycle would require a fundamental redesign
- The cost of breaking exceeds the maintenance benefit

### Recommended Actions
1. **Review** with the original authors if possible
2. **Evaluate** if the cycle actually causes problems
3. **Consider** accepting it with documentation
4. **Plan** a larger refactoring effort if truly needed
""",
        }
        
        return explanations.get(reason, f"Cycle between {node_list} could not be broken.")



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
