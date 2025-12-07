"""Cycle Detector Agent - Detects cycles in dependency graphs.

This agent takes a DependencyGraph and identifies all cyclic dependencies,
classifying them by severity and providing enough context for the refactoring
pipeline to address them.

This is an optional pre-pipeline step used together with DependencyAnalyzerAgent.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from .agent_base import Agent, AgentResult
from models.schemas import (
    DependencyGraph, 
    DetectedCycle, 
    CycleSpec, 
    GraphSpec,
    AnalysisResult,
)
from utils.graph_cycle_detector import (
    find_cycles,
    get_cycle_summary,
    CycleDetectorConfig,
)
from utils.logging import get_logger

logger = get_logger("cycle_detector")


class CycleDetectorAgent(Agent):
    """Detects cyclic dependencies in a dependency graph.
    
    Takes a DependencyGraph (from DependencyAnalyzerAgent) and identifies
    all cycles, ranking them by severity. Outputs CycleSpec objects that
    can be fed into the existing refactoring pipeline.
    
    Usage:
        analyzer = DependencyAnalyzerAgent()
        detector = CycleDetectorAgent()
        
        analysis = analyzer.run({"project_dir": "/path/to/project"})
        cycles = detector.run({"graph": analysis.data["graph"]})
        
        for cycle_spec in cycles.data["cycle_specs"]:
            # Feed to refactoring pipeline
            orchestrator.run_pipeline(cycle_spec)
    """
    
    name = "cycle_detector"
    version = "1.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cycle detector agent.
        
        Args:
            config: Optional configuration dict with keys:
                - max_cycles: Maximum cycles to find (default 100)
                - min_severity: Minimum severity to report ("minor", "major", "critical")
                - include_minor: Whether to include minor cycles (default True)
        """
        self.config = config or {}
        self.max_cycles = self.config.get("max_cycles", 100)
        self.min_severity = self.config.get("min_severity", "minor")
        self.include_minor = self.config.get("include_minor", True)
    
    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Detect cycles in a dependency graph.
        
        Args:
            input_data: Dict containing:
                - graph (required): DependencyGraph to analyze
                - max_cycles (optional): Override max cycles to find
                - include_minor (optional): Include minor 2-node cycles
                - project_dir (optional): Base directory for relative paths
                
        Returns:
            AgentResult with:
                - success: True if detection completed
                - data: Contains 'cycles', 'cycle_specs', 'summary'
                - error: Error message if failed
        """
        start_time = time.time()
        
        # Get graph from input
        graph = input_data.get("graph")
        if graph is None:
            return AgentResult(
                status="error",
                output={"error": "graph is required"},
            )
        
        # Handle both DependencyGraph objects and dicts
        if isinstance(graph, dict):
            try:
                graph = DependencyGraph(**graph)
            except Exception as e:
                return AgentResult(
                    status="error",
                    output={"error": f"Invalid graph format: {e}"},
                )
        
        if not isinstance(graph, DependencyGraph):
            return AgentResult(
                status="error",
                output={"error": f"Expected DependencyGraph, got {type(graph).__name__}"},
            )
        
        # Get config overrides
        max_cycles = input_data.get("max_cycles", self.max_cycles)
        include_minor = input_data.get("include_minor", self.include_minor)
        project_dir = input_data.get("project_dir", graph.root_dir or "")
        
        logger.info(f"Detecting cycles in graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        logger.debug(f"Max cycles: {max_cycles}, include minor: {include_minor}")
        logger.debug(f"Min severity filter: {self.min_severity}")
        logger.debug(f"Project dir for file resolution: {project_dir or '(not set)'}")
        
        try:
            # Configure detector
            detector_config = CycleDetectorConfig(
                max_cycles=max_cycles,
                include_minor=include_minor,
            )
            
            # Run cycle detection
            detected_cycles = find_cycles(graph, config=detector_config)
            
            # Filter by minimum severity if configured
            severity_order = {"minor": 0, "major": 1, "critical": 2}
            min_severity_level = severity_order.get(self.min_severity, 0)
            
            filtered_cycles = [
                c for c in detected_cycles
                if severity_order.get(c.severity, 0) >= min_severity_level
            ]
            
            if len(filtered_cycles) < len(detected_cycles):
                logger.info(f"Filtered {len(detected_cycles) - len(filtered_cycles)} cycles below min_severity='{self.min_severity}'")
            
            # Convert to CycleSpecs for pipeline
            logger.debug(f"Converting {len(filtered_cycles)} cycles to CycleSpec objects...")
            cycle_specs = self._convert_to_cycle_specs(filtered_cycles, project_dir)
            
            if len(cycle_specs) < len(filtered_cycles):
                logger.warning(f"Failed to convert {len(filtered_cycles) - len(cycle_specs)} cycles to CycleSpec")
            
            # Generate summary
            summary = get_cycle_summary(detected_cycles)
            
            elapsed = time.time() - start_time
            
            logger.info(f"Detection complete in {elapsed:.2f}s")
            logger.info(f"Found {len(detected_cycles)} cycles "
                       f"({len(filtered_cycles)} after filtering)")
            
            return AgentResult(
                status="success",
                output={
                    "cycles": filtered_cycles,
                    "cycle_specs": cycle_specs,
                    "summary": summary,
                    "total_found": len(detected_cycles),
                    "after_filter": len(filtered_cycles),
                    "detection_time_seconds": elapsed,
                },
            )
            
        except Exception as e:
            logger.error(f"Cycle detection failed: {type(e).__name__}: {e}")
            logger.debug("Exception details", exc_info=True)
            return AgentResult(
                status="error",
                output={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "detection_time_seconds": time.time() - start_time,
                },
            )
    
    def _convert_to_cycle_specs(
        self, 
        cycles: List[DetectedCycle], 
        project_dir: str,
    ) -> List[CycleSpec]:
        """Convert DetectedCycle objects to CycleSpec for pipeline.
        
        Args:
            cycles: List of detected cycles
            project_dir: Base directory for file paths
            
        Returns:
            List of CycleSpec objects ready for refactoring pipeline
        """
        cycle_specs = []
        files_read = 0
        files_missing = 0
        
        for cycle in cycles:
            try:
                # Use the to_cycle_spec method from DetectedCycle
                spec = cycle.to_cycle_spec(project_dir)
                if spec:
                    # Track file loading stats
                    for f in spec.files:
                        if f.content:
                            files_read += 1
                        else:
                            files_missing += 1
                    cycle_specs.append(spec)
            except Exception as e:
                logger.warning(f"Failed to convert cycle {cycle.id}: {type(e).__name__}: {e}")
                continue
        
        if files_missing > 0:
            logger.warning(f"CycleSpec conversion: {files_missing} files could not be read from disk")
        logger.debug(f"CycleSpec conversion: {len(cycle_specs)} specs created, {files_read} files loaded")
        
        return cycle_specs
    
    def get_prioritized_cycles(
        self, 
        cycles: List[DetectedCycle],
        strategy: str = "severity_first",
    ) -> List[DetectedCycle]:
        """Sort cycles by priority for refactoring order.
        
        Args:
            cycles: List of detected cycles
            strategy: Prioritization strategy:
                - "severity_first": Critical > Major > Minor
                - "size_first": Smaller cycles first (easier to fix)
                - "impact_first": Based on number of affected files
                
        Returns:
            Sorted list of cycles
        """
        if strategy == "severity_first":
            severity_order = {"critical": 0, "major": 1, "minor": 2}
            return sorted(cycles, key=lambda c: (
                severity_order.get(c.severity, 2),
                len(c.nodes),  # Smaller cycles first within severity
            ))
        
        elif strategy == "size_first":
            return sorted(cycles, key=lambda c: len(c.nodes))
        
        elif strategy == "impact_first":
            # Sort by total edges (more connections = higher impact)
            return sorted(cycles, key=lambda c: len(c.edges), reverse=True)
        
        else:
            # Default: severity first
            return self.get_prioritized_cycles(cycles, "severity_first")
    
    def analyze_and_detect(
        self, 
        project_dir: str,
        analyzer_config: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Convenience method: analyze project AND detect cycles in one call.
        
        This combines DependencyAnalyzerAgent and CycleDetectorAgent into
        a single operation for simpler usage.
        
        Args:
            project_dir: Path to project root
            analyzer_config: Optional config for dependency analysis
            
        Returns:
            AgentResult with graph, cycles, and cycle_specs
        """
        from .dependency_analyzer import DependencyAnalyzerAgent
        
        # Step 1: Analyze project
        analyzer = DependencyAnalyzerAgent(analyzer_config)
        analysis_result = analyzer.run({"project_dir": project_dir})
        
        if analysis_result.status != "success":
            return AgentResult(
                status="error",
                output={"error": f"Dependency analysis failed: {analysis_result.output.get('error', 'unknown')}"},
            )
        
        # Step 2: Detect cycles
        graph = analysis_result.output["graph"]
        detection_result = self.run({
            "graph": graph,
            "project_dir": project_dir,
        })
        
        if detection_result.status != "success":
            return AgentResult(
                status="error",
                output={"error": f"Cycle detection failed: {detection_result.output.get('error', 'unknown')}", "graph": graph},
            )
        
        # Combine results
        return AgentResult(
            status="success",
            output={
                "graph": graph,
                "project_info": analysis_result.output.get("project_info"),
                "cycles": detection_result.output.get("cycles", []),
                "cycle_specs": detection_result.output.get("cycle_specs", []),
                "summary": detection_result.output.get("summary", {}),
            },
        )
