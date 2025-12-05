from typing import Dict, Any, List, Optional, Union, Set, Tuple
import json
import re
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, QueryIntent
from models.schemas import (
    CycleSpec,
    CycleDescription,
    RefactorProposal,
    ValidationReport,
    ValidationIssue,
)

logger = get_logger("validator")


class IssueSeverity:
    """Issue severity levels for prioritization."""
    CRITICAL = "critical"  # Blocks approval (syntax errors, broken imports)
    MAJOR = "major"        # Strongly suggests rejection (cycle not broken)
    MINOR = "minor"        # Suggestions for improvement (style, naming)
    INFO = "info"          # Observations, not issues


class ValidatorAgent(Agent):
    """Validates refactor proposals and provides actionable feedback.

    This agent:
    1. Performs rule-based checks (syntax, imports, cycle impact)
    2. Uses LLM for semantic validation
    3. Categorizes issues by severity
    4. Provides specific, actionable suggestions for retry
    """

    name = "validator"
    version = "0.4"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        linters: Dict[str, str] = None,
        test_command: str = None,
        max_file_chars: int = 4000,
        rag_service=None,
    ):
        """
        Args:
            llm: Optional LLM client for semantic review.
            prompt_template: Path to prompt file or inline template string.
            linters: Mapping of linter name to shell command (future integration).
            test_command: Shell command to run tests (future integration).
            max_file_chars: Truncation limit per file snippet.
            rag_service: Optional RAG service for validation criteria.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.linters = linters or {}
        self.test_command = test_command
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
        self.query_builder = RAGQueryBuilder()

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        proposal: RefactorProposal,
    ) -> str:
        """Construct a prompt for the LLM to review the refactor proposal.
        
        Optimized for limited context: sends diffs instead of full patched files.
        """
        file_paths = cycle.get_file_paths()

        # Build DIFF snippets instead of full patched files (more context-efficient)
        diff_snippets = []
        total_chars = 0
        max_diff_chars = self.max_file_chars * 2  # Allow more room for diffs
        
        for p in proposal.patches:
            diff = p.diff or ""
            if not diff:
                continue
            
            if total_chars + len(diff) > max_diff_chars:
                remaining = len([x for x in proposal.patches if x.diff])
                diff_snippets.append(f"... [{remaining} more file diffs truncated for context limit]")
                break
            
            diff_snippets.append(f"### {p.path}\n```diff\n{diff}\n```")
            total_chars += len(diff)

        diffs_text = "\n\n".join(diff_snippets) if diff_snippets else "(no changes detected)"
        
        # Count what changed for context
        files_changed = len([p for p in proposal.patches if p.diff])
        files_unchanged = len(proposal.patches) - files_changed

        # If we have a template file/path, load it
        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            return safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                patched_files=diffs_text,  # Now contains diffs, not full files
                diffs=diffs_text,
                files_changed=files_changed,
                files_unchanged=files_unchanged,
            )

        # Default prompt - optimized for limited context
        return f"""You are validating a refactor proposal to break a cyclic dependency.

## Cycle
- ID: {cycle.id}
- Nodes: {', '.join(cycle.graph.nodes)}
- Edges: {cycle.graph.edges}

## Problem Description
{description.text[:1500] if len(description.text) > 1500 else description.text}

## Changes ({files_changed} files modified, {files_unchanged} unchanged)
{diffs_text}

## Validation Checklist
1. **Cycle Broken?** Do the changes remove/invert the problematic dependency?
2. **Syntax Valid?** Any unbalanced brackets, missing imports, or broken references?
3. **Complete?** Are there remaining cyclic paths that weren't addressed?
4. **Functionality Preserved?** Does the refactor maintain existing behavior?

## Output (JSON only)
{{"decision": "APPROVED"|"NEEDS_REVISION", "summary": "...", "issues": [{{"path": "...", "line": null, "comment": "...", "severity": "critical|major|minor"}}], "suggestions": ["<specific actionable fix>"]}}
"""

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse the LLM JSON response; return dict or fallback."""
        # Try to extract JSON from the response (model may wrap in markdown)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: treat entire text as notes
        return {
            "decision": "NEEDS_REVISION",
            "summary": "Could not parse LLM response; manual review required.",
            "issues": [],
            "suggestions": [],
            "raw_response": text,
        }

    # -------------------------------------------------------------------------
    # Issue categorization and actionable suggestions
    # -------------------------------------------------------------------------

    def _categorize_issue_severity(self, issue: ValidationIssue) -> str:
        """Classify issue severity for prioritization."""
        comment_lower = issue.comment.lower()
        
        # Critical: blocks approval
        critical_patterns = [
            "syntax error", "unbalanced", "missing import", "broken reference",
            "undefined", "not found", "cannot resolve"
        ]
        for pattern in critical_patterns:
            if pattern in comment_lower:
                return IssueSeverity.CRITICAL
        
        # Major: strongly suggests rejection
        major_patterns = [
            "cycle not broken", "dependency still exists", "incomplete",
            "no changes", "original unchanged", "still imports"
        ]
        for pattern in major_patterns:
            if pattern in comment_lower:
                return IssueSeverity.MAJOR
        
        # Minor: suggestions
        minor_patterns = [
            "naming", "style", "convention", "could be improved",
            "consider", "optional"
        ]
        for pattern in minor_patterns:
            if pattern in comment_lower:
                return IssueSeverity.MINOR
        
        return IssueSeverity.MAJOR  # Default to major for unknown issues

    def _generate_actionable_suggestions(
        self,
        issues: List[ValidationIssue],
        proposal: RefactorProposal,
        cycle: CycleSpec,
    ) -> List[str]:
        """Generate specific, actionable suggestions based on issues."""
        suggestions = []
        
        # Group issues by type
        has_no_changes = any("no changes" in i.comment.lower() for i in issues)
        has_cycle_issue = any("cycle" in i.comment.lower() for i in issues)
        has_syntax_issue = any("syntax" in i.comment.lower() or "unbalanced" in i.comment.lower() for i in issues)
        has_import_issue = any("import" in i.comment.lower() for i in issues)
        
        if has_no_changes:
            # Find which files weren't changed
            unchanged_files = [p.path for p in proposal.patches if not p.diff]
            if unchanged_files:
                file_list = ", ".join(unchanged_files[:3])
                suggestions.append(
                    f"Files {file_list} were not modified. To break the cycle, you must either: "
                    f"(1) Extract an interface from one of the cycle members, "
                    f"(2) Move shared code to a new module, or "
                    f"(3) Remove the problematic import by restructuring the code."
                )
        
        if has_cycle_issue and not has_no_changes:
            nodes = cycle.graph.nodes
            if len(nodes) == 2:
                suggestions.append(
                    f"The bidirectional dependency between {nodes[0]} and {nodes[1]} still exists. "
                    f"Create an interface (e.g., I{nodes[0]}Service) that {nodes[0]} implements, "
                    f"and have {nodes[1]} depend on the interface instead of the concrete class."
                )
            else:
                suggestions.append(
                    f"The cycle still exists. Look for the specific import statement in one of the cycle members "
                    f"that creates the problematic edge, and either remove it, replace it with an interface, "
                    f"or move the shared functionality to a common module."
                )
        
        if has_syntax_issue:
            suggestions.append(
                "Fix the syntax errors first. Ensure all brackets are balanced, "
                "all statements are properly terminated, and the code is syntactically valid."
            )
        
        if has_import_issue:
            suggestions.append(
                "Check that all imports are correct after refactoring. "
                "If you extracted an interface or moved code, update the import statements accordingly."
            )
        
        # Add general guidance if no specific suggestions
        if not suggestions:
            suggestions.append(
                "Review the issues above and ensure your refactoring: "
                "(1) Actually modifies the problematic files, "
                "(2) Removes or inverts the dependency that creates the cycle, "
                "(3) Maintains valid syntax and imports."
            )
        
        return suggestions

    # -------------------------------------------------------------------------
    # Rule-based checks (used when no LLM or as additional layer)
    # -------------------------------------------------------------------------

    def _extract_imports_csharp(self, content: str) -> Set[str]:
        """Extract 'using' statements from C# code."""
        imports = set()
        for match in re.finditer(r'^\s*using\s+([\w.]+)\s*;', content, re.MULTILINE):
            imports.add(match.group(1))
        return imports
    
    def _extract_imports_python(self, content: str) -> Set[str]:
        """Extract import statements from Python code."""
        imports = set()
        # import X, from X import Y
        for match in re.finditer(r'^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)', content, re.MULTILINE):
            if match.group(1):
                imports.add(match.group(1))
            for mod in match.group(2).split(','):
                imports.add(mod.strip().split()[0])  # Handle 'import X as Y'
        return imports
    
    def _extract_type_references(self, content: str, nodes: List[str]) -> Set[str]:
        """Find references to cycle nodes in code content."""
        refs = set()
        for node in nodes:
            # Look for the node name as a type reference (class, interface, etc.)
            # Match: ClassName, IClassName, _className, etc.
            pattern = rf'\b{re.escape(node)}\b'
            if re.search(pattern, content):
                refs.add(node)
        return refs
    
    def _check_syntax_errors(self, path: str, content: str) -> List[ValidationIssue]:
        """Check for obvious syntax errors without external tools."""
        issues = []
        
        # Determine language from extension
        ext = path.split('.')[-1].lower() if '.' in path else ''
        
        if ext == 'py':
            # Python: check for unbalanced brackets/parens
            issues.extend(self._check_bracket_balance(path, content, 
                [('(', ')'), ('[', ']'), ('{', '}')]))
            # Check for common Python syntax issues
            if re.search(r'^\s*def\s+\w+[^:]*$', content, re.MULTILINE):
                issues.append(ValidationIssue(
                    path=path, line=None, 
                    comment="Possible missing colon after function definition",
                    severity="major",
                    issue_type="syntax"
                ))
        
        elif ext == 'cs':
            # C#: check for unbalanced braces
            issues.extend(self._check_bracket_balance(path, content, 
                [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]))
            # Check for missing semicolons (heuristic)
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Lines that should end with ; but don't
                if (stripped and 
                    not stripped.endswith((';', '{', '}', ')', ',', '//')) and
                    not stripped.startswith(('if', 'else', 'for', 'while', 'using', 'namespace', 'class', 'interface', 'public', 'private', 'protected', '//', '/*', '*', '#')) and
                    not stripped.endswith('=>') and
                    '=' in stripped and
                    not stripped.endswith('{')):
                    # This is a heuristic, may have false positives
                    pass  # Too noisy, skip for now
        
        return issues
    
    def _check_bracket_balance(self, path: str, content: str, 
                                pairs: List[Tuple[str, str]]) -> List[ValidationIssue]:
        """Check if brackets/braces are balanced."""
        issues = []
        for open_char, close_char in pairs:
            # Simple count (doesn't handle strings/comments, but catches obvious errors)
            open_count = content.count(open_char)
            close_count = content.count(close_char)
            if open_count != close_count:
                issues.append(ValidationIssue(
                    path=path, line=None,
                    comment=f"Unbalanced '{open_char}{close_char}': {open_count} open, {close_count} close",
                    severity="critical",
                    issue_type="syntax"
                ))
        return issues
    
    def _analyze_cycle_impact(self, cycle: CycleSpec, proposal: RefactorProposal) -> Tuple[List[str], List[ValidationIssue]]:
        """Analyze whether the patches likely break the cycle.
        
        Returns:
            Tuple of (observations, issues)
        """
        observations = []
        issues = []
        nodes = set(cycle.graph.nodes)
        
        for patch in proposal.patches:
            if not patch.diff:
                continue
            
            original = patch.original or ""
            patched = patch.patched or ""
            
            # Check which nodes were referenced before and after
            refs_before = self._extract_type_references(original, cycle.graph.nodes)
            refs_after = self._extract_type_references(patched, cycle.graph.nodes)
            
            removed_refs = refs_before - refs_after
            added_refs = refs_after - refs_before
            
            if removed_refs:
                observations.append(f"{patch.path}: Removed references to {', '.join(removed_refs)}")
                logger.info(f"Cycle impact: {patch.path} removed refs to {removed_refs}")
            
            if added_refs:
                observations.append(f"{patch.path}: Added references to {', '.join(added_refs)}")
                # Adding new references might indicate the cycle isn't broken
                logger.warning(f"Cycle impact: {patch.path} added refs to {added_refs}")
            
            # Check for interface extraction pattern (common cycle-breaking technique)
            if re.search(r'\binterface\s+I\w+', patched) and not re.search(r'\binterface\s+I\w+', original):
                observations.append(f"{patch.path}: New interface defined (dependency inversion pattern)")
                logger.info(f"Detected interface extraction in {patch.path}")
        
        # If no references were removed, the cycle might not be broken
        if not any("Removed references" in obs for obs in observations):
            if any(patch.diff for patch in proposal.patches):
                issues.append(ValidationIssue(
                    path="(cycle analysis)",
                    line=None,
                    comment="Changes detected but no direct references between cycle nodes were removed. Verify cycle is actually broken.",
                    severity="major",
                    issue_type="cycle"
                ))
                ))
        
        return observations, issues

    def _rule_based_checks(
        self, cycle: CycleSpec, proposal: RefactorProposal
    ) -> Tuple[List[ValidationIssue], List[str]]:
        """Perform lightweight deterministic checks on the proposal.
        
        Returns:
            Tuple of (issues, observations)
        """
        issues: List[ValidationIssue] = []
        observations: List[str] = []
        known_paths = {f.path for f in cycle.files}

        logger.info("Running rule-based validation checks...")

        for p in proposal.patches:
            path = p.path
            
            # Check patch targets known file
            if path not in known_paths and path.split("/")[-1] not in {
                pth.split("/")[-1] for pth in known_paths
            }:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="Patch targets unknown file",
                                   severity="major", issue_type="semantic")
                )
                logger.warning(f"Patch targets unknown file: {path}")
            
            # Check non-empty patched content
            if not p.patched:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="Patched content is empty",
                                   severity="critical", issue_type="syntax")
                )
                logger.warning(f"Patch is empty: {path}")
            
            # Check diff exists (i.e., something changed)
            if p.original == p.patched:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="No changes detected in patch",
                                   severity="minor", issue_type="semantic")
                )
                logger.debug(f"No changes in patch: {path}")
            else:
                # Run syntax checks on changed files
                syntax_issues = self._check_syntax_errors(path, p.patched or "")
                if syntax_issues:
                    logger.warning(f"Syntax issues in {path}: {len(syntax_issues)}")
                issues.extend(syntax_issues)

        # Analyze cycle impact
        cycle_observations, cycle_issues = self._analyze_cycle_impact(cycle, proposal)
        observations.extend(cycle_observations)
        issues.extend(cycle_issues)
        
        # Log summary
        files_changed = len([p for p in proposal.patches if p.diff])
        logger.info(f"Rule-based checks complete: {len(issues)} issues, {files_changed} files changed")
        for obs in observations:
            logger.info(f"  Observation: {obs}")

        return issues, observations

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        proposal: Union[RefactorProposal, Dict[str, Any]] = None,
    ) -> AgentResult:
        """Validate the proposal and return approval or feedback.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            proposal: RefactorProposal model or dict from refactor agent.

        Returns:
            AgentResult with ValidationReport output.
        """
        logger.info("ValidatorAgent.run() starting")

        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            description = CycleDescription(text="")
        elif isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if proposal is None:
            logger.error("Missing proposal input")
            return AgentResult(
                status="error", output=None, logs="Missing proposal"
            )
        if isinstance(proposal, dict):
            proposal = RefactorProposal.model_validate(proposal)

        logger.debug(f"Validating proposal with {len(proposal.patches)} patches")

        # Rule-based checks first
        rule_issues, observations = self._rule_based_checks(cycle_spec, proposal)
        logger.debug(f"Rule-based checks found {len(rule_issues)} issues, {len(observations)} observations")

        # If no LLM, return rule-based result with actionable suggestions
        if self.llm is None:
            # Categorize issues by severity
            critical_issues = [i for i in rule_issues 
                             if self._categorize_issue_severity(i) == IssueSeverity.CRITICAL]
            major_issues = [i for i in rule_issues 
                          if self._categorize_issue_severity(i) == IssueSeverity.MAJOR]
            
            # Approve only if no critical or major issues
            approved = len(critical_issues) == 0 and len(major_issues) == 0
            logger.info(f"No LLM provided, rule-based validation: approved={approved} "
                       f"(critical={len(critical_issues)}, major={len(major_issues)})")
            
            # Generate actionable suggestions
            suggestions = self._generate_actionable_suggestions(rule_issues, proposal, cycle_spec)
            
            # Include observations in summary for visibility
            summary = "Rule-based validation only (no LLM provided)."
            if observations:
                summary += " Observations: " + "; ".join(observations[:3])
            
            report = ValidationReport(
                approved=approved,
                decision="APPROVED" if approved else "NEEDS_REVISION",
                summary=summary,
                issues=rule_issues,
                suggestions=suggestions,
            )
            return AgentResult(status="success", output=report.model_dump())

        # LLM-based review
        prompt = self._build_prompt(cycle_spec, description, proposal)
        
        # Append observations to prompt if any
        if observations:
            prompt += "\n\nPreliminary observations from static analysis:\n"
            for obs in observations:
                prompt += f"- {obs}\n"
        
        logger.debug(f"Built validation prompt with {len(prompt)} chars")

        try:
            logger.info("Calling LLM for semantic validation")
            response = call_llm(self.llm, prompt)
            text = response if isinstance(response, str) else json.dumps(response)
            parsed = self._parse_llm_response(text)
            logger.debug(f"LLM decision: {parsed.get('decision', 'unknown')}")

            # Merge rule-based issues with parsed issues
            parsed_issues = [
                ValidationIssue.model_validate(i) if isinstance(i, dict) else i
                for i in parsed.get("issues", [])
            ]
            all_issues = rule_issues + parsed_issues
            
            # Categorize all issues by severity
            critical_count = sum(1 for i in all_issues 
                               if self._categorize_issue_severity(i) == IssueSeverity.CRITICAL)
            major_count = sum(1 for i in all_issues 
                            if self._categorize_issue_severity(i) == IssueSeverity.MAJOR)
            
            decision = parsed.get("decision", "NEEDS_REVISION")
            # Override approval if there are critical/major issues
            approved = decision == "APPROVED" and critical_count == 0 and major_count == 0

            # Generate actionable suggestions if not approved
            llm_suggestions = parsed.get("suggestions", [])
            if not approved and not llm_suggestions:
                llm_suggestions = self._generate_actionable_suggestions(
                    all_issues, proposal, cycle_spec
                )

            logger.info(f"ValidatorAgent completed: decision={decision}, approved={approved}, "
                       f"issues={len(all_issues)} (critical={critical_count}, major={major_count})")
            report = ValidationReport(
                approved=approved,
                decision="APPROVED" if approved else "NEEDS_REVISION",
                summary=parsed.get("summary", ""),
                issues=all_issues,
                suggestions=llm_suggestions,
            )
            return AgentResult(status="success", output=report.model_dump())

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(
                status="error", output=None, logs=f"LLM call failed: {e}"
            )
