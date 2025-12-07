"""Compile/lint checking utilities.

This module provides language-specific compilation and linting
to validate patched code before applying changes.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
import shutil
import os
import re

from utils.logging import get_logger

logger = get_logger("compile_checker")


@dataclass
class CompileError:
    """A compilation or lint error."""
    file: str
    line: Optional[int]
    column: Optional[int]
    message: str
    severity: str  # "error", "warning", "info"
    code: Optional[str] = None  # Error code (e.g., CS0103, E0001)
    
    def __str__(self) -> str:
        loc = f"{self.file}"
        if self.line:
            loc += f":{self.line}"
            if self.column:
                loc += f":{self.column}"
        return f"[{self.severity.upper()}] {loc}: {self.message}"


@dataclass
class CompileResult:
    """Result of a compile/lint check."""
    success: bool
    errors: List[CompileError] = field(default_factory=list)
    warnings: List[CompileError] = field(default_factory=list)
    output: str = ""
    tool_used: str = ""
    tool_available: bool = True
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def summary(self) -> str:
        if not self.tool_available:
            return f"Tool not available: {self.tool_used}"
        if self.success:
            return f"✓ Compiled successfully ({self.warning_count} warnings)"
        return f"✗ {self.error_count} errors, {self.warning_count} warnings"


class CompileChecker:
    """Multi-language compile/lint checker."""
    
    # Mapping of file extensions to language handlers
    LANGUAGE_MAP = {
        # C#
        '.cs': 'csharp',
        # Python
        '.py': 'python',
        # TypeScript/JavaScript
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        # Java
        '.java': 'java',
        # C/C++
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
    }
    
    def __init__(self, enabled: bool = True, timeout: int = 30):
        """Initialize compile checker.
        
        Args:
            enabled: Whether to actually run checks (False = skip)
            timeout: Timeout in seconds for compile commands
        """
        self.enabled = enabled
        self.timeout = timeout
        self._tool_cache: Dict[str, bool] = {}
    
    def _check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available on the system."""
        if tool in self._tool_cache:
            return self._tool_cache[tool]
        
        result = shutil.which(tool) is not None
        self._tool_cache[tool] = result
        
        if not result:
            logger.debug(f"Tool '{tool}' not found in PATH")
        
        return result
    
    def get_language(self, path: str) -> Optional[str]:
        """Get language from file path."""
        ext = Path(path).suffix.lower()
        return self.LANGUAGE_MAP.get(ext)
    
    def check_file(self, path: str, content: str) -> CompileResult:
        """Check a single file for compile/lint errors.
        
        Args:
            path: File path (for determining language)
            content: File content to check
            
        Returns:
            CompileResult with any errors found
        """
        if not self.enabled:
            return CompileResult(
                success=True,
                tool_used="disabled",
                tool_available=True,
            )
        
        language = self.get_language(path)
        if not language:
            logger.debug(f"No language handler for {path}")
            return CompileResult(
                success=True,
                tool_used="none",
                tool_available=True,
            )
        
        # Dispatch to language-specific handler
        handlers = {
            'python': self._check_python,
            'csharp': self._check_csharp,
            'typescript': self._check_typescript,
            'javascript': self._check_javascript,
            'java': self._check_java,
            'c': self._check_c,
            'cpp': self._check_cpp,
        }
        
        handler = handlers.get(language)
        if not handler:
            return CompileResult(
                success=True,
                tool_used=f"no_handler_{language}",
                tool_available=True,
            )
        
        return handler(path, content)
    
    def check_multiple(
        self, 
        files: List[Tuple[str, str]]
    ) -> Dict[str, CompileResult]:
        """Check multiple files.
        
        Args:
            files: List of (path, content) tuples
            
        Returns:
            Dict mapping path to CompileResult
        """
        results = {}
        for path, content in files:
            results[path] = self.check_file(path, content)
        return results
    
    def _run_command(
        self, 
        cmd: List[str], 
        cwd: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {' '.join(cmd)}")
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return -1, "", str(e)
    
    def _check_python(self, path: str, content: str) -> CompileResult:
        """Check Python file using py_compile and optional flake8."""
        errors = []
        warnings = []
        tool_used = "python"
        
        # First, use Python's built-in compile() for syntax check
        try:
            compile(content, path, 'exec')
        except SyntaxError as e:
            errors.append(CompileError(
                file=path,
                line=e.lineno,
                column=e.offset,
                message=e.msg or str(e),
                severity="error",
                code="SyntaxError",
            ))
            return CompileResult(
                success=False,
                errors=errors,
                warnings=warnings,
                tool_used=tool_used,
                tool_available=True,
            )
        
        # Optionally use flake8 for more checks
        if self._check_tool_available("flake8"):
            tool_used = "python+flake8"
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                returncode, stdout, stderr = self._run_command(
                    ["flake8", "--max-line-length=120", temp_path]
                )
                
                # Parse flake8 output: filename:line:col: code message
                for line in stdout.split('\n'):
                    if not line.strip():
                        continue
                    match = re.match(
                        r'^[^:]+:(\d+):(\d+): ([A-Z]\d+) (.+)$', 
                        line
                    )
                    if match:
                        line_no = int(match.group(1))
                        col = int(match.group(2))
                        code = match.group(3)
                        msg = match.group(4)
                        
                        # E/W are errors/warnings, F are fatal
                        severity = "error" if code.startswith(('E9', 'F')) else "warning"
                        issue = CompileError(
                            file=path,
                            line=line_no,
                            column=col,
                            message=msg,
                            severity=severity,
                            code=code,
                        )
                        if severity == "error":
                            errors.append(issue)
                        else:
                            warnings.append(issue)
            finally:
                os.unlink(temp_path)
        
        return CompileResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_used=tool_used,
            tool_available=True,
        )
    
    def _check_csharp(self, path: str, content: str) -> CompileResult:
        """Check C# file using Roslyn syntax check via dotnet."""
        errors = []
        warnings = []
        
        # Try using dotnet CLI to check syntax
        if not self._check_tool_available("dotnet"):
            return CompileResult(
                success=True,
                tool_used="dotnet",
                tool_available=False,
            )
        
        # Create a minimal project structure for syntax checking
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal csproj
            csproj_content = '''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>'''
            
            csproj_path = os.path.join(temp_dir, "check.csproj")
            with open(csproj_path, 'w') as f:
                f.write(csproj_content)
            
            # Write the C# file
            cs_path = os.path.join(temp_dir, os.path.basename(path))
            with open(cs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Run dotnet build --no-restore (faster, just checks syntax)
            returncode, stdout, stderr = self._run_command(
                ["dotnet", "build", "--no-restore", "-v", "q"],
                cwd=temp_dir
            )
            
            # Parse output for errors
            output = stdout + stderr
            # Pattern: filename(line,col): error CS####: message
            for line in output.split('\n'):
                match = re.search(
                    r'[^(]+\((\d+),(\d+)\): (error|warning) (CS\d+): (.+)$',
                    line
                )
                if match:
                    line_no = int(match.group(1))
                    col = int(match.group(2))
                    severity = match.group(3)
                    code = match.group(4)
                    msg = match.group(5)
                    
                    issue = CompileError(
                        file=path,
                        line=line_no,
                        column=col,
                        message=msg,
                        severity=severity,
                        code=code,
                    )
                    if severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
        
        return CompileResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            output=output if errors or warnings else "",
            tool_used="dotnet",
            tool_available=True,
        )
    
    def _check_typescript(self, path: str, content: str) -> CompileResult:
        """Check TypeScript file using tsc."""
        if not self._check_tool_available("tsc"):
            return CompileResult(
                success=True,
                tool_used="tsc",
                tool_available=False,
            )
        
        errors = []
        warnings = []
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.ts',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            returncode, stdout, stderr = self._run_command(
                ["tsc", "--noEmit", "--skipLibCheck", temp_path]
            )
            
            output = stdout + stderr
            # Pattern: filename(line,col): error TS####: message
            for line in output.split('\n'):
                match = re.search(
                    r'[^(]+\((\d+),(\d+)\): (error|warning) (TS\d+): (.+)$',
                    line
                )
                if match:
                    line_no = int(match.group(1))
                    col = int(match.group(2))
                    severity = match.group(3)
                    code = match.group(4)
                    msg = match.group(5)
                    
                    issue = CompileError(
                        file=path,
                        line=line_no,
                        column=col,
                        message=msg,
                        severity=severity,
                        code=code,
                    )
                    if severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
        finally:
            os.unlink(temp_path)
        
        return CompileResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_used="tsc",
            tool_available=True,
        )
    
    def _check_javascript(self, path: str, content: str) -> CompileResult:
        """Check JavaScript file using ESLint or simple parse."""
        errors = []
        warnings = []
        
        # Try ESLint first
        if self._check_tool_available("eslint"):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.js',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                returncode, stdout, stderr = self._run_command(
                    ["eslint", "--no-eslintrc", "--env", "es2021", 
                     "--parser-options", "ecmaVersion:latest", 
                     "-f", "json", temp_path]
                )
                
                import json
                try:
                    results = json.loads(stdout)
                    for result in results:
                        for msg in result.get("messages", []):
                            issue = CompileError(
                                file=path,
                                line=msg.get("line"),
                                column=msg.get("column"),
                                message=msg.get("message", ""),
                                severity="error" if msg.get("severity", 0) == 2 else "warning",
                                code=msg.get("ruleId"),
                            )
                            if issue.severity == "error":
                                errors.append(issue)
                            else:
                                warnings.append(issue)
                except json.JSONDecodeError:
                    pass
            finally:
                os.unlink(temp_path)
            
            return CompileResult(
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                tool_used="eslint",
                tool_available=True,
            )
        
        # Fallback: use Node.js to parse
        if self._check_tool_available("node"):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.js',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                returncode, stdout, stderr = self._run_command(
                    ["node", "--check", temp_path]
                )
                
                if returncode != 0:
                    # Parse Node.js syntax error
                    match = re.search(r':(\d+)\s*\n(.+Error:.+)', stderr)
                    if match:
                        errors.append(CompileError(
                            file=path,
                            line=int(match.group(1)),
                            column=None,
                            message=match.group(2).strip(),
                            severity="error",
                        ))
            finally:
                os.unlink(temp_path)
            
            return CompileResult(
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                tool_used="node",
                tool_available=True,
            )
        
        return CompileResult(
            success=True,
            tool_used="eslint|node",
            tool_available=False,
        )
    
    def _check_java(self, path: str, content: str) -> CompileResult:
        """Check Java file using javac."""
        if not self._check_tool_available("javac"):
            return CompileResult(
                success=True,
                tool_used="javac",
                tool_available=False,
            )
        
        errors = []
        warnings = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Java requires file name to match class name
            temp_path = os.path.join(temp_dir, os.path.basename(path))
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            returncode, stdout, stderr = self._run_command(
                ["javac", "-Xlint:all", temp_path]
            )
            
            output = stderr  # javac outputs to stderr
            # Pattern: filename:line: error: message
            for line in output.split('\n'):
                match = re.search(
                    r'[^:]+:(\d+): (error|warning): (.+)$',
                    line
                )
                if match:
                    line_no = int(match.group(1))
                    severity = match.group(2)
                    msg = match.group(3)
                    
                    issue = CompileError(
                        file=path,
                        line=line_no,
                        column=None,
                        message=msg,
                        severity=severity,
                    )
                    if severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
        
        return CompileResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_used="javac",
            tool_available=True,
        )
    
    def _check_c(self, path: str, content: str) -> CompileResult:
        """Check C file using gcc."""
        return self._check_c_cpp(path, content, "gcc", ["-std=c11"])
    
    def _check_cpp(self, path: str, content: str) -> CompileResult:
        """Check C++ file using g++."""
        return self._check_c_cpp(path, content, "g++", ["-std=c++17"])
    
    def _check_c_cpp(
        self, 
        path: str, 
        content: str, 
        compiler: str,
        extra_flags: List[str]
    ) -> CompileResult:
        """Check C/C++ file."""
        if not self._check_tool_available(compiler):
            return CompileResult(
                success=True,
                tool_used=compiler,
                tool_available=False,
            )
        
        errors = []
        warnings = []
        
        suffix = Path(path).suffix or '.cpp'
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            returncode, stdout, stderr = self._run_command(
                [compiler, "-fsyntax-only", "-Wall", "-Wextra"] + extra_flags + [temp_path]
            )
            
            output = stderr
            # Pattern: filename:line:col: error: message
            for line in output.split('\n'):
                match = re.search(
                    r'[^:]+:(\d+):(\d+): (error|warning): (.+)$',
                    line
                )
                if match:
                    line_no = int(match.group(1))
                    col = int(match.group(2))
                    severity = match.group(3)
                    msg = match.group(4)
                    
                    issue = CompileError(
                        file=path,
                        line=line_no,
                        column=col,
                        message=msg,
                        severity=severity,
                    )
                    if severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
        finally:
            os.unlink(temp_path)
        
        return CompileResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tool_used=compiler,
            tool_available=True,
        )


# Global instance for convenience
_default_checker: Optional[CompileChecker] = None


def get_compile_checker(enabled: bool = True) -> CompileChecker:
    """Get or create the default compile checker."""
    global _default_checker
    if _default_checker is None:
        _default_checker = CompileChecker(enabled=enabled)
    return _default_checker


def check_file_syntax(path: str, content: str) -> CompileResult:
    """Convenience function to check a single file."""
    return get_compile_checker().check_file(path, content)
