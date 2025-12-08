"""Scaffolding-first mode for interface extraction refactoring.

This module implements the scaffolding approach where new files (interfaces,
abstract classes, shared modules) are created BEFORE modifying existing files.

Benefits:
- Validates that the abstraction compiles before using it
- Prevents hallucination of non-existent types
- Enables incremental refactoring with checkpoints

Flow:
1. LLM generates scaffold (interface/abstraction) based on plan
2. Scaffold is syntax-checked and optionally compile-checked
3. If valid, existing files are updated to use the new abstraction
4. If invalid, we can report what was attempted without breaking anything
"""

import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logging import get_logger

logger = get_logger("scaffolding")


@dataclass
class ScaffoldResult:
    """Result of scaffolding phase."""
    success: bool = False
    files_created: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    syntax_valid: bool = True
    compile_valid: Optional[bool] = None
    raw_llm_response: str = ""


def detect_language(file_paths: List[str]) -> str:
    """Detect programming language from file paths.
    
    Args:
        file_paths: List of file paths in the cycle
        
    Returns:
        Language identifier: csharp, python, typescript, java, or unknown
    """
    extensions = {}
    for path in file_paths:
        ext = Path(path).suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    # Find most common extension
    if not extensions:
        return "unknown"
    
    most_common = max(extensions, key=extensions.get)
    
    ext_to_lang = {
        '.cs': 'csharp',
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.kt': 'kotlin',
    }
    
    return ext_to_lang.get(most_common, 'unknown')


def detect_namespace(file_contents: List[str], language: str) -> str:
    """Detect namespace/package from existing file contents.
    
    Args:
        file_contents: List of file content strings
        language: Programming language
        
    Returns:
        Detected namespace string, or empty if not found
    """
    for content in file_contents:
        if not content:
            continue
        
        if language == 'csharp':
            # Match: namespace Some.Namespace or namespace Some.Namespace;
            match = re.search(r'\bnamespace\s+([\w\.]+)', content)
            if match:
                return match.group(1)
        
        elif language == 'python':
            # Python doesn't have namespaces, but we can infer from imports
            # Look for common package patterns
            match = re.search(r'from\s+([\w\.]+)\s+import', content)
            if match:
                parts = match.group(1).split('.')
                if len(parts) > 1:
                    return '.'.join(parts[:-1])  # Parent package
        
        elif language == 'typescript' or language == 'javascript':
            # Look for common export patterns or file path hints
            # TypeScript doesn't require namespace declarations
            return ""  # Let LLM infer from file path
        
        elif language == 'java':
            match = re.search(r'\bpackage\s+([\w\.]+);', content)
            if match:
                return match.group(1)
    
    return ""


def extract_method_signatures(
    file_contents: Dict[str, str],  # path -> content
    target_class: str,
    language: str,
) -> List[Dict[str, Any]]:
    """Extract method signatures from a class that will be interfaced.
    
    This helps the LLM know exactly what methods the interface needs.
    
    Args:
        file_contents: Dict mapping file paths to content
        target_class: Name of the class to extract methods from
        language: Programming language
        
    Returns:
        List of method signature dicts with name, signature, return_type
    """
    methods = []
    
    for path, content in file_contents.items():
        if not content:
            continue
        
        if language == 'csharp':
            methods.extend(_extract_csharp_methods(content, target_class))
        elif language == 'python':
            methods.extend(_extract_python_methods(content, target_class))
        elif language in ('typescript', 'javascript'):
            methods.extend(_extract_typescript_methods(content, target_class))
        elif language == 'java':
            methods.extend(_extract_java_methods(content, target_class))
    
    return methods


def _extract_csharp_methods(content: str, target_class: str) -> List[Dict[str, Any]]:
    """Extract public method signatures from C# class."""
    methods = []
    
    # Find the class
    class_pattern = rf'\bclass\s+{re.escape(target_class)}\b[^{{]*\{{([^}}]*(?:\{{[^}}]*\}}[^}}]*)*)\}}'
    class_match = re.search(class_pattern, content, re.DOTALL)
    
    if not class_match:
        # Try simpler approach - find methods anywhere that look public
        method_pattern = r'\bpublic\s+(?:async\s+)?(?:virtual\s+)?(?:override\s+)?([\w<>\[\],\s]+)\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(method_pattern, content):
            return_type = match.group(1).strip()
            method_name = match.group(2)
            params = match.group(3).strip()
            
            # Skip constructors (return type == class name)
            if return_type == target_class or method_name == target_class:
                continue
            
            methods.append({
                'name': method_name,
                'return_type': return_type,
                'params': params,
                'signature': f"{return_type} {method_name}({params})",
            })
    
    return methods


def _extract_python_methods(content: str, target_class: str) -> List[Dict[str, Any]]:
    """Extract public method signatures from Python class."""
    methods = []
    
    # Find methods with type hints
    method_pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([\w\[\],\s\.]+))?:'
    
    for match in re.finditer(method_pattern, content):
        method_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3) or 'None'
        
        # Skip private/magic methods
        if method_name.startswith('_') and not method_name.startswith('__'):
            continue
        if method_name in ('__init__', '__new__', '__del__'):
            continue
        
        methods.append({
            'name': method_name,
            'return_type': return_type.strip(),
            'params': params,
            'signature': f"def {method_name}({params}) -> {return_type.strip()}",
        })
    
    return methods


def _extract_typescript_methods(content: str, target_class: str) -> List[Dict[str, Any]]:
    """Extract public method signatures from TypeScript class."""
    methods = []
    
    # Find methods
    method_pattern = r'(?:public\s+)?(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*([\w<>\[\],\s|]+))?'
    
    for match in re.finditer(method_pattern, content):
        method_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3) or 'void'
        
        # Skip constructor and private
        if method_name in ('constructor', 'private', 'protected'):
            continue
        
        methods.append({
            'name': method_name,
            'return_type': return_type.strip(),
            'params': params,
            'signature': f"{method_name}({params}): {return_type.strip()}",
        })
    
    return methods


def _extract_java_methods(content: str, target_class: str) -> List[Dict[str, Any]]:
    """Extract public method signatures from Java class."""
    methods = []
    
    method_pattern = r'\bpublic\s+(?:static\s+)?(?:final\s+)?([\w<>\[\],\s]+)\s+(\w+)\s*\(([^)]*)\)'
    
    for match in re.finditer(method_pattern, content):
        return_type = match.group(1).strip()
        method_name = match.group(2)
        params = match.group(3).strip()
        
        # Skip constructors
        if return_type == target_class or method_name == target_class:
            continue
        
        methods.append({
            'name': method_name,
            'return_type': return_type,
            'params': params,
            'signature': f"{return_type} {method_name}({params})",
        })
    
    return methods


def extract_interface_requirements(
    plan: Dict[str, Any],
    file_contents: Dict[str, str],
    language: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract method signatures needed for the interface based on the plan.
    
    Analyzes the plan to find which class is being interfaced and extracts
    its public methods.
    
    Args:
        plan: The refactoring plan
        file_contents: Dict of file path -> content
        language: Programming language
        
    Returns:
        Tuple of (method_signatures, target_class_name)
    """
    # Try to find the target class from the plan
    target_class = ""
    
    for change in plan.get("file_changes", []):
        changes = change.get("changes", [])
        for c in changes:
            c_lower = c.lower()
            # Look for patterns like "implement IFoo" or "use interface"
            impl_match = re.search(r'implement\s+i(\w+)', c_lower)
            if impl_match:
                target_class = impl_match.group(1)
                break
            
            # Look for "extract interface from Foo"
            extract_match = re.search(r'extract.*from\s+(\w+)', c_lower)
            if extract_match:
                target_class = extract_match.group(1)
                break
    
    if not target_class:
        # Try to infer from new_files
        for nf in plan.get("new_files", []):
            path = nf.get("path", "")
            # Interface files often named IClassName
            match = re.search(r'I([A-Z]\w+)', Path(path).stem)
            if match:
                target_class = match.group(1)
                break
    
    methods = []
    if target_class:
        methods = extract_method_signatures(file_contents, target_class, language)
        logger.info(f"Extracted {len(methods)} method signatures from {target_class}")
    
    return methods, target_class


def extract_scaffold_from_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract scaffold file specifications from a refactoring plan.
    
    Args:
        plan: The refactoring plan from planning phase
        
    Returns:
        List of scaffold file specs with path, purpose, content_description
    """
    scaffolds = []
    
    # Check for new_files in plan
    new_files = plan.get("new_files", [])
    for nf in new_files:
        path = nf.get("path", "")
        purpose = nf.get("purpose", "")
        content_desc = nf.get("content_description", "")
        
        # Only treat as scaffold if it's an interface/abstraction
        if any(kw in purpose.lower() or kw in content_desc.lower() 
               for kw in ["interface", "abstract", "contract", "shared", "common"]):
            scaffolds.append({
                "path": path,
                "purpose": purpose,
                "content_description": content_desc,
                "is_interface": "interface" in purpose.lower() or "interface" in content_desc.lower(),
            })
    
    return scaffolds


def validate_scaffold_syntax(content: str, path: str) -> Tuple[bool, List[str]]:
    """Validate scaffold file syntax based on language.
    
    Args:
        content: File content to validate
        path: File path (for language detection)
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    path_lower = path.lower()
    
    # Basic checks for all languages
    if not content or not content.strip():
        return False, ["Empty file content"]
    
    # Bracket balance check
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
    
    open_parens = content.count('(')
    close_parens = content.count(')')
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
    
    # Language-specific checks
    if path_lower.endswith('.cs'):
        errors.extend(_validate_csharp_scaffold(content))
    elif path_lower.endswith('.py'):
        errors.extend(_validate_python_scaffold(content))
    elif path_lower.endswith(('.ts', '.js')):
        errors.extend(_validate_typescript_scaffold(content))
    elif path_lower.endswith('.java'):
        errors.extend(_validate_java_scaffold(content))
    
    return len(errors) == 0, errors


def _validate_csharp_scaffold(content: str) -> List[str]:
    """Validate C# scaffold content."""
    errors = []
    
    # Check for interface/class definition
    if not re.search(r'\b(interface|class|struct|record)\s+\w+', content):
        errors.append("No interface, class, struct, or record definition found")
    
    # Check for namespace
    if not re.search(r'\bnamespace\s+[\w\.]+', content):
        errors.append("No namespace declaration found")
    
    # Check that interface methods end with semicolon
    interface_match = re.search(r'\binterface\s+\w+[^{]*\{([^}]*)\}', content, re.DOTALL)
    if interface_match:
        interface_body = interface_match.group(1)
        # Methods in interface should end with ;
        method_lines = [l.strip() for l in interface_body.split('\n') if l.strip() and not l.strip().startswith('//')]
        for line in method_lines:
            if re.match(r'.*\w+\s*\([^)]*\)\s*[^;{]', line):
                if not line.endswith(';') and not line.endswith('{'):
                    errors.append(f"Interface method may be missing semicolon: {line[:50]}")
    
    return errors


def _validate_python_scaffold(content: str) -> List[str]:
    """Validate Python scaffold content."""
    errors = []
    
    # Try to compile
    try:
        compile(content, '<scaffold>', 'exec')
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e.msg} at line {e.lineno}")
    
    # Check for class definition (ABC or Protocol)
    if not re.search(r'\bclass\s+\w+', content):
        errors.append("No class definition found")
    
    return errors


def _validate_typescript_scaffold(content: str) -> List[str]:
    """Validate TypeScript scaffold content."""
    errors = []
    
    # Check for interface/class/type definition
    if not re.search(r'\b(interface|class|type|abstract class)\s+\w+', content):
        errors.append("No interface, class, or type definition found")
    
    # Check for export
    if not re.search(r'\bexport\s+(interface|class|type|abstract)', content):
        errors.append("Interface/class should be exported")
    
    return errors


def _validate_java_scaffold(content: str) -> List[str]:
    """Validate Java scaffold content."""
    errors = []
    
    # Check for interface/class definition
    if not re.search(r'\b(interface|class|abstract class)\s+\w+', content):
        errors.append("No interface or class definition found")
    
    # Check for package
    if not re.search(r'\bpackage\s+[\w\.]+;', content):
        errors.append("No package declaration found")
    
    return errors


def compile_check_scaffold(
    path: str, 
    content: str, 
    project_root: str,
    timeout: int = 30
) -> Tuple[bool, List[str]]:
    """Run language-specific compile check on scaffold file.
    
    Args:
        path: Path where file would be created
        content: File content
        project_root: Project root directory
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (compile_success, error_messages)
    """
    path_lower = path.lower()
    errors = []
    
    # For now, we just do syntax validation
    # Full compile check would require writing temp files and running compilers
    # This can be extended based on project setup
    
    if path_lower.endswith('.py'):
        try:
            compile(content, path, 'exec')
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error: {e.msg} at line {e.lineno}"]
    
    # For other languages, syntax validation is sufficient for scaffolding
    is_valid, errors = validate_scaffold_syntax(content, path)
    return is_valid, errors


def build_scaffold_prompt(
    scaffold_specs: List[Dict[str, Any]],
    cycle_info: Dict[str, Any],
    file_snippets: str,
    method_signatures: List[Dict[str, Any]] = None,
    detected_namespace: str = "",
    language: str = "csharp",
) -> str:
    """Build prompt to generate scaffold file content.
    
    This prompt is designed to be specific enough for smaller LLMs by:
    1. Providing concrete method signatures to include
    2. Showing the exact namespace/package to use
    3. Including language-specific templates as examples
    
    Args:
        scaffold_specs: List of scaffold specifications from plan
        cycle_info: Cycle information dict
        file_snippets: Relevant file snippets for context
        method_signatures: Extracted method signatures the interface needs
        detected_namespace: Detected namespace from existing files
        language: Programming language (csharp, python, typescript, java)
        
    Returns:
        Prompt string for LLM
    """
    scaffold_list = []
    for i, spec in enumerate(scaffold_specs, 1):
        scaffold_list.append(f"""
### Scaffold {i}: {spec.get('path', 'unknown')}
- Purpose: {spec.get('purpose', 'Not specified')}
- Description: {spec.get('content_description', 'Not specified')}
- Is Interface: {spec.get('is_interface', False)}
""")
    
    scaffolds_str = "\n".join(scaffold_list)
    
    # Build method signatures section
    methods_section = ""
    if method_signatures:
        methods_list = []
        for sig in method_signatures:
            methods_list.append(f"  - {sig.get('signature', sig.get('name', 'unknown'))}")
        methods_section = f"""
## Method Signatures to Include
The interface MUST include these methods (extracted from existing code):
{chr(10).join(methods_list)}
"""
    
    # Language-specific template examples
    templates = _get_language_template(language, detected_namespace)
    
    return f"""You are creating scaffold files (interfaces/abstractions) for a refactoring.

## Cycle Being Fixed
{cycle_info.get('id', 'unknown')}

## Detected Namespace/Package
Use this namespace: `{detected_namespace or 'infer from existing files'}`

## Scaffold Files to Create
{scaffolds_str}
{methods_section}
## Existing Code Context
{file_snippets}

## Language: {language.upper()}
{templates}

## Your Task
Generate the COMPLETE content for each scaffold file. These files will be created
BEFORE modifying existing code, so they must be syntactically valid and complete.

## Output Format (JSON)
```json
{{
  "scaffold_files": [
    {{
      "path": "<file path>",
      "content": "<COMPLETE file content - properly formatted with newlines as \\n>",
      "purpose": "<interface|abstract_class|shared_module>",
      "exports": ["<list of exported types/functions>"]
    }}
  ],
  "notes": "<any implementation notes>"
}}
```

## CRITICAL Requirements
1. Each file must be COMPLETE and syntactically valid
2. Use the EXACT namespace/package from existing files: `{detected_namespace}`
3. Include ALL method signatures listed above
4. Follow the template structure for {language}
5. The content must be ready to save directly to a file

Output ONLY the JSON object.
"""


def _get_language_template(language: str, namespace: str) -> str:
    """Get language-specific template example."""
    
    if language == "csharp":
        return f"""
### C# Interface Template
```csharp
namespace {namespace or 'YourNamespace'}
{{
    /// <summary>
    /// Interface description here
    /// </summary>
    public interface IYourInterface
    {{
        // Method signatures - no implementation, just signature + semicolon
        ReturnType MethodName(ParamType param);
        
        // Properties
        PropertyType PropertyName {{ get; set; }}
    }}
}}
```
IMPORTANT for C#:
- Interface methods end with semicolon, NO body
- Use proper XML doc comments
- Match the namespace from existing files exactly
"""
    
    elif language == "python":
        return f"""
### Python Interface Template (using ABC)
```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from other_module import SomeType

class IYourInterface(ABC):
    \"\"\"Interface description here.\"\"\"
    
    @abstractmethod
    def method_name(self, param: ParamType) -> ReturnType:
        \"\"\"Method description.\"\"\"
        pass
    
    @property
    @abstractmethod
    def property_name(self) -> PropertyType:
        \"\"\"Property description.\"\"\"
        pass
```
IMPORTANT for Python:
- Use ABC and @abstractmethod
- Use TYPE_CHECKING for circular import prevention
- Include type hints
"""
    
    elif language == "typescript":
        return f"""
### TypeScript Interface Template
```typescript
// Interfaces/{namespace or 'YourInterface'}.ts

/**
 * Interface description here
 */
export interface IYourInterface {{
    // Method signatures
    methodName(param: ParamType): ReturnType;
    
    // Properties
    propertyName: PropertyType;
}}
```
IMPORTANT for TypeScript:
- Always export the interface
- Use proper JSDoc comments
- Semicolons after each member
"""
    
    elif language == "java":
        return f"""
### Java Interface Template
```java
package {namespace or 'com.example'};

/**
 * Interface description here
 */
public interface IYourInterface {{
    // Method signatures - no implementation
    ReturnType methodName(ParamType param);
    
    // Default methods allowed if needed
    default void helperMethod() {{
        // implementation
    }}
}}
```
IMPORTANT for Java:
- Match the package from existing files
- Interface methods are implicitly public abstract
"""
    
    return "Follow standard conventions for the language."


def parse_scaffold_response(response: str) -> List[Dict[str, Any]]:
    """Parse LLM response for scaffold files.
    
    Args:
        response: LLM response string
        
    Returns:
        List of scaffold file dicts with path, content, purpose
    """
    import json
    
    # Try to extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        logger.error("No JSON found in scaffold response")
        return []
    
    try:
        data = json.loads(json_match.group())
        return data.get("scaffold_files", [])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse scaffold JSON: {e}")
        return []


def run_scaffolding_phase(
    plan: Dict[str, Any],
    cycle_info: Dict[str, Any],
    file_snippets: str,
    llm_call_fn,  # Function to call LLM
    validate: bool = True,
    source_files: Optional[Dict[str, str]] = None,  # path -> content mapping
) -> ScaffoldResult:
    """Run the complete scaffolding phase.
    
    Args:
        plan: Refactoring plan from planning phase
        cycle_info: Cycle information
        file_snippets: File context for LLM
        llm_call_fn: Function to call LLM with prompt
        validate: Whether to validate scaffold syntax
        source_files: Optional dict of source file path -> content for method extraction
        
    Returns:
        ScaffoldResult with created files and validation status
    """
    result = ScaffoldResult()
    
    # Extract scaffold specs from plan
    scaffold_specs = extract_scaffold_from_plan(plan)
    
    if not scaffold_specs:
        logger.info("No scaffold files needed in plan")
        result.success = True
        return result
    
    logger.info(f"Scaffolding phase: generating {len(scaffold_specs)} files")
    
    # Extract interface requirements from plan and source files
    interface_reqs = None
    method_signatures: List[str] = []
    detected_namespace = ""
    detected_language = "unknown"
    
    if source_files:
        # Detect language from file extensions
        for path in source_files.keys():
            lang = detect_language(path)
            if lang != "unknown":
                detected_language = lang
                break
        
        # Detect namespace from source files
        for path, content in source_files.items():
            ns = detect_namespace(content, detect_language(path))
            if ns:
                detected_namespace = ns
                break
        
        # Extract interface requirements (finds target class and methods)
        interface_reqs = extract_interface_requirements(plan, source_files)
        if interface_reqs:
            method_signatures = interface_reqs.get("methods", [])
            if interface_reqs.get("namespace"):
                detected_namespace = interface_reqs["namespace"]
            logger.info(f"Extracted {len(method_signatures)} method signatures for scaffold")
            logger.info(f"Detected namespace: {detected_namespace}, language: {detected_language}")
        else:
            logger.warning("Could not extract interface requirements from source files")
    
    # Build prompt with enhanced context
    prompt = build_scaffold_prompt(
        scaffold_specs, 
        cycle_info, 
        file_snippets,
        method_signatures=method_signatures,
        detected_namespace=detected_namespace,
        language=detected_language
    )
    
    try:
        response = llm_call_fn(prompt)
        result.raw_llm_response = str(response)
        
        # Parse response
        scaffold_files = parse_scaffold_response(str(response))
        
        if not scaffold_files:
            result.success = False
            result.validation_errors.append("Failed to parse scaffold files from LLM response")
            return result
        
        # Validate each scaffold file
        all_valid = True
        for sf in scaffold_files:
            path = sf.get("path", "unknown")
            content = sf.get("content", "")
            purpose = sf.get("purpose", "unknown")
            
            file_info = {
                "path": path,
                "content": content,
                "purpose": purpose,
                "valid": True,
                "errors": []
            }
            
            if validate:
                is_valid, errors = validate_scaffold_syntax(content, path)
                file_info["valid"] = is_valid
                file_info["errors"] = errors
                
                if not is_valid:
                    all_valid = False
                    result.validation_errors.extend([f"{path}: {e}" for e in errors])
                    logger.warning(f"Scaffold validation failed for {path}: {errors}")
                else:
                    logger.info(f"Scaffold validated: {path}")
            
            result.files_created.append(file_info)
        
        result.syntax_valid = all_valid
        result.success = all_valid or not validate
        
    except Exception as e:
        logger.error(f"Scaffolding phase failed: {e}")
        result.success = False
        result.validation_errors.append(str(e))
    
    return result
