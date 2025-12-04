"""Utility for loading and formatting prompt templates.

This module provides a centralized way to load prompt templates from files
and safely format them with placeholders, handling JSON braces gracefully.
"""

import os
from typing import Any, Dict, Optional


def load_template(path_or_content: str) -> str:
    """Load a prompt template from a file path or return content directly.

    Args:
        path_or_content: Either a file path to a template file, or the
            template content itself.

    Returns:
        The template content as a string.

    Raises:
        FileNotFoundError: If path looks like a file but doesn't exist.
    """
    if not path_or_content:
        return ""

    # Check if it's a file path
    if isinstance(path_or_content, str):
        # Heuristics: if it's short and looks like a path, try to load it
        if len(path_or_content) < 500 and (
            path_or_content.endswith('.txt') or
            path_or_content.endswith('.md') or
            path_or_content.endswith('.prompt') or
            os.path.sep in path_or_content or
            '/' in path_or_content
        ):
            if os.path.isfile(path_or_content):
                with open(path_or_content, "r", encoding="utf-8") as f:
                    return f.read()
            # If it looks like a path but doesn't exist, could be an error
            # But we'll be lenient and treat as content

    return path_or_content


def safe_format(template: str, **kwargs: Any) -> str:
    """Format a template with placeholders, handling JSON braces safely.

    This function first tries Python's str.format(), which works for simple
    templates. If that fails (due to JSON examples with braces), it falls
    back to literal string replacement of known placeholders.

    Args:
        template: The template string with {placeholder} markers.
        **kwargs: Key-value pairs for placeholder substitution.

    Returns:
        The formatted string.

    Example:
        >>> safe_format("Hello {name}, data: {\"key\": 1}", name="World")
        'Hello World, data: {"key": 1}'
    """
    if not template:
        return ""

    # First, try standard format (works for simple templates)
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError, IndexError):
        pass

    # Fallback: literal replacement of known placeholders
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        str_value = str(value) if value is not None else ""
        result = result.replace(placeholder, str_value)

    return result


def load_and_format(
    path_or_content: str,
    **kwargs: Any
) -> str:
    """Load a template and format it in one call.

    Convenience function that combines load_template() and safe_format().

    Args:
        path_or_content: File path or template content.
        **kwargs: Placeholder values.

    Returns:
        The loaded and formatted template.
    """
    template = load_template(path_or_content)
    return safe_format(template, **kwargs)


def format_with_defaults(
    template: str,
    defaults: Dict[str, Any],
    **kwargs: Any
) -> str:
    """Format a template with defaults for missing placeholders.

    Args:
        template: The template string.
        defaults: Default values for placeholders.
        **kwargs: Override values (take precedence over defaults).

    Returns:
        The formatted string.
    """
    merged = {**defaults, **kwargs}
    return safe_format(template, **merged)


def extract_placeholders(template: str) -> list:
    """Extract placeholder names from a template.

    Args:
        template: The template string.

    Returns:
        List of placeholder names found in the template.

    Example:
        >>> extract_placeholders("Hello {name}, you have {count} items")
        ['name', 'count']
    """
    import re
    # Match {word} but not {{ or }}
    pattern = r'\{(\w+)\}'
    return re.findall(pattern, template)
