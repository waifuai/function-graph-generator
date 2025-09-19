"""
Function Graph Generator - A Python package for analyzing and visualizing function call relationships.

This package provides tools to parse source code, extract function definitions and their calls,
and generate interactive visual graphs showing the relationships between functions.
"""

__version__ = "2.0.0"
__author__ = "Function Graph Generator Team"
__description__ = "Generate function call graphs from source code"

from .graph import (
    CallGraphGenerator,
    LanguageParser,
    PythonParser,
    OutputFormat,
    ParseResult,
    DEFAULT_CONFIG,
    VISUALIZATION_DEFAULTS
)

__all__ = [
    "CallGraphGenerator",
    "LanguageParser",
    "PythonParser",
    "OutputFormat",
    "ParseResult",
    "DEFAULT_CONFIG",
    "VISUALIZATION_DEFAULTS"
]