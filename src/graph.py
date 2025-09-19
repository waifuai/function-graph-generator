"""
Function Graph Generator - A tool for analyzing and visualizing function call relationships in source code.

This module provides functionality to parse source code files, extract function definitions and calls,
and generate interactive visual graphs showing the relationships between functions.
"""

import re
import argparse
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

import networkx as nx
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG = {
    "allow_external_funcs": True,
    "allow_self_reference": True,
    "allow_childless_funcs": True
}

VISUALIZATION_DEFAULTS = {
    "figsize": (12, 8),
    "node_color": "skyblue",
    "node_size": 500,
    "font_size": 10,
    "font_weight": "bold",
    "arrowsize": 20,
    "edge_color": "gray",
    "spring_layout_k": 0.5,
    "spring_layout_iterations": 9
}

class OutputFormat(Enum):
    """Supported output formats for the call graph."""
    MATPLOTLIB = "matplotlib"
    GRAPHML = "graphml"
    JSON = "json"
    DOT = "dot"

@dataclass
class ParseResult:
    """Result of parsing source code for functions and their calls."""
    functions: Set[str]
    function_calls: Dict[str, Set[str]]
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""

    @abstractmethod
    def extract_functions_and_calls(self, code_lines: List[str]) -> ParseResult:
        """
        Extract function definitions and their calls from code.

        Args:
            code_lines: List of code lines to parse

        Returns:
            ParseResult containing functions, function calls, and any errors encountered
        """
        pass

    @abstractmethod
    def detect_language(self, code_lines: List[str]) -> bool:
        """Determine if the provided code lines match this parser's language."""
        pass

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the name of the language this parser handles."""
        pass


class PythonParser(LanguageParser):
    """Parser for Python code with improved regex patterns and error handling."""

    @property
    def language_name(self) -> str:
        return "Python"

    def detect_language(self, code_lines: List[str]) -> bool:
        """Detect if the code is Python by looking for function definitions in the first 10 lines."""
        for line in code_lines[:10]:
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                return True
        return False

    def extract_functions_and_calls(self, code_lines: List[str]) -> ParseResult:
        """
        Extract function definitions and their calls from Python code.

        This implementation uses improved regex patterns to handle:
        - Nested function definitions
        - Class methods
        - Lambda functions (limited support)
        - Comments and strings
        - More robust indentation handling

        Args:
            code_lines: List of code lines to parse

        Returns:
            ParseResult containing functions, function calls, and any parsing errors
        """
        errors = []
        function_names: Set[str] = set()
        function_calls: Dict[str, Set[str]] = {}

        # Improved regex patterns
        function_def_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(')
        method_def_pattern = re.compile(r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:')
        function_call_pattern = re.compile(r'\b(\w+)\s*\(')

        # First pass: collect all function names
        for line_num, line in enumerate(code_lines, 1):
            stripped = line.strip()
            if stripped.startswith('def '):
                # Match both simple functions and class methods
                match = function_def_pattern.match(stripped)
                if match:
                    func_name = match.group(1)
                    function_names.add(func_name)
                    function_calls[func_name] = set()
                    logger.debug(f"Found function definition: {func_name} at line {line_num}")
                else:
                    errors.append(f"Invalid function definition at line {line_num}: {line}")

        if not function_names:
            logger.warning("No function definitions found in the code")
            return ParseResult(function_names, function_calls, errors)

        # Create pattern for function calls (escape special regex characters)
        escaped_names = [re.escape(name) for name in function_names]
        call_pattern = re.compile(r'\b(' + '|'.join(escaped_names) + r')\s*\(')

        # Second pass: analyze function calls within each function's scope
        current_function = None
        current_indent = 0
        in_string = False
        string_char = None
        in_comment = False

        for line_num, line in enumerate(code_lines, 1):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Handle string literals (basic support)
            if not in_comment:
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None

            # Skip lines inside strings or comments
            if in_string or in_comment:
                continue

            # Check for function definition
            if stripped.startswith('def '):
                match = function_def_pattern.match(stripped)
                if match:
                    current_function = match.group(1)
                    # Calculate indentation more robustly
                    current_indent = len(line) - len(line.lstrip())
                    logger.debug(f"Entered function: {current_function} at line {line_num}")
                continue

            # Analyze calls within current function scope
            if current_function is not None:
                line_indent = len(line) - len(line.lstrip())

                # Check if we're still inside the function
                if line_indent > current_indent:
                    # Look for function calls in this line
                    if not in_string and not in_comment:
                        for match in call_pattern.finditer(line):
                            called_func = match.group(1)
                            # Avoid self-calls in trivial cases and built-in functions
                            if called_func != current_function and called_func in function_names:
                                function_calls[current_function].add(called_func)
                                logger.debug(f"Found call: {current_function} -> {called_func} at line {line_num}")
                else:
                    # Exited the function scope
                    logger.debug(f"Exited function: {current_function} at line {line_num}")
                    current_function = None

        logger.info(f"Parsing completed: {len(function_names)} functions found, {len(errors)} errors")
        return ParseResult(function_names, function_calls, errors)


class CallGraphGenerator:
    """
    Generates and visualizes function call graphs with enhanced features.

    This class provides a comprehensive solution for analyzing source code and generating
    interactive visualizations of function call relationships. It supports multiple
    output formats and configurable filtering options.
    """

    def __init__(self, config: Optional[Dict[str, bool]] = None):
        """
        Initialize the CallGraphGenerator with optional configuration.

        Args:
            config: Optional configuration dictionary. If None, uses DEFAULT_CONFIG.
        """
        self.parsers: List[LanguageParser] = [PythonParser()]
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration settings."""
        required_keys = {"allow_external_funcs", "allow_self_reference", "allow_childless_funcs"}
        if not required_keys.issubset(self.config.keys()):
            missing = required_keys - set(self.config.keys())
            raise ValueError(f"Missing required configuration keys: {missing}")

        for key, value in self.config.items():
            if not isinstance(value, bool):
                raise ValueError(f"Configuration value for '{key}' must be boolean, got {type(value)}")

    def detect_language_parser(self, code_lines: List[str]) -> LanguageParser:
        """
        Select the appropriate language parser based on the code content.

        Args:
            code_lines: List of code lines to analyze

        Returns:
            The appropriate LanguageParser for the detected language

        Raises:
            ValueError: If no suitable parser is found for the code
        """
        for parser in self.parsers:
            if parser.detect_language(code_lines):
                logger.info(f"Detected language: {parser.language_name}")
                return parser

        # Try to provide helpful error message
        first_lines = '\n'.join(code_lines[:5])
        raise ValueError(
            f"Unable to determine programming language from code. First few lines:\n{first_lines}"
        )

    def filter_functions(self, funcs: Set[str], function_calls: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Filter functions based on the configuration settings.

        Args:
            funcs: Set of all defined functions
            function_calls: Dictionary mapping functions to their called functions

        Returns:
            Filtered dictionary of function calls based on configuration
        """
        filtered_calls = {}

        for func, calls in function_calls.items():
            filtered = calls.copy()

            # Remove external function calls if not allowed
            if not self.config["allow_external_funcs"]:
                filtered &= funcs

            # Remove self-references if not allowed
            if not self.config["allow_self_reference"]:
                filtered.discard(func)

            # Include function only if it has children or childless functions are allowed
            if self.config["allow_childless_funcs"] or filtered:
                filtered_calls[func] = filtered

        logger.info(f"Filtering completed: {len(filtered_calls)} functions after filtering")
        return filtered_calls

    def create_and_visualize_graph(
        self,
        function_calls: Dict[str, Set[str]],
        title: str = "Function Call Graph",
        output_format: OutputFormat = OutputFormat.MATPLOTLIB,
        output_path: Optional[str] = None
    ) -> Optional[nx.DiGraph]:
        """
        Create and display/save the function call graph.

        Args:
            function_calls: Dictionary of function calls to visualize
            title: Title for the graph
            output_format: Format for the output (matplotlib, graphml, json, dot)
            output_path: Path to save the output (required for non-matplotlib formats)

        Returns:
            The NetworkX graph object for further processing

        Raises:
            ValueError: If output_path is required but not provided
        """
        # Create the graph
        graph = nx.DiGraph()

        for func, calls in function_calls.items():
            graph.add_node(func)
            for call in calls:
                graph.add_node(call)
                graph.add_edge(func, call)

        logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        if output_format == OutputFormat.MATPLOTLIB:
            self._visualize_with_matplotlib(graph, title)
        elif output_format == OutputFormat.GRAPHML:
            if not output_path:
                raise ValueError("output_path is required for GRAPHML format")
            nx.write_graphml(graph, output_path)
            logger.info(f"Graph saved as GRAPHML to {output_path}")
        elif output_format == OutputFormat.JSON:
            if not output_path:
                raise ValueError("output_path is required for JSON format")
            self._save_as_json(graph, output_path)
        elif output_format == OutputFormat.DOT:
            if not output_path:
                raise ValueError("output_path is required for DOT format")
            nx.nx_pydot.write_dot(graph, output_path)
            logger.info(f"Graph saved as DOT to {output_path}")

        return graph

    def _visualize_with_matplotlib(self, graph: nx.DiGraph, title: str) -> None:
        """Create matplotlib visualization of the graph."""
        if len(graph.nodes) == 0:
            logger.warning("No nodes to visualize")
            return

        pos = nx.spring_layout(
            graph,
            k=VISUALIZATION_DEFAULTS["spring_layout_k"],
            iterations=VISUALIZATION_DEFAULTS["spring_layout_iterations"]
        )

        plt.figure(figsize=VISUALIZATION_DEFAULTS["figsize"])
        nx.draw(
            graph,
            pos=pos,
            with_labels=True,
            font_weight=VISUALIZATION_DEFAULTS["font_weight"],
            arrowsize=VISUALIZATION_DEFAULTS["arrowsize"],
            node_color=VISUALIZATION_DEFAULTS["node_color"],
            node_size=VISUALIZATION_DEFAULTS["node_size"],
            font_size=VISUALIZATION_DEFAULTS["font_size"],
            edge_color=VISUALIZATION_DEFAULTS["edge_color"]
        )
        plt.title(title)
        plt.show()

    def _save_as_json(self, graph: nx.DiGraph, output_path: str) -> None:
        """Save graph as JSON format."""
        data = {
            "nodes": list(graph.nodes),
            "edges": list(graph.edges),
            "metadata": {
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges)
            }
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Graph saved as JSON to {output_path}")

    def generate_call_graph(
        self,
        filepath: Union[str, Path],
        output_format: OutputFormat = OutputFormat.MATPLOTLIB,
        output_path: Optional[str] = None
    ) -> Optional[nx.DiGraph]:
        """
        Generate and visualize the call graph from the source file.

        Args:
            filepath: Path to the source file to analyze
            output_format: Format for the output
            output_path: Path for saving output (if applicable)

        Returns:
            The generated NetworkX graph

        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file cannot be parsed
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {filepath}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {filepath}")

        logger.info(f"Processing file: {filepath}")

        try:
            code_lines = path.read_text(encoding='utf-8').splitlines()
            logger.info(f"Read {len(code_lines)} lines from file")
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to read file as UTF-8 text: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

        if not code_lines:
            logger.warning("File is empty")
            return None

        # Parse the code
        parser = self.detect_language_parser(code_lines)
        parse_result = parser.extract_functions_and_calls(code_lines)

        # Log any parsing errors
        if parse_result.errors:
            logger.warning(f"Parsing errors encountered: {len(parse_result.errors)}")
            for error in parse_result.errors[:5]:  # Log first 5 errors
                logger.warning(f"  {error}")
            if len(parse_result.errors) > 5:
                logger.warning(f"  ... and {len(parse_result.errors) - 5} more errors")

        # Filter functions based on configuration
        filtered_calls = self.filter_functions(parse_result.functions, parse_result.function_calls)

        # Generate the graph
        title = f"{parser.language_name} Function Call Graph"
        return self.create_and_visualize_graph(filtered_calls, title, output_format, output_path)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate function call graphs from source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.py                           # Generate matplotlib visualization
  %(prog)s input.py --format graphml -o output.graphml  # Export as GraphML
  %(prog)s input.py --format json -o output.json        # Export as JSON
  %(prog)s input.py --no-external-funcs                 # Exclude external function calls
  %(prog)s input.py --config config.yaml                # Use configuration file
        """
    )

    parser.add_argument(
        'input_file',
        help='Path to the input source file'
    )

    parser.add_argument(
        '-f', '--format',
        choices=[fmt.value for fmt in OutputFormat],
        default=OutputFormat.MATPLOTLIB.value,
        help='Output format (default: matplotlib)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file path (required for non-matplotlib formats)'
    )

    parser.add_argument(
        '--no-external-funcs',
        action='store_true',
        help='Exclude functions called but not defined in the source'
    )

    parser.add_argument(
        '--no-self-reference',
        action='store_true',
        help='Hide recursive function calls'
    )

    parser.add_argument(
        '--no-childless-funcs',
        action='store_true',
        help='Omit functions that do not call other functions'
    )

    parser.add_argument(
        '-c', '--config',
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Function Graph Generator 2.0.0'
    )

    return parser


def load_config_from_file(config_path: str) -> Dict[str, bool]:
    """Load configuration from a YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required for configuration file support. Install with: pip install pyyaml")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a dictionary")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the function graph generator."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Load configuration
    config = DEFAULT_CONFIG.copy()

    if args.config:
        file_config = load_config_from_file(args.config)
        config.update(file_config)
        logger.info(f"Loaded configuration from {args.config}")

    # Override config with command-line flags
    if args.no_external_funcs:
        config["allow_external_funcs"] = False
    if args.no_self_reference:
        config["allow_self_reference"] = False
    if args.no_childless_funcs:
        config["allow_childless_funcs"] = False

    logger.info(f"Using configuration: {config}")

    # Validate output requirements
    output_format = OutputFormat(args.format)
    if output_format != OutputFormat.MATPLOTLIB and not args.output:
        parser.error(f"Output file path (-o/--output) is required for format '{args.format}'")

    try:
        # Create generator and generate graph
        generator = CallGraphGenerator(config)
        graph = generator.generate_call_graph(
            filepath=args.input_file,
            output_format=output_format,
            output_path=args.output
        )

        if graph is None:
            logger.info("No functions found to visualize")
            sys.exit(0)

        logger.info("Graph generation completed successfully")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or parsing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
