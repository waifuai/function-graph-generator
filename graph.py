import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""

    @abstractmethod
    def extract_functions_and_calls(self, code_lines: List[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """Extract function definitions and their calls from code.

        Returns:
            A tuple with:
            - A set of all function names.
            - A dictionary mapping each function name to a set of called functions.
        """
        pass

    @abstractmethod
    def detect_language(self, code_lines: List[str]) -> bool:
        """Determine if the provided code lines match this parser's language."""
        pass


class PythonParser(LanguageParser):
    """Parser for Python code."""

    def detect_language(self, code_lines: List[str]) -> bool:
        for line in code_lines[:10]:
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                return True
        return False

    def extract_functions_and_calls(self, code_lines: List[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
        # First pass: collect all function names.
        function_names: Set[str] = set()
        for line in code_lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                match = re.match(r'^def\s+(\w+)\s*\(', stripped)
                if match:
                    function_names.add(match.group(1))

        # Initialize mapping for each function.
        function_calls: Dict[str, Set[str]] = {func: set() for func in function_names}

        if not function_names:
            return function_names, function_calls

        # Create regex pattern to match function calls.
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, function_names)) + r')\s*\(')

        current_function = None
        current_indent = 0

        for line in code_lines:
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if stripped.startswith('def '):
                # Enter a new function definition.
                match = re.match(r'^def\s+(\w+)\s*\(', stripped)
                if match:
                    current_function = match.group(1)
                    current_indent = indent
                continue

            if current_function is not None:
                # Check if still within the current function block.
                if indent > current_indent:
                    # Look for function calls within the function block.
                    for call in pattern.findall(line):
                        function_calls[current_function].add(call)
                else:
                    # Exited the current function block.
                    current_function = None

        return function_names, function_calls


class CallGraphGenerator:
    """Generates and visualizes function call graphs."""

    def __init__(self):
        self.parsers: List[LanguageParser] = [
            PythonParser()
        ]
        # Default configuration matching the documentation.
        self.config: Dict[str, bool] = {
            "allow_external_funcs": True,
            "allow_self_reference": True,
            "allow_childless_funcs": True
        }

    def detect_language_parser(self, code_lines: List[str]) -> LanguageParser:
        """Select the appropriate language parser based on the code."""
        for parser in self.parsers:
            if parser.detect_language(code_lines):
                return parser
        raise ValueError("Unable to determine programming language from code.")

    def filter_functions(self, funcs: Set[str], function_calls: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Filter functions based on the configuration settings."""
        filtered_calls = {}
        for func, calls in function_calls.items():
            filtered = calls.copy()
            if not self.config["allow_external_funcs"]:
                filtered &= funcs
            if not self.config["allow_self_reference"]:
                filtered.discard(func)
            # Only include functions with children if not allowed to include childless functions.
            if self.config["allow_childless_funcs"] or filtered:
                filtered_calls[func] = filtered
        return filtered_calls

    def create_and_visualize_graph(self, function_calls: Dict[str, Set[str]], title: str = "Function Call Graph") -> None:
        """Create and display the function call graph using networkx and matplotlib."""
        graph = nx.DiGraph()
        for func, calls in function_calls.items():
            graph.add_node(func)
            for call in calls:
                graph.add_node(call)
                graph.add_edge(func, call)

        pos = nx.spring_layout(graph, k=0.5, iterations=9)
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos=pos, with_labels=True, font_weight='bold',
                arrowsize=20, node_color="skyblue", node_size=500,
                font_size=10, edge_color="gray")
        plt.title(title)
        plt.show()

    def generate_call_graph(self, filepath: str) -> None:
        """Generate and visualize the call graph from the source file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        code_lines = path.read_text().splitlines()
        parser = self.detect_language_parser(code_lines)
        funcs, function_calls = parser.extract_functions_and_calls(code_lines)
        filtered_calls = self.filter_functions(funcs, function_calls)
        language_name = parser.__class__.__name__.replace('Parser', '')
        self.create_and_visualize_graph(filtered_calls, f"{language_name} Function Call Graph")


def main() -> None:
    generator = CallGraphGenerator()
    generator.generate_call_graph("input.txt")


if __name__ == "__main__":
    main()
