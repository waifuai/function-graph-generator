"""
Comprehensive test suite for the Function Graph Generator.

This module contains unit tests for all major components of the function graph generator,
including parsing, filtering, visualization, and command-line interface functionality.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import sys
import io
import json

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import graph
from graph import (
    CallGraphGenerator,
    PythonParser,
    ParseResult,
    OutputFormat,
    DEFAULT_CONFIG
)
import networkx as nx


class TestPythonParser(unittest.TestCase):
    """Test cases for the PythonParser class."""

    def setUp(self):
        self.parser = PythonParser()

    def test_detect_language_python(self):
        """Test language detection for Python code."""
        python_code = [
            "def hello():",
            "    print('Hello, World!')",
            "",
            "def main():",
            "    hello()"
        ]
        self.assertTrue(self.parser.detect_language(python_code))

    def test_detect_language_non_python(self):
        """Test language detection for non-Python code."""
        non_python_code = [
            "function hello() {",
            "    console.log('Hello, World!');",
            "}",
            "",
            "hello();"
        ]
        self.assertFalse(self.parser.detect_language(non_python_code))

    def test_extract_functions_and_calls_simple(self):
        """Test basic function extraction and call detection."""
        code = [
            "def foo(x):",
            "    bar(x)",
            "    baz(x)",
            "def bar(x):",
            "    baz(x)",
            "def baz(x):",
            "    return x"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertIsInstance(result, ParseResult)
        self.assertEqual(result.functions, {"foo", "bar", "baz"})
        self.assertEqual(result.function_calls["foo"], {"bar", "baz"})
        self.assertEqual(result.function_calls["bar"], {"baz"})
        self.assertEqual(result.function_calls["baz"], set())
        self.assertEqual(len(result.errors), 0)

    def test_extract_functions_with_nested_calls(self):
        """Test extraction with nested function calls."""
        code = [
            "def outer():",
            "    inner()",
            "    inner2()",
            "def inner():",
            "    inner2()",
            "def inner2():",
            "    pass"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertEqual(result.functions, {"outer", "inner", "inner2"})
        self.assertEqual(result.function_calls["outer"], {"inner", "inner2"})
        self.assertEqual(result.function_calls["inner"], {"inner2"})
        self.assertEqual(result.function_calls["inner2"], set())

    def test_extract_functions_with_self_reference(self):
        """Test extraction with recursive function calls."""
        code = [
            "def factorial(n):",
            "    if n <= 1:",
            "        return 1",
            "    return n * factorial(n - 1)"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertEqual(result.functions, {"factorial"})
        self.assertEqual(result.function_calls["factorial"], {"factorial"})

    def test_extract_functions_with_external_calls(self):
        """Test extraction with calls to undefined functions."""
        code = [
            "def local_func():",
            "    external_func()",
            "    another_external()"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertEqual(result.functions, {"local_func"})
        # External functions should not be in function_calls since they're not defined
        self.assertEqual(result.function_calls["local_func"], set())

    def test_extract_functions_empty_code(self):
        """Test extraction with empty or no functions."""
        code = [
            "print('Hello, World!')",
            "x = 1 + 2",
            "# This is a comment"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertEqual(result.functions, set())
        self.assertEqual(result.function_calls, {})
        self.assertEqual(len(result.errors), 0)

    def test_extract_functions_with_comments_and_strings(self):
        """Test extraction with comments and string literals."""
        code = [
            "def func():",
            "    # This is a comment with def and function()",
            '    print("This string contains def and function()")',
            "    real_call()",
            "    # function() is just a comment"
        ]

        result = self.parser.extract_functions_and_calls(code)

        self.assertEqual(result.functions, {"func"})
        # real_call is not defined, so it shouldn't be in calls
        self.assertEqual(result.function_calls["func"], set())

    def test_parse_result_creation(self):
        """Test ParseResult dataclass functionality."""
        functions = {"func1", "func2"}
        function_calls = {"func1": {"func2"}}
        errors = ["Error 1", "Error 2"]

        result = ParseResult(functions, function_calls, errors)

        self.assertEqual(result.functions, functions)
        self.assertEqual(result.function_calls, function_calls)
        self.assertEqual(result.errors, errors)

        # Test default errors list
        result2 = ParseResult(functions, function_calls)
        self.assertEqual(result2.errors, [])


class TestCallGraphGenerator(unittest.TestCase):
    """Test cases for the CallGraphGenerator class."""

    def setUp(self):
        self.generator = CallGraphGenerator()

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        generator = CallGraphGenerator()
        self.assertEqual(generator.config, DEFAULT_CONFIG)

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "allow_external_funcs": False,
            "allow_self_reference": False,
            "allow_childless_funcs": False
        }
        generator = CallGraphGenerator(custom_config)
        self.assertEqual(generator.config, custom_config)

    def test_invalid_config_missing_keys(self):
        """Test initialization with invalid configuration (missing keys)."""
        with self.assertRaises(ValueError):
            CallGraphGenerator({})

    def test_invalid_config_wrong_types(self):
        """Test initialization with invalid configuration (wrong types)."""
        with self.assertRaises(ValueError):
            CallGraphGenerator({
                "allow_external_funcs": "not_boolean",
                "allow_self_reference": True,
                "allow_childless_funcs": True
            })

    def test_filter_functions_basic(self):
        """Test basic function filtering."""
        funcs = {"a", "b", "c", "d"}
        function_calls = {"a": {"b", "c", "a", "e"}, "b": {"c"}, "c": set()}

        # Test with all filters disabled
        self.generator.config = {
            "allow_external_funcs": False,
            "allow_self_reference": False,
            "allow_childless_funcs": False
        }
        filtered = self.generator.filter_functions(funcs, function_calls)
        self.assertEqual(filtered, {'a': {'b', 'c'}, 'b': {'c'}})

        # Test with all filters enabled
        self.generator.config = {
            "allow_external_funcs": True,
            "allow_self_reference": True,
            "allow_childless_funcs": True
        }
        filtered = self.generator.filter_functions(funcs, function_calls)
        expected = {"a": {"b", "c", "a", "e"}, "b": {"c"}, "c": set()}
        self.assertEqual(filtered, expected)

    def test_filter_functions_external_funcs_only(self):
        """Test filtering with external functions only."""
        funcs = {"local"}
        function_calls = {"local": {"external", "local"}}

        self.generator.config = {
            "allow_external_funcs": False,
            "allow_self_reference": True,
            "allow_childless_funcs": True
        }
        filtered = self.generator.filter_functions(funcs, function_calls)
        self.assertEqual(filtered, {"local": {"local"}})

    def test_detect_language_parser_python(self):
        """Test language parser detection for Python."""
        code_lines = [
            "def hello():",
            "    print('Hello')",
            "def main():",
            "    hello()"
        ]

        parser = self.generator.detect_language_parser(code_lines)
        self.assertIsInstance(parser, PythonParser)
        self.assertEqual(parser.language_name, "Python")

    def test_detect_language_parser_unknown(self):
        """Test language parser detection for unknown language."""
        code_lines = [
            "function hello() {",
            "    console.log('Hello');",
            "}"
        ]

        with self.assertRaises(ValueError):
            self.generator.detect_language_parser(code_lines)

    @patch('matplotlib.pyplot.show')
    def test_create_and_visualize_graph_matplotlib(self, mock_show):
        """Test graph creation and visualization with matplotlib."""
        function_calls = {"a": {"b", "c"}, "b": {"c"}}

        graph = self.generator.create_and_visualize_graph(function_calls)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(set(graph.nodes), {"a", "b", "c"})
        self.assertEqual(set(graph.edges), {("a", "b"), ("a", "c"), ("b", "c")})
        mock_show.assert_called_once()

    def test_create_and_visualize_graph_graphml(self):
        """Test graph creation and export to GraphML format."""
        function_calls = {"a": {"b", "c"}, "b": {"c"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            temp_path = f.name

        try:
            graph = self.generator.create_and_visualize_graph(
                function_calls,
                output_format=OutputFormat.GRAPHML,
                output_path=temp_path
            )

            self.assertIsInstance(graph, nx.DiGraph)
            self.assertTrue(os.path.exists(temp_path))

            # Verify it's a valid GraphML file
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertIn('<graphml', content)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_create_and_visualize_graph_json(self):
        """Test graph creation and export to JSON format."""
        function_calls = {"a": {"b", "c"}, "b": {"c"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            graph = self.generator.create_and_visualize_graph(
                function_calls,
                output_format=OutputFormat.JSON,
                output_path=temp_path
            )

            self.assertIsInstance(graph, nx.DiGraph)
            self.assertTrue(os.path.exists(temp_path))

            # Verify it's valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)
                self.assertIn('nodes', data)
                self.assertIn('edges', data)
                self.assertIn('metadata', data)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_generate_call_graph_success(self):
        """Test successful call graph generation from file."""
        code_content = "def foo():\n    bar()\ndef bar():\n    pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_content)
            temp_path = f.name

        try:
            with patch('matplotlib.pyplot.show'):
                graph = self.generator.generate_call_graph(temp_path)

            self.assertIsInstance(graph, nx.DiGraph)
            self.assertEqual(set(graph.nodes), {"foo", "bar"})
            self.assertEqual(set(graph.edges), {("foo", "bar")})

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_generate_call_graph_file_not_found(self):
        """Test call graph generation with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.generator.generate_call_graph("non_existent_file.py")

    def test_generate_call_graph_empty_file(self):
        """Test call graph generation with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name

        try:
            graph = self.generator.generate_call_graph(temp_path)
            self.assertIsNone(graph)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCommandLineInterface(unittest.TestCase):
    """Test cases for the command-line interface."""

    def test_create_argument_parser(self):
        """Test argument parser creation."""
        from graph import create_argument_parser
        parser = create_argument_parser()

        # Test that all expected arguments are present
        args = parser.parse_args(['input.py'])
        self.assertEqual(args.input_file, 'input.py')
        self.assertEqual(args.format, 'matplotlib')
        self.assertIsNone(args.output)

    def test_main_with_valid_file(self):
        """Test main function with valid input file."""
        code_content = "def foo():\n    bar()\ndef bar():\n    pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_content)
            temp_path = f.name

        try:
            with patch('sys.argv', ['graph.py', temp_path]):
                with patch('matplotlib.pyplot.show'):
                    with patch('graph.logger') as mock_logger:
                        # Mock the logger to avoid output during tests
                        mock_logger.info = MagicMock()
                        mock_logger.error = MagicMock()

                        # This should not raise an exception
                        graph.main()

                        mock_logger.info.assert_called()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_main_with_non_existent_file(self):
        """Test main function with non-existent input file."""
        with patch('sys.argv', ['graph.py', 'non_existent_file.py']):
            with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
                with self.assertRaises(SystemExit) as cm:
                    graph.main()

                self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
