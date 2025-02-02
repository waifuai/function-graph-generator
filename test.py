import unittest
import matplotlib
matplotlib.use('Agg')  # Set backend before other imports

import io
from pathlib import Path
from unittest.mock import patch

import graph
import networkx as nx


class TestGraphGenerator(unittest.TestCase):

    def test_extract_functions_and_calls_python(self):
        code = [
            "def foo(x):",
            "    bar(x)",
            "    baz(x)",
            "def bar(x):",
            "    baz(x)",
            "def baz(x):",
            "    return x"
        ]
        parser = graph.PythonParser()
        funcs, function_calls = parser.extract_functions_and_calls(code)
        self.assertEqual(funcs, {"foo", "bar", "baz"})
        self.assertEqual(function_calls["foo"], {"bar", "baz"})
        self.assertEqual(function_calls["bar"], {"baz"})
        self.assertEqual(function_calls["baz"], set())

    def test_filter_functions(self):
        funcs = {"a", "b", "c", "d"}
        dfuncs = {"a": {"b", "c", "a", "e"}, "b": {"c"}, "c": set()}

        generator = graph.CallGraphGenerator()
        generator.config["allow_external_funcs"] = False
        generator.config["allow_self_reference"] = False
        generator.config["allow_childless_funcs"] = False
        filtered = generator.filter_functions(funcs, dfuncs)
        self.assertEqual(filtered, {'a': {'b', 'c'}, 'b': {'c'}})

        generator.config = {
            "allow_external_funcs": True,
            "allow_self_reference": True,
            "allow_childless_funcs": True
        }
        filtered = generator.filter_functions(funcs, dfuncs)
        expected = {"a": {"b", "c", "a", "e"}, "b": {"c"}, "c": set()}
        self.assertEqual(filtered, expected)

    @patch('matplotlib.pyplot.show')
    def test_create_and_visualize_graph(self, mock_show):
        generator = graph.CallGraphGenerator()
        function_calls = {"a": {"b", "c"}, "b": {"c"}}
        generator.create_and_visualize_graph(function_calls)

    def test_main_python(self):
        with patch('matplotlib.pyplot.show'):
            fake_file = io.StringIO("def foo(x):\n    bar(x)\ndef bar(x):\n    return x")
            with patch('pathlib.Path.read_text', return_value=fake_file.getvalue()):
                with patch('pathlib.Path.exists', return_value=True):
                    graph.main()


if __name__ == '__main__':
    unittest.main()
