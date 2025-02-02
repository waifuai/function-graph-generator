# Function Graph Generator

Automatically generate function call graphs for Python code.

`graph.py` is a Python script that analyzes Python source code and generates visual function call graphs showing dependencies between functions.

## Features

- Supports Python
- Configurable graph generation
- Interactive visualization with matplotlib
- Unit tested with sample code cases

## Usage

### 1. Prepare Source Code
Create a file containing your Python source code. For example, you can combine multiple Python files into a single input file:

```bash
find ./src/ -name '*.py' -exec cat {} \; > input.txt
```
or
```bash
cat ./src/*.py > input.txt
```

### 2. Generate Call Graph
Run the script:
```bash
python3 graph.py
```

A window will display the interactive graph. You can zoom, pan, and inspect nodes.

## Configuration

Modify the generator's configuration before generating the graph:
```python
generator = CallGraphGenerator()
generator.config["allow_external_funcs"] = False  # Exclude functions called but not defined in the source
generator.config["allow_self_reference"] = False  # Hide recursive calls
generator.config["allow_childless_funcs"] = False  # Omit functions that do not call other functions
generator.generate_call_graph("input.txt")
```

| Setting                | Default | Description                                                                    |
|------------------------|---------|--------------------------------------------------------------------------------|
| `allow_external_funcs` | True    | Show functions called but not defined in the source                            |
| `allow_self_reference` | True    | Display edges for recursive calls                                              |
| `allow_childless_funcs`| True    | Include functions that don't call other functions                              |

## Testing

Unit tests verify function extraction and graph generation. Run tests with:
```bash
python3 -m unittest test.py
```

## Dependencies

- Python 3.6+
- networkx
- matplotlib

Install requirements:
```bash
pip install --user networkx matplotlib
```

## How It Works

1. **Language Detection**: Checks the first 10 lines for the Python-specific pattern:
   - Python: `def` keywords at the beginning of a line (allowing for leading spaces).

2. **Parsing**: The Python parser extracts function definitions and calls.

3. **Graph Construction**: Uses networkx to build a directed graph from call relationships.

4. **Visualization**: Applies the spring layout algorithm and matplotlib for interactive display.

## Notes

- For large codebases, consider splitting into modules. The script processes all input as a single file.
- The spring layout's appearance can be adjusted in `create_and_visualize_graph()`.
- Function detection uses basic pattern matching. Complex syntax may require parser improvements.

## Example Output

Interact with the graph to:
- Drag nodes to rearrange the layout
- Zoom with the scroll wheel
- Click nodes to highlight connections