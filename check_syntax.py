import json
import ast
import sys

def check_syntax(code):
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return str(e)

def check_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    errors = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            error = check_syntax(code)
            if error:
                errors.append(f"Cell {i+1}: {error}")
    return errors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_syntax.py <notebook_path>")
        sys.exit(1)
    notebook_path = sys.argv[1]
    errors = check_notebook(notebook_path)
    if errors:
        print(f"Syntax errors in {notebook_path}:")
        for error in errors:
            print(error)
    else:
        print(f"No syntax errors found in {notebook_path}.")
