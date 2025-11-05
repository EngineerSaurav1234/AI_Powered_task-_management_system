import json
import sys
import re

def fix_json(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        nb = json.loads(content)
        print(f"No JSON errors in {notebook_path}.")
        return nb
    except json.JSONDecodeError as e:
        print(f"JSON error in {notebook_path}: {e}")
        lines = content.split('\n')
        line_num = e.lineno - 1  # 0-based
        col_num = e.colno - 1
        if line_num < len(lines):
            line = lines[line_num]
            print(f"Line {e.lineno}: {line}")
            print(" " * (col_num) + "^")
        # Attempt to fix: replace "name": "nbconvert_exporter": "python", with "name": "nbconvert_exporter", "python",
        # The error is Expecting ',' delimiter, and the line shows "name": "nbconvert_exporter": "python",
        # It seems like it's "name": "nbconvert_exporter": "python", but should be "name": "nbconvert_exporter", "python": something
        # Looking at the error, it's expecting , after "python",
        # Probably it's "name": "nbconvert_exporter": "python", but missing comma before "python"
        # Actually, it looks like "name": "nbconvert_exporter": "python", which is invalid JSON.
        # Perhaps it's meant to be "name": "nbconvert_exporter", "python": something
        # But to fix, we can replace the : with , before "python"
        # Let's try to fix by replacing the pattern
        fixed_content = re.sub(r'"name": "nbconvert_exporter": "python",', '"name": "nbconvert_exporter", "python": "python",', content)
        try:
            nb = json.loads(fixed_content)
            print(f"Fixed JSON in {notebook_path}.")
            # Write back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            return nb
        except json.JSONDecodeError as e2:
            print(f"Still error after fix: {e2}")
            return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json.py <notebook_path>")
        sys.exit(1)
    notebook_path = sys.argv[1]
    fix_json(notebook_path)
