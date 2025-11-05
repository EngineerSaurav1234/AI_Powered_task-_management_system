import os
import ast
import sys

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return None
    except SyntaxError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file: {e}"

def main():
    errors = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                error = check_syntax(file_path)
                if error:
                    errors.append(f"{file_path}: {error}")
    if errors:
        print("Syntax errors found:")
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("No syntax errors found in any Python files.")

if __name__ == "__main__":
    main()
