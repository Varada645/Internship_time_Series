import os

EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', '.idea', '.mypy_cache', '.vscode'}
EXCLUDE_FILES = {'.DS_Store'}

def list_tree(startpath="."):
    output = []
    for root, dirs, files in os.walk(startpath):
        # Remove unwanted directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        files = [f for f in files if f not in EXCLUDE_FILES and not f.startswith('.')]

        level = root.replace(startpath, '').count(os.sep)
        indent = '    ' * level
        output.append(f"{indent}{os.path.basename(root)}/")
        subindent = '    ' * (level + 1)
        for file in files:
            output.append(f"{subindent}{file}")

    return '\n'.join(output)

if __name__ == "__main__":
    tree = list_tree("c:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease")
    print("```\n" + tree + "\n```")  # Markdown code block
