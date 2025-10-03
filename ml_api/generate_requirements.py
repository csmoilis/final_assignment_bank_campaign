import ast
import importlib.metadata
import sys
from pathlib import Path

# Mapping from import names (used in code) to PyPI package names
PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "lxml": "lxml",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",  # e.g. "from mpl_toolkits.mplot3d import Axes3D"
    "torch": "torch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pd": "pandas",  # people often write "import pandas as pd"
    "np": "numpy",   # same for numpy
}

def extract_imports(py_file: str):
    """Extract top-level imported packages from a Python file."""
    with open(py_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=py_file)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return sorted(imports)

def get_installed_version(package: str):
    """Get installed version of a package if available."""
    # Look up in alias list, otherwise keep the name
    package_name = PACKAGE_ALIASES.get(package, package)
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

def check_unresolved(imports):
    """Warn about imports that could not be matched to an installed package."""
    unresolved = []
    for package in imports:
        package_name = PACKAGE_ALIASES.get(package, package)
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            unresolved.append(package)
    if unresolved:
        print("\n⚠️ The following imports could not be matched to installed packages:")
        for pkg in unresolved:
            alias = PACKAGE_ALIASES.get(pkg)
            if alias:
                print(f"  - {pkg} (alias '{alias}' was not installed)")
            else:
                print(f"  - {pkg} (no alias defined)")
        print("Tip: Add a mapping in PACKAGE_ALIASES or install the package.\n")

def write_requirements(imports, output_file="requirements.txt"):
    """Write dependencies to requirements.txt style."""
    with open(output_file, "w", encoding="utf-8") as f:
        for package in imports:
            package_name = PACKAGE_ALIASES.get(package, package)
            version = get_installed_version(package)
            if version:
                f.write(f"{package_name}=={version}\n")
            else:
                f.write(f"{package_name}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_requirements.py your_script.py")
        sys.exit(1)

    py_file = sys.argv[1]
    if not Path(py_file).is_file():
        print(f"Error: {py_file} not found.")
        sys.exit(1)

    imports = extract_imports(py_file)
    write_requirements(imports)
    check_unresolved(imports)
    print("requirements.txt generated successfully.")

if __name__ == "__main__":
    main()
