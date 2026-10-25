"""Enforces the layering rules (loose coupling) by parsing the imports of every source module."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "turnaround"

ALLOWED = {
    "domain": set(),
    "config": {"domain"},
    "analytics": {"domain"},
    "storage": {"domain"},
    "ingestion": {"domain", "config"},
    "detection": {"domain", "config"},
    "spatial": {"domain", "config"},
    "temporal": {"domain", "config", "spatial"},
    "runtime": set(),
    "utils": set(),
    "visualization": {"domain", "spatial", "temporal"},
    "pipeline": {"domain", "config", "analytics", "storage", "ingestion", "detection", "spatial", "temporal", "runtime", "visualization", "utils"},
    "dashboard": {"domain", "analytics", "config", "storage", "runtime"},
}


def internal_imports(path: Path):
    """Names of the turnaround sub-packages imported by a file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("turnaround."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[1] for a in node.names if a.name.startswith("turnaround."))
    return found


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_package_only_imports_allowed_layers(package):
    for path in sorted((SRC / package).glob("*.py")):
        illegal = internal_imports(path) - ALLOWED[package] - {package}
        assert not illegal, f"{path.relative_to(SRC)} imports forbidden layer(s): {sorted(illegal)}"


def test_every_package_is_covered_by_the_rules():
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert packages == set(ALLOWED)


def test_heavy_optional_libraries_are_imported_lazily():
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not (name or "").split(".")[0] in {"ultralytics", "torch", "yt_dlp"}, f"{path.name}: eager import of {name}"
