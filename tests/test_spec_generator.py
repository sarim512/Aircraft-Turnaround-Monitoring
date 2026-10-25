"""The generated specification must be complete: no doc gaps and every function present."""
import ast

import generate_spec as gen


def test_no_documentation_gaps_and_placeholders_filled():
    text, problems = gen.build_spec()
    assert problems == []
    assert "{{" not in text
    for heading in ("### 1. High-Level Architecture", "### 2. Data Schemas", "### 3. Comprehensive File", "### 4. Configuration", "### 5. Modification"):
        assert heading in text
    assert "```mermaid" in text and "CREATE TABLE IF NOT EXISTS sessions" in text


def test_every_function_and_class_appears_in_the_spec():
    text, _ = gen.build_spec()
    for path in gen.python_files():
        rel = path.relative_to(gen.REPO_ROOT).as_posix()
        assert f"`{rel}`" in text, f"file missing from spec: {rel}"
        if rel.startswith("tests/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                assert f"`{node.name}" in text, f"class missing: {rel}:{node.name}"
            elif isinstance(node, ast.FunctionDef):
                assert f"`{node.name}`" in text or f".{node.name}`" in text, f"function missing: {rel}:{node.name}"


def test_parse_docstring_sections():
    doc = gen.parse_docstring("""Summary line.

    More text.

    Args:
        a: first
            continued
        b: second
    Returns:
        thing
    Logic:
        1. step
    """)
    assert doc.summary == "Summary line." and doc.description == "More text."
    assert gen.parse_named_items(doc.sections["Args"]) == {"a": "first continued", "b": "second"}
    assert doc.sections["Returns"] == "thing" and doc.sections["Logic"].startswith("1. step")
