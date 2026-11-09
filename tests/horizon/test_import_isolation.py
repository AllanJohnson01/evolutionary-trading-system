"""
Import isolation: every .py file in the Horizon package must never import
from evo_trader.portfolio.manager or evo_trader.evolution.engine.
That import boundary IS Horizon's isolation guarantee — structural, not a
flag that can be misconfigured.
"""
from __future__ import annotations

import ast
from pathlib import Path

HORIZON_SRC = Path(__file__).parent.parent.parent / "src" / "evo_trader" / "horizon"

FORBIDDEN = {
    "evo_trader.portfolio.manager",
    "evo_trader.evolution.engine",
}


def _all_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _check_file(path: Path) -> list[str]:
    violations = []
    for imp in _all_imports(path.read_text(encoding="utf-8")):
        for forbidden in FORBIDDEN:
            if forbidden in imp:
                violations.append(f"{path.name}: imports {imp!r}")
    return violations


def test_no_horizon_module_imports_portfolio_manager():
    py_files = list(HORIZON_SRC.glob("*.py"))
    assert py_files, f"No .py files found under {HORIZON_SRC}"
    violations = []
    for path in py_files:
        for imp in _all_imports(path.read_text(encoding="utf-8")):
            if "portfolio.manager" in imp:
                violations.append(f"{path.name}: {imp!r}")
    assert not violations, f"portfolio.manager imported by Horizon: {violations}"


def test_no_horizon_module_imports_evolution_engine():
    py_files = list(HORIZON_SRC.glob("*.py"))
    assert py_files
    violations = []
    for path in py_files:
        for imp in _all_imports(path.read_text(encoding="utf-8")):
            if "evolution.engine" in imp:
                violations.append(f"{path.name}: {imp!r}")
    assert not violations, f"evolution.engine imported by Horizon: {violations}"


def test_self_review_does_not_import_forbidden_modules():
    """self_review.py specifically must stay isolated."""
    path = HORIZON_SRC / "self_review.py"
    assert path.exists(), "self_review.py not found"
    violations = _check_file(path)
    assert not violations, f"Forbidden imports in self_review.py: {violations}"
