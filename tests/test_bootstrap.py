"""Guards for PLAN 0.5: no import-time side effects, idempotent Django setup."""

import ast
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ABMEM = REPO_ROOT / "abmem"
EXCLUDED_PARTS = {"migrations", "static", "templates", "__pycache__"}


def _source_files() -> list[pathlib.Path]:
    """Return every Python source file of the abmem package."""
    return [p for p in ABMEM.rglob("*.py") if not EXCLUDED_PARTS & set(p.parts)]


def _is_call(node: ast.AST, owner: str, attr: str) -> bool:
    """True if ``node`` is a call of ``owner.attr(...)`` or ``owner.x.attr(...)``."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != attr:
        return False
    base = node.func.value
    while isinstance(base, ast.Attribute):
        base = base.value
    return isinstance(base, ast.Name) and base.id == owner


def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements, including those nested in module-level if/try blocks."""
    out: list[ast.stmt] = []
    stack = list(tree.body)
    while stack:
        stmt = stack.pop()
        out.append(stmt)
        if isinstance(stmt, (ast.If, ast.Try)):
            stack.extend(stmt.body)
            stack.extend(getattr(stmt, "orelse", []))
            stack.extend(getattr(stmt, "finalbody", []))
    return out


def test_source_files_are_found() -> None:
    assert len(_source_files()) > 20


@pytest.mark.parametrize(
    "path", _source_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_sys_path_append_anywhere(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [n.lineno for n in ast.walk(tree) if _is_call(n, "sys", "append")]
    assert not offenders, f"sys.path.append at lines {offenders}"


@pytest.mark.parametrize(
    "path", _source_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_module_level_django_setup(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for stmt in _module_level_statements(tree):
        if isinstance(stmt, ast.Expr) and _is_call(stmt.value, "django", "setup"):
            offenders.append(stmt.lineno)
    assert not offenders, f"module-level django.setup() at lines {offenders}"


def test_setup_django_is_idempotent() -> None:
    from abmem.bootstrap import setup_django

    setup_django()
    setup_django()
    from django.apps import apps

    assert apps.ready


@pytest.mark.parametrize(
    "module",
    [
        "abmem.bootstrap",
        "abmem.services.algorithms.optimization.hyperparam_optimization",
    ],
)
def test_module_imports_without_django_setup(module: str) -> None:
    """A fresh interpreter can import process entry modules before Django is configured."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "from django.apps import apps\n"
        "sys.exit(0 if not apps.ready else 3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_data_dir_is_absolute_and_independent_of_cwd(tmp_path, monkeypatch) -> None:
    from abmem.constants.paths import DATA_DIR

    monkeypatch.chdir(tmp_path)
    assert DATA_DIR.is_absolute()
    assert (DATA_DIR / "marketData.xlsx").is_file()
