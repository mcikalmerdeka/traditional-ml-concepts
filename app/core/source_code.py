"""Read real class source from src/models for the in-page code display.

The app must always show the current implementation — single source of truth.
Extraction is cached per (file stat, class) so editing src/models is reflected
on the next page load without restarting the app.

Extraction is a pure ast.parse of the file — never an import. Some model
modules use relative imports (`from .tree_models import ...`), which cannot
execute standalone, and even where they run, exec'ing a module just to show
its text has no upside: the parse cannot have side effects on the app.
"""

import ast
from pathlib import Path

from app.paths import ROOT

# (rel_path, class_name) -> (mtime_ns, size, source)
_CACHE: dict[tuple[str, str], tuple[int, int, str]] = {}


def _module_path(rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p


def _class_segment(text: str, tree: ast.Module, class_name: str) -> str:
    """Slice a top-level ClassDef (decorator lines included) out of `text`."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = text.splitlines()
            return "\n".join(lines[start - 1 : node.end_lineno])
    raise ValueError(f"class not found: {class_name}")


def get_class_source(rel_path: str, class_name: str) -> str:
    """Return the full source text of `class_name` from the module file.

    rel_path is repo-root-relative (or absolute). Raises FileNotFoundError
    when the file is missing, ValueError when the class is absent or the
    file does not parse.
    """
    path = _module_path(rel_path)
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")

    stat = path.stat()
    key = (rel_path, class_name)
    cached = _CACHE.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    text = path.read_text(encoding="utf-8")
    source = _class_segment(text, ast.parse(text), class_name)
    _CACHE[key] = (stat.st_mtime_ns, stat.st_size, source)
    return source
