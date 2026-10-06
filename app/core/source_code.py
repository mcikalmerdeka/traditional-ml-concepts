"""Read real class source from src/models for the in-page code display.

The app must always show the current implementation — single source of truth.
Extraction is cached per (file stat, class) so editing src/models is reflected
on the next page load without restarting the app.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

from app.paths import ROOT

# (rel_path, class_name) -> (mtime_ns, size, source)
_CACHE: dict[tuple[str, str], tuple[int, int, str]] = {}


def _module_path(rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p


def get_class_source(rel_path: str, class_name: str) -> str:
    """Return the full source text of `class_name` from the module file.

    rel_path is repo-root-relative (or absolute). Raises FileNotFoundError
    when the file is missing, ValueError when the class is absent.
    """
    path = _module_path(rel_path)
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")

    stat = path.stat()
    key = (rel_path, class_name)
    cached = _CACHE.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    module_name = f"_source_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # register before exec: inspect.getsource resolves the class's module via
    # sys.modules; an unregistered module makes inspect treat it as built-in
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # noqa: S102 — repo-local source, trusted

    try:
        source = inspect.getsource(getattr(module, class_name))
    except AttributeError as exc:
        raise ValueError(f"class not found: {class_name} in {rel_path}") from exc

    _CACHE[key] = (stat.st_mtime_ns, stat.st_size, source)
    return source
