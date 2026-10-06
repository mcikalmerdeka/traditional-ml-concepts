"""Bootstrap so `app.*` imports work regardless of the launch directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
