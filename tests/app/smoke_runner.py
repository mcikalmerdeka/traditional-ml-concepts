"""Rendered by AppTest.from_file with SMOKE_CARD_ID env var set."""

import os

from app.paths import ensure_root_on_path

ensure_root_on_path()

from app.registry.discovery import get_card  # noqa: E402 (after bootstrap)
from app.core.page import run_card_page  # noqa: E402

run_card_page(os.environ["SMOKE_CARD_ID"])
