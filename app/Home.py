import sys
from pathlib import Path

# Launch-contract bootstrap (cleanup minor #3): `streamlit run` puts ONLY the
# script's directory on sys.path — no cwd, no PYTHONPATH — so the repo root
# must be added here before any app.* import can resolve. Guarded so reruns
# of this script (Streamlit's run-on-save) never grow sys.path. Home is not
# importable before the insert — chicken-and-egg — so no app.paths import.
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st

from app.navigation import build_nav

st.set_page_config(page_title="Traditional ML Concepts", layout="wide")
st.navigation(build_nav()).run()
