# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Make 'load_from_env' importable as a top-level module for train.py
sys.path.insert(0, str(ROOT / "src" / "assignment"))

# Also keep these (helpful for other imports)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

