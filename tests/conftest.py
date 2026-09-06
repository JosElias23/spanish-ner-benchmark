import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The repository root too, so tests can import the `serve` package.
sys.path.insert(0, str(ROOT))

from spanish_ner.data import load_all  # noqa: E402
from spanish_ner.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def splits(config):
    """All three official splits, loaded once per test session."""
    return load_all(config, include_test=True)
