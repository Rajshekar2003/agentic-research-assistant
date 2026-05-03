"""
Session-level test fixtures for the Agentic Research Assistant backend.

Why module-level os.environ over a session-scoped monkeypatch fixture:
pytest loads conftest.py before collecting and importing test modules, so
setting os.environ at module-load time here guarantees the dummy keys are
present before any test file runs `from app.main import app` (which may
trigger TestClient instantiation and, with it, the lifespan context that
calls get_settings()). A session-scoped monkeypatch fixture using
pytest.MonkeyPatch() would also work but runs slightly later in the
collection cycle, risking a race where get_settings() is called before
the fixture sets the keys.

os.environ.setdefault() is used so that real keys already in the environment
(e.g., from a CI secrets store) are not overwritten.
"""

import os

import pytest

# Set dummy keys at module-load time — before any test file import can
# trigger Settings() construction.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

# Clear any cached Settings built before these vars were set (e.g., during
# a prior pytest session in the same process, or from a stale import).
from app.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True, scope="session")
def configure_test_env() -> None:
    """Ensure dummy API keys remain set for the entire test session.

    The actual env-var population happens at module load above; this fixture
    documents the intent and provides a clean hook for future session-level
    setup (e.g., spinning up a test database).

    Yields:
        None
    """
    yield
