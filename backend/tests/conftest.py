import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Make backend/ importable from any test file without package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_rag_system():
    """Configured MagicMock that mimics the RAGSystem public interface."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "session_1"
    mock.query.return_value = (
        "Test answer",
        [{"name": "Source A", "link": "http://example.com"}],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python 101", "Django Basics"],
    }
    return mock


@pytest.fixture(scope="module")
def app_client():
    """
    Yield (TestClient, mock_rag_instance) for FastAPI endpoint tests.

    Patches out RAGSystem and StaticFiles before importing app.py so that
    the module-level RAGSystem() call and the ../frontend directory check
    do not run against real components.
    """
    sys.modules.pop("app", None)

    _mock_rag = MagicMock()
    _mock_rag.session_manager.create_session.return_value = "session_1"
    _mock_rag.query.return_value = ("Default answer", [])
    _mock_rag.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }

    with patch("fastapi.staticfiles.StaticFiles"), \
         patch("rag_system.RAGSystem", return_value=_mock_rag):
        import app as _app_module  # imported fresh within the patches
        _app_module.rag_system = _mock_rag  # replace the module-level instance

        from fastapi.testclient import TestClient

        client = TestClient(_app_module.app, raise_server_exceptions=False)
        yield client, _mock_rag

    sys.modules.pop("app", None)
