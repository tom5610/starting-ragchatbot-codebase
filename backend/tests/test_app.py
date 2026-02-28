import pytest

# app_client fixture is defined in conftest.py and yields (TestClient, mock_rag_instance).
# Each test configures the mock's return values before calling the endpoint so that
# tests sharing the module-scoped fixture remain independent.


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_success_returns_answer_sources_session(app_client):
    """200 response contains answer, sources list, and session_id."""
    client, mock_rag = app_client
    mock_rag.session_manager.create_session.return_value = "session_1"
    mock_rag.query.return_value = (
        "Python is a high-level language.",
        [{"name": "Python 101 - Lesson 1", "link": "http://example.com/l1"}],
    )

    response = client.post("/api/query", json={"query": "What is Python?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Python is a high-level language."
    assert data["session_id"] == "session_1"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["name"] == "Python 101 - Lesson 1"
    assert data["sources"][0]["link"] == "http://example.com/l1"


def test_query_creates_new_session_when_none_provided(app_client):
    """A fresh session_id is created and returned when the request omits session_id."""
    client, mock_rag = app_client
    mock_rag.session_manager.reset_mock()
    mock_rag.session_manager.create_session.return_value = "auto_session"
    mock_rag.query.return_value = ("Answer", [])

    response = client.post("/api/query", json={"query": "Tell me about Django"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "auto_session"
    mock_rag.session_manager.create_session.assert_called_once()


def test_query_uses_provided_session_id(app_client):
    """When a session_id is given, it is forwarded and create_session is not called."""
    client, mock_rag = app_client
    mock_rag.session_manager.reset_mock()
    mock_rag.query.return_value = ("Contextual answer", [])

    response = client.post(
        "/api/query",
        json={"query": "Follow-up question", "session_id": "existing_session"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "existing_session"
    mock_rag.session_manager.create_session.assert_not_called()


def test_query_missing_query_field_returns_422(app_client):
    """Request without the required 'query' field is rejected with 422."""
    client, _ = app_client

    response = client.post("/api/query", json={})

    assert response.status_code == 422


def test_query_sources_can_be_empty(app_client):
    """Sources list may be empty when no relevant content was retrieved."""
    client, mock_rag = app_client
    mock_rag.session_manager.create_session.return_value = "session_1"
    mock_rag.query.return_value = ("General answer with no sources.", [])

    response = client.post("/api/query", json={"query": "Anything"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_query_rag_error_returns_500(app_client):
    """An exception raised by the RAG system results in a 500 response."""
    client, mock_rag = app_client
    mock_rag.query.side_effect = RuntimeError("Vector store unavailable")

    response = client.post("/api/query", json={"query": "What is Python?"})

    assert response.status_code == 500
    mock_rag.query.side_effect = None  # reset for subsequent tests


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

def test_get_courses_returns_stats(app_client):
    """200 response contains total_courses count and list of course_titles."""
    client, mock_rag = app_client
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python 101", "Django Basics", "FastAPI Deep Dive"],
    }

    response = client.get("/api/courses")

    assert response.status_code == 200
    data = response.json()
    assert data["total_courses"] == 3
    assert "Python 101" in data["course_titles"]
    assert len(data["course_titles"]) == 3


def test_get_courses_empty_catalog(app_client):
    """Works correctly when no courses have been loaded."""
    client, mock_rag = app_client
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }

    response = client.get("/api/courses")

    assert response.status_code == 200
    data = response.json()
    assert data["total_courses"] == 0
    assert data["course_titles"] == []


def test_get_courses_error_returns_500(app_client):
    """An exception from get_course_analytics results in a 500 response."""
    client, mock_rag = app_client
    mock_rag.get_course_analytics.side_effect = RuntimeError("DB unavailable")

    response = client.get("/api/courses")

    assert response.status_code == 500
    mock_rag.get_course_analytics.side_effect = None  # reset for subsequent tests


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

def test_delete_session_returns_ok(app_client):
    """Deleting a session returns {"status": "ok"} and delegates to session_manager."""
    client, mock_rag = app_client
    mock_rag.session_manager.reset_mock()

    response = client.delete("/api/session/my_session_42")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_rag.session_manager.clear_session.assert_called_once_with("my_session_42")


def test_delete_session_different_ids_are_forwarded(app_client):
    """The session_id path parameter is passed verbatim to clear_session."""
    client, mock_rag = app_client
    mock_rag.session_manager.reset_mock()

    client.delete("/api/session/alpha")
    client.delete("/api/session/beta")

    calls = [c.args[0] for c in mock_rag.session_manager.clear_session.call_args_list]
    assert "alpha" in calls
    assert "beta" in calls
