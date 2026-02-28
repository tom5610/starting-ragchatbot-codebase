import pytest
from unittest.mock import MagicMock, patch

# conftest.py ensures backend/ is on sys.path


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_bundle():
    """RAGSystem with its AI generator and vector store mocked out."""
    mock_gen = MagicMock()
    mock_gen.generate_response.return_value = "Mocked AI response"
    mock_gen_cls = MagicMock(return_value=mock_gen)

    with (
        patch("rag_system.AIGenerator", mock_gen_cls),
        patch("rag_system.VectorStore"),
        patch("rag_system.DocumentProcessor"),
    ):

        from config import Config
        from rag_system import RAGSystem

        system = RAGSystem(Config())

    # Patches are undone, but system.ai_generator is already mock_gen
    return system, mock_gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_returns_tuple_response_sources(rag_bundle):
    rag, _ = rag_bundle
    result = rag.query("What is Python?")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


def test_query_passes_tool_definitions_to_generator(rag_bundle):
    rag, mock_gen = rag_bundle
    rag.query("What is Python?")
    call_kwargs = mock_gen.generate_response.call_args.kwargs
    assert "tools" in call_kwargs
    assert isinstance(call_kwargs["tools"], list)
    assert len(call_kwargs["tools"]) > 0


def test_query_sources_reset_after_retrieval(rag_bundle):
    """tool_manager.reset_sources() is called after get_last_sources()."""
    rag, _ = rag_bundle
    # Spy on tool_manager methods
    original_reset = rag.tool_manager.reset_sources
    reset_calls = []

    def spy_reset():
        reset_calls.append(True)
        original_reset()

    rag.tool_manager.reset_sources = spy_reset
    rag.query("query")
    assert len(reset_calls) == 1


def test_query_history_fetched_for_existing_session(rag_bundle):
    """Conversation history is passed to the generator on subsequent queries."""
    rag, mock_gen = rag_bundle
    session_id = rag.session_manager.create_session()

    # First query seeds the history
    rag.query("First question", session_id)
    mock_gen.generate_response.reset_mock()

    # Second query should have history
    rag.query("Follow-up question", session_id)
    call_kwargs = mock_gen.generate_response.call_args.kwargs
    assert call_kwargs.get("conversation_history") is not None


def test_query_history_none_for_new_session(rag_bundle):
    """Brand-new session has no history on its first query."""
    rag, mock_gen = rag_bundle
    session_id = rag.session_manager.create_session()

    rag.query("First ever question", session_id)
    call_kwargs = mock_gen.generate_response.call_args.kwargs
    assert call_kwargs.get("conversation_history") is None


def test_query_exchange_added_to_session(rag_bundle):
    """query+response are stored in session history after each call."""
    rag, _ = rag_bundle
    session_id = rag.session_manager.create_session()

    rag.query("What is Python?", session_id)
    history = rag.session_manager.get_conversation_history(session_id)

    assert history is not None
    assert "What is Python?" in history
    assert "Mocked AI response" in history


def test_query_with_unknown_session_id_does_not_raise(rag_bundle):
    """A stale / unknown session_id returns None history without raising."""
    rag, mock_gen = rag_bundle
    # Use a session_id that was never created
    result = rag.query("Query with stale session", "stale_session_xyz_999")
    assert result is not None
    call_kwargs = mock_gen.generate_response.call_args.kwargs
    assert call_kwargs.get("conversation_history") is None
