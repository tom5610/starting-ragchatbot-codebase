import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_results(docs, metas, distances=None, error=None):
    if error:
        return SearchResults.empty(error)
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.5] * len(docs),
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "http://example.com/lesson"
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_search_returns_formatted_text(mock_store):
    mock_store.search.return_value = make_results(
        docs=["Intro content"],
        metas=[{"course_title": "Python 101", "lesson_number": 1}],
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="basics")
    assert "[Python 101 - Lesson 1]" in result
    assert "Intro content" in result


def test_empty_results_returns_no_content_message(mock_store):
    mock_store.search.return_value = make_results(docs=[], metas=[])
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="something")
    assert "No relevant content found" in result


def test_error_from_vector_store_returned_as_string(mock_store):
    mock_store.search.return_value = make_results(
        docs=[], metas=[], error="Search error: something went wrong"
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="something")
    assert "Search error" in result


def test_course_name_filter_passed_to_store(mock_store):
    mock_store.search.return_value = make_results(docs=[], metas=[])
    tool = CourseSearchTool(mock_store)
    tool.execute(query="content", course_name="Python 101")
    mock_store.search.assert_called_once_with(
        query="content", course_name="Python 101", lesson_number=None
    )


def test_lesson_number_filter_passed_to_store(mock_store):
    mock_store.search.return_value = make_results(docs=[], metas=[])
    tool = CourseSearchTool(mock_store)
    tool.execute(query="content", lesson_number=3)
    mock_store.search.assert_called_once_with(
        query="content", course_name=None, lesson_number=3
    )


def test_sources_populated_after_search(mock_store):
    mock_store.search.return_value = make_results(
        docs=["Content"],
        metas=[{"course_title": "Python 101", "lesson_number": 2}],
    )
    mock_store.get_lesson_link.return_value = "http://example.com/l2"
    tool = CourseSearchTool(mock_store)
    tool.execute(query="something")
    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["name"] == "Python 101 - Lesson 2"
    assert tool.last_sources[0]["link"] == "http://example.com/l2"


def test_lesson_zero_appears_in_header(mock_store):
    """Lesson 0 (falsy int) must NOT be silently dropped from the header."""
    mock_store.search.return_value = make_results(
        docs=["Intro content"],
        metas=[{"course_title": "Python 101", "lesson_number": 0}],
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="intro")
    assert "Lesson 0" in result


def test_lesson_zero_appears_in_sources(mock_store):
    """Lesson 0 must be tracked correctly in last_sources."""
    mock_store.search.return_value = make_results(
        docs=["Intro content"],
        metas=[{"course_title": "Python 101", "lesson_number": 0}],
    )
    mock_store.get_lesson_link.return_value = "http://example.com/l0"
    tool = CourseSearchTool(mock_store)
    tool.execute(query="intro")
    assert len(tool.last_sources) == 1
    assert "Lesson 0" in tool.last_sources[0]["name"]
