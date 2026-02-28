import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator

# ---------------------------------------------------------------------------
# Lightweight stand-ins for Anthropic SDK content blocks
# ---------------------------------------------------------------------------


class ThinkingBlock:
    """Simulates an Anthropic ThinkingBlock — has no .text attribute."""

    type = "thinking"
    thinking = "I'm thinking..."


class TextBlock:
    """Simulates an Anthropic TextBlock."""

    type = "text"

    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    """Simulates an Anthropic ToolUseBlock."""

    type = "tool_use"

    def __init__(self, id_: str, name: str, input_: dict):
        self.id = id_
        self.name = name
        self.input = input_


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    with patch("ai_generator.anthropic.Anthropic") as mock_cls:
        yield mock_cls.return_value


def _make_generator(mock_client):
    return AIGenerator(api_key="test-key", model="claude-test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_response_no_tools(mock_client):
    """When stop_reason is end_turn, content text is returned directly."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [TextBlock("Hello, world!")]
    mock_client.messages.create.return_value = response

    gen = _make_generator(mock_client)
    result = gen.generate_response(query="Hi")

    assert result == "Hello, world!"


def test_direct_response_with_thinking_block(mock_client):
    """Direct response must still work when content[0] is a non-TextBlock."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [ThinkingBlock(), TextBlock("Direct answer")]
    mock_client.messages.create.return_value = response

    gen = _make_generator(mock_client)
    result = gen.generate_response(query="Hi")

    assert result == "Direct answer"


def test_tool_use_triggers_handle_execution(mock_client):
    """Two API calls are made when stop_reason is tool_use."""
    tool_block = ToolUseBlock("call_1", "search_course_content", {"query": "test"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_block])
    final_response = MagicMock(stop_reason="end_turn", content=[TextBlock("Answer")])
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "found content"

    gen = _make_generator(mock_client)
    result = gen.generate_response(
        query="What is in lesson 1?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 2
    assert result == "Answer"


def test_tool_execute_called_with_correct_args(mock_client):
    """tool_manager.execute_tool is called with the tool name and its input args."""
    tool_block = ToolUseBlock(
        "call_1", "search_course_content", {"query": "lesson 1 content"}
    )
    first_response = MagicMock(stop_reason="tool_use", content=[tool_block])
    final_response = MagicMock(stop_reason="end_turn", content=[TextBlock("answer")])
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "found"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="What is in lesson 1?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="lesson 1 content"
    )


def test_tool_results_included_in_final_messages(mock_client):
    """Final API call messages must contain a tool_result block."""
    tool_block = ToolUseBlock("call_1", "search_course_content", {"query": "python"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_block])
    final_response = MagicMock(stop_reason="end_turn", content=[TextBlock("answer")])
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "course content here"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    final_call_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    last_user_messages = [m for m in final_call_messages if m.get("role") == "user"]
    last_user = last_user_messages[-1]
    assert isinstance(last_user["content"], list)
    assert any(item.get("type") == "tool_result" for item in last_user["content"])


def test_final_call_excludes_tools_key(mock_client):
    """tools key must NOT appear in the forced-final API call after 2 tool-use rounds."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "python"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "django"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_block1])
    second_response = MagicMock(stop_reason="tool_use", content=[tool_block2])
    final_response = MagicMock(stop_reason="end_turn", content=[TextBlock("answer")])
    mock_client.messages.create.side_effect = [
        first_response,
        second_response,
        final_response,
    ]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "content"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    final_kwargs = mock_client.messages.create.call_args_list[2].kwargs
    assert "tools" not in final_kwargs


def test_final_response_accesses_content_0_text(mock_client):
    """Text is extracted correctly even when content[0] is a ThinkingBlock."""
    tool_block = ToolUseBlock("call_1", "search_course_content", {"query": "python"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_block])
    # ThinkingBlock has no .text; TextBlock carries the real answer
    final_response = MagicMock(
        stop_reason="end_turn",
        content=[ThinkingBlock(), TextBlock("The actual answer")],
    )
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "content"

    gen = _make_generator(mock_client)
    result = gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "The actual answer"


def test_multiple_tool_calls_all_executed(mock_client):
    """All tool_use blocks in a single response are executed."""
    tool1 = ToolUseBlock("call_1", "search_course_content", {"query": "python"})
    tool2 = ToolUseBlock("call_2", "search_course_content", {"query": "django"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool1, tool2])
    final_response = MagicMock(
        stop_reason="end_turn", content=[TextBlock("Combined answer")]
    )
    mock_client.messages.create.side_effect = [first_response, final_response]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="Python and Django",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert tool_manager.execute_tool.call_count == 2


def test_two_round_tool_use_makes_three_api_calls(mock_client):
    """Two tool-use rounds followed by end_turn results in exactly 3 API calls."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "first"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "second"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block1])
    r2 = MagicMock(stop_reason="tool_use", content=[tool_block2])
    r3 = MagicMock(stop_reason="end_turn", content=[TextBlock("Final answer")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    result = gen.generate_response(
        query="complex question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 3
    assert result == "Final answer"


def test_two_round_tools_executed_twice(mock_client):
    """execute_tool is called once per tool-use round (2 total for 2 rounds)."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "first"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "second"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block1])
    r2 = MagicMock(stop_reason="tool_use", content=[tool_block2])
    r3 = MagicMock(stop_reason="end_turn", content=[TextBlock("Final answer")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="complex question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert tool_manager.execute_tool.call_count == 2


def test_intermediate_calls_include_tools(mock_client):
    """Tools are present in the first 2 API calls but absent in the forced-final call."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "first"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "second"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block1])
    r2 = MagicMock(stop_reason="tool_use", content=[tool_block2])
    r3 = MagicMock(stop_reason="end_turn", content=[TextBlock("answer")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    calls = mock_client.messages.create.call_args_list
    assert "tools" in calls[0].kwargs
    assert "tools" in calls[1].kwargs
    assert "tools" not in calls[2].kwargs


def test_single_round_returns_text_without_forced_final(mock_client):
    """When the second in-loop call returns end_turn, no forced-final call is made (2 calls total)."""
    tool_block = ToolUseBlock("call_1", "search_course_content", {"query": "test"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block])
    r2 = MagicMock(stop_reason="end_turn", content=[TextBlock("Direct answer")])
    mock_client.messages.create.side_effect = [r1, r2]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    result = gen.generate_response(
        query="question",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 2
    assert result == "Direct answer"


def test_message_history_grows_across_two_rounds(mock_client):
    """After 2 tool-use rounds the forced-final call receives 5 messages in order."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "first"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "second"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block1])
    r2 = MagicMock(stop_reason="tool_use", content=[tool_block2])
    r3 = MagicMock(stop_reason="end_turn", content=[TextBlock("answer")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
    assert len(messages) == 5
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_max_rounds_capped_at_two(mock_client):
    """Loop never exceeds MAX_ROUNDS=2; a 3rd tool_use response is never requested."""
    tool_block1 = ToolUseBlock("call_1", "search_course_content", {"query": "first"})
    tool_block2 = ToolUseBlock("call_2", "search_course_content", {"query": "second"})
    r1 = MagicMock(stop_reason="tool_use", content=[tool_block1])
    r2 = MagicMock(stop_reason="tool_use", content=[tool_block2])
    r3 = MagicMock(stop_reason="end_turn", content=[TextBlock("synthesised")])
    mock_client.messages.create.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_client)
    gen.generate_response(
        query="query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    # Exactly 3 calls: round 1, round 2, forced-final — no 4th call
    assert mock_client.messages.create.call_count == 3
