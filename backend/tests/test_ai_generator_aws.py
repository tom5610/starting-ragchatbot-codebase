import pytest
from unittest.mock import MagicMock, patch

from ai_generator_aws import AIGenerator


# ---------------------------------------------------------------------------
# Helpers — minimal Bedrock response structures
# ---------------------------------------------------------------------------

def bedrock_text_response(text: str) -> dict:
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
    }


def bedrock_tool_use_response(tool_use_id: str, name: str, input_: dict) -> dict:
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_}}
                ],
            }
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_boto_client():
    with patch("ai_generator_aws.boto3.client") as mock_fn:
        yield mock_fn.return_value


def _make_generator(mock_boto_client):
    return AIGenerator(model="test-model", region="us-east-1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_convert_tools_format(mock_boto_client):
    """Anthropic-format tools are correctly converted to Bedrock toolConfig."""
    gen = _make_generator(mock_boto_client)
    anthropic_tools = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    result = gen._convert_tools(anthropic_tools)

    assert "tools" in result
    assert "toolChoice" in result
    tool_spec = result["tools"][0]["toolSpec"]
    assert tool_spec["name"] == "search_course_content"
    assert tool_spec["description"] == "Search course materials"
    assert tool_spec["inputSchema"]["json"] == anthropic_tools[0]["input_schema"]


def test_tool_use_triggers_handle_execution(mock_boto_client):
    """Bedrock stopReason == tool_use triggers _handle_tool_execution."""
    first = bedrock_tool_use_response("call_1", "search_course_content", {"query": "python"})
    final = bedrock_text_response("Final answer")
    mock_boto_client.converse.side_effect = [first, final]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "found content"

    gen = _make_generator(mock_boto_client)
    result = gen.generate_response(
        query="What is in lesson 1?",
        tools=[
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ],
        tool_manager=tool_manager,
    )

    assert mock_boto_client.converse.call_count == 2
    assert result == "Final answer"


def test_toolconfig_included_in_final_call(mock_boto_client):
    """toolConfig must be present in all 3 calls including the forced-final after 2 rounds."""
    first = bedrock_tool_use_response("call_1", "search_course_content", {"query": "python"})
    second = bedrock_tool_use_response("call_2", "search_course_content", {"query": "django"})
    final = bedrock_text_response("Final answer")
    mock_boto_client.converse.side_effect = [first, second, final]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "content"

    gen = _make_generator(mock_boto_client)
    gen.generate_response(
        query="query",
        tools=[
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ],
        tool_manager=tool_manager,
    )

    calls = mock_boto_client.converse.call_args_list
    assert "toolConfig" in calls[0].kwargs
    assert "toolConfig" in calls[1].kwargs
    assert "toolConfig" in calls[2].kwargs


def test_extract_text_from_bedrock_response(mock_boto_client):
    """_extract_text correctly pulls the text value from a Bedrock response dict."""
    gen = _make_generator(mock_boto_client)
    response = {
        "output": {
            "message": {
                "content": [{"text": "Hello from Bedrock"}]
            }
        }
    }
    assert gen._extract_text(response) == "Hello from Bedrock"


def test_bedrock_tool_results_format(mock_boto_client):
    """Tool results sent back to Bedrock use toolResult / toolUseId format."""
    first = bedrock_tool_use_response("call_1", "search_course_content", {"query": "python"})
    final = bedrock_text_response("Answer")
    mock_boto_client.converse.side_effect = [first, final]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result text"

    gen = _make_generator(mock_boto_client)
    gen.generate_response(
        query="query",
        tools=[
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ],
        tool_manager=tool_manager,
    )

    # Inspect the messages passed to the final converse call
    final_messages = mock_boto_client.converse.call_args_list[1].kwargs["messages"]
    user_messages = [m for m in final_messages if m.get("role") == "user"]
    last_user = user_messages[-1]
    # Each tool result block must use Bedrock's toolResult key
    assert any("toolResult" in item for item in last_user["content"])
    tool_result_block = next(item["toolResult"] for item in last_user["content"] if "toolResult" in item)
    assert tool_result_block["toolUseId"] == "call_1"


_TOOLS = [
    {
        "name": "search_course_content",
        "description": "Search",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


def test_bedrock_two_round_makes_three_converse_calls(mock_boto_client):
    """Two tool-use rounds followed by end_turn results in exactly 3 converse calls."""
    r1 = bedrock_tool_use_response("call_1", "search_course_content", {"query": "first"})
    r2 = bedrock_tool_use_response("call_2", "search_course_content", {"query": "second"})
    r3 = bedrock_text_response("Final answer")
    mock_boto_client.converse.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_boto_client)
    result = gen.generate_response(query="complex question", tools=_TOOLS, tool_manager=tool_manager)

    assert mock_boto_client.converse.call_count == 3
    assert result == "Final answer"


def test_bedrock_two_rounds_execute_tools_twice(mock_boto_client):
    """execute_tool is called once per round — twice for a 2-round sequence."""
    r1 = bedrock_tool_use_response("call_1", "search_course_content", {"query": "first"})
    r2 = bedrock_tool_use_response("call_2", "search_course_content", {"query": "second"})
    r3 = bedrock_text_response("Final answer")
    mock_boto_client.converse.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_boto_client)
    gen.generate_response(query="complex question", tools=_TOOLS, tool_manager=tool_manager)

    assert tool_manager.execute_tool.call_count == 2


def test_bedrock_toolconfig_present_in_forced_final(mock_boto_client):
    """toolConfig must be present in the forced-final call after 2 tool-use rounds."""
    r1 = bedrock_tool_use_response("call_1", "search_course_content", {"query": "first"})
    r2 = bedrock_tool_use_response("call_2", "search_course_content", {"query": "second"})
    r3 = bedrock_text_response("answer")
    mock_boto_client.converse.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_boto_client)
    gen.generate_response(query="query", tools=_TOOLS, tool_manager=tool_manager)

    assert "toolConfig" in mock_boto_client.converse.call_args_list[2].kwargs


def test_bedrock_message_history_structure_after_two_rounds(mock_boto_client):
    """Forced-final call receives 5 messages with correct Bedrock structure."""
    r1 = bedrock_tool_use_response("call_1", "search_course_content", {"query": "first"})
    r2 = bedrock_tool_use_response("call_2", "search_course_content", {"query": "second"})
    r3 = bedrock_text_response("answer")
    mock_boto_client.converse.side_effect = [r1, r2, r3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_boto_client)
    gen.generate_response(query="query", tools=_TOOLS, tool_manager=tool_manager)

    messages = mock_boto_client.converse.call_args_list[2].kwargs["messages"]
    assert len(messages) == 5
    # Verify Bedrock toolUse/toolResult shapes are present in the interleaved turns
    assert any("toolUse" in block for block in messages[1]["content"])
    assert any("toolResult" in block for block in messages[2]["content"])
    assert any("toolUse" in block for block in messages[3]["content"])
    assert any("toolResult" in block for block in messages[4]["content"])


def test_bedrock_single_round_returns_text_without_forced_final(mock_boto_client):
    """When the second in-loop call returns end_turn, no forced-final call is made."""
    r1 = bedrock_tool_use_response("call_1", "search_course_content", {"query": "test"})
    r2 = bedrock_text_response("Direct answer")
    mock_boto_client.converse.side_effect = [r1, r2]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    gen = _make_generator(mock_boto_client)
    result = gen.generate_response(query="question", tools=_TOOLS, tool_manager=tool_manager)

    assert mock_boto_client.converse.call_count == 2
    assert result == "Direct answer"
