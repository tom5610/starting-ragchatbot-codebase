import boto3
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Claude via Amazon Bedrock Converse API"""

    MAX_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **get_course_outline**: Use for outline or structure questions (e.g. "what lessons does X have?", "show me the course outline"). Returns the course title, course link, and all lesson numbers and titles.
- **search_course_content**: Use for questions about specific content or concepts inside a course. You may make up to 2 sequential searches when the first result is insufficient or when the query requires information from two distinct sources (e.g., comparing two courses, or finding a course that covers a topic from a specific lesson). Use a second search only when genuinely needed.
- Synthesize tool results into accurate, fact-based responses
- If a tool returns no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, model: str, region: str = "us-east-1"):
        # Bedrock uses AWS credentials from environment/IAM — no API key needed.
        # model should be a Bedrock model ID, e.g.:
        #   "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "modelId": self.model,
            "inferenceConfig": {
                "temperature": 0,
                "maxTokens": 800,
            },
        }

    def _convert_tools(self, tools: List[Dict]) -> Dict:
        """Convert Anthropic-format tool definitions to Bedrock toolConfig format."""
        bedrock_tools = [
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {"json": tool["input_schema"]},
                }
            }
            for tool in tools
        ]
        return {
            "tools": bedrock_tools,
            "toolChoice": {"auto": {}},
        }

    def _extract_text(self, response: Dict) -> str:
        """Extract text from a Bedrock Converse response message."""
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                return block["text"]
        return ""

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_ROUNDS sequential tool-call rounds before forcing a
        final text-only synthesis call.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (Anthropic format — converted internally)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": [{"text": query}]}]

        api_params = {
            **self.base_params,
            "system": [{"text": system_content}],
            "messages": messages,
        }

        if tools:
            api_params["toolConfig"] = self._convert_tools(tools)

        for _ in range(self.MAX_ROUNDS):
            response = self.client.converse(**api_params)

            # No tool call or no tool_manager — return text directly
            if response["stopReason"] != "tool_use" or not tool_manager:
                return self._extract_text(response)

            # Append assistant's tool-use turn
            assistant_message = response["output"]["message"]
            messages.append(assistant_message)

            # Execute all tool_use blocks in this response
            tool_results = []
            for block in assistant_message["content"]:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    result = tool_manager.execute_tool(
                        tool_use["name"], **tool_use["input"]
                    )
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"text": result}],
                            }
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        # Round limit reached — force a final call. toolConfig must be kept because
        # messages contain toolUse/toolResult blocks (Bedrock rejects calls without it).
        final_params = {
            **self.base_params,
            "system": api_params["system"],
            "messages": messages,
            "toolConfig": api_params["toolConfig"],
        }
        final_response = self.client.converse(**final_params)
        return self._extract_text(final_response)
