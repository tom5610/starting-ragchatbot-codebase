import boto3
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Claude via Amazon Bedrock Converse API"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

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

        api_params = {
            **self.base_params,
            "system": [{"text": system_content}],
            "messages": [{"role": "user", "content": [{"text": query}]}],
        }

        if tools:
            api_params["toolConfig"] = self._convert_tools(tools)

        print("before calling model")
        response = self.client.converse(**api_params)
        print("after calling model")
        print(f"{response=}")

        if response["stopReason"] == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        return self._extract_text(response)

    def _handle_tool_execution(
        self, initial_response: Dict, base_params: Dict[str, Any], tool_manager
    ) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The Bedrock response containing tool use requests
            base_params: Base API parameters used for the initial call
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()

        # Add assistant's tool-use turn to the conversation
        assistant_message = initial_response["output"]["message"]
        messages.append(assistant_message)

        # Execute all tool calls and collect results
        tool_results = []
        for block in assistant_message["content"]:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_result = tool_manager.execute_tool(
                    tool_use["name"],
                    **tool_use["input"],
                )
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": tool_result}],
                    }
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Final call — toolConfig must be included because messages contain toolUse/toolResult blocks
        final_params = {
            **self.base_params,
            "system": base_params["system"],
            "messages": messages,
            "toolConfig": base_params["toolConfig"],
        }

        final_response = self.client.converse(**final_params)
        return self._extract_text(final_response)
