import json
import unittest
from unittest.mock import patch, MagicMock

from src.models.schema import MessagesRequest, Message, Tool
from src.services.converter import convert_anthropic_to_litellm, convert_litellm_to_anthropic


class TestConverter(unittest.TestCase):
    """Test the converter functions"""

    def setUp(self):
        """Setup common test data"""
        # Simple test request with text only
        self.simple_request = MessagesRequest(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[
                Message(role="user", content="Hello, world!")
            ]
        )

        # Request with system prompt
        self.system_request = MessagesRequest(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            system="You are a helpful assistant.",
            messages=[
                Message(role="user", content="Tell me about Paris.")
            ]
        )

        # Request with tools
        self.calculator_tool = Tool(
            name="calculator",
            description="Evaluate mathematical expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        )

        self.tool_request = MessagesRequest(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[
                Message(role="user", content="Calculate 2+2")
            ],
            tools=[self.calculator_tool],
            tool_choice={"type": "auto"}
        )

    def test_simple_request_conversion(self):
        """Test conversion of a simple request"""
        litellm_request = convert_anthropic_to_litellm(self.simple_request)

        # Check basic structure
        self.assertEqual(litellm_request["model"], "claude-3-sonnet-20240229")
        self.assertEqual(litellm_request["max_tokens"], 300)
        self.assertEqual(litellm_request["temperature"], 1.0)

        # Check messages
        self.assertEqual(len(litellm_request["messages"]), 1)
        self.assertEqual(litellm_request["messages"][0]["role"], "user")
        self.assertEqual(litellm_request["messages"][0]["content"], "Hello, world!")

    def test_system_request_conversion(self):
        """Test conversion of a request with system prompt"""
        litellm_request = convert_anthropic_to_litellm(self.system_request)

        # Check system message is included
        self.assertEqual(len(litellm_request["messages"]), 2)
        self.assertEqual(litellm_request["messages"][0]["role"], "system")
        self.assertEqual(litellm_request["messages"][0]["content"], "You are a helpful assistant.")

        # Check user message
        self.assertEqual(litellm_request["messages"][1]["role"], "user")
        self.assertEqual(litellm_request["messages"][1]["content"], "Tell me about Paris.")

    def test_tool_request_conversion(self):
        """Test conversion of a request with tools"""
        litellm_request = convert_anthropic_to_litellm(self.tool_request)

        # Check tools are properly converted
        self.assertIn("tools", litellm_request)
        self.assertEqual(len(litellm_request["tools"]), 1)
        self.assertEqual(litellm_request["tools"][0]["type"], "function")
        self.assertEqual(litellm_request["tools"][0]["function"]["name"], "calculator")

        # Check tool_choice
        self.assertEqual(litellm_request["tool_choice"], "auto")

    def test_litellm_to_anthropic_text_response(self):
        """Test conversion of a text response from LiteLLM to Anthropic format"""
        # Mock LiteLLM response
        litellm_response = MagicMock()
        litellm_response.id = "resp_abc123"
        litellm_response.choices = [MagicMock()]
        litellm_response.choices[0].message.content = "Hello, this is a test response."
        litellm_response.choices[0].finish_reason = "stop"
        litellm_response.usage.prompt_tokens = 10
        litellm_response.usage.completion_tokens = 20

        anthropic_response = convert_litellm_to_anthropic(litellm_response, self.simple_request)

        # Check basic structure
        self.assertEqual(anthropic_response.id, "resp_abc123")
        self.assertEqual(anthropic_response.model, "claude-3-sonnet-20240229")
        self.assertEqual(anthropic_response.role, "assistant")
        self.assertEqual(anthropic_response.stop_reason, "end_turn")

        # Check content
        self.assertEqual(len(anthropic_response.content), 1)
        self.assertEqual(anthropic_response.content[0]["type"], "text")
        self.assertEqual(anthropic_response.content[0]["text"], "Hello, this is a test response.")

        # Check usage
        self.assertEqual(anthropic_response.usage.input_tokens, 10)
        self.assertEqual(anthropic_response.usage.output_tokens, 20)

    def test_litellm_to_anthropic_tool_response(self):
        """Test conversion of a tool use response from LiteLLM to Anthropic format"""
        # Mock LiteLLM response with tool calls
        litellm_response = MagicMock()
        litellm_response.id = "resp_tool123"
        litellm_response.choices = [MagicMock()]
        litellm_response.choices[0].message.content = "I'll calculate that for you."

        # Add tool_calls to the message
        tool_call = MagicMock()
        tool_call.id = "call_abc123"
        tool_call.function.name = "calculator"
        tool_call.function.arguments = json.dumps({"expression": "2+2"})
        litellm_response.choices[0].message.tool_calls = [tool_call]

        litellm_response.choices[0].finish_reason = "tool_calls"
        litellm_response.usage.prompt_tokens = 15
        litellm_response.usage.completion_tokens = 25

        # Set is_claude_model to True by manipulating the model name
        self.tool_request.model = "claude-3-sonnet-20240229"

        anthropic_response = convert_litellm_to_anthropic(litellm_response, self.tool_request)

        # Check for tool use in content
        self.assertTrue(any(block.get("type") == "tool_use" for block in anthropic_response.content))

        # Find the tool use block
        tool_use_block = next((block for block in anthropic_response.content if block.get("type") == "tool_use"), None)

        # Check tool use structure
        self.assertIsNotNone(tool_use_block)
        self.assertEqual(tool_use_block["name"], "calculator")
        self.assertEqual(tool_use_block["input"]["expression"], "2+2")

        # Check stop reason is tool_use
        self.assertEqual(anthropic_response.stop_reason, "tool_use")

    def test_error_handling(self):
        """Test error handling in conversion"""
        # Simulate a conversion error
        with patch('src.services.converter.logger') as mock_logger:
            # Create a response that will cause an error when accessing attributes
            bad_response = object()

            # This should not raise an exception but return an error message
            result = convert_litellm_to_anthropic(bad_response, self.simple_request)

            # Verify logger was called with error
            self.assertTrue(mock_logger.error.called)

            # Verify error response was created
            self.assertIn("Error converting response", result.content[0]["text"])


if __name__ == '__main__':
    unittest.main()
