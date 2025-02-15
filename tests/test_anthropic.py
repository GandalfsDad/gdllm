import unittest
from unittest.mock import MagicMock, patch
import json
import anthropic
from src.gdllm.implementations.anthropic.config import AnthropicGPTConfig
from src.gdllm.implementations.anthropic.message import AnthropicMessage, AnthropicResponse, AnthropicToolResponse, AnthropicToolResultResponse
from src.gdllm.implementations.anthropic.llm import Anthropic
from src.gdllm.implementations.anthropic.tool import AnthropicToolProvider
from src.gdllm.abstract.tool import AbstractTool
from src.gdllm.util.tool_registry import register_tool

class TestAnthropicConfig(unittest.TestCase):
    def test_config(self):
        config = AnthropicGPTConfig(
            api_key="test_key",
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=100
        )
        
        self.assertEqual(config.provider, "Anthropic")
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.model, "claude-3-opus-20240229")
        
        args = config.get_call_args()
        self.assertEqual(args["temperature"], 0.5)
        self.assertEqual(args["max_tokens"], 100)

class TestAnthropicMessage(unittest.TestCase):
    def test_user_message(self):
        message = AnthropicMessage("Hello", "user")
        chat_message = message.to_chat_message()
        
        self.assertEqual(chat_message["role"], "user")
        self.assertEqual(chat_message["content"], "Hello")
        
        # Test print() by capturing stdout
        from io import StringIO
        import sys
        stdout = StringIO()
        sys.stdout = stdout
        message.print()
        sys.stdout = sys.__stdout__
        self.assertEqual(stdout.getvalue().strip(), "Role: user\nContent: Hello")
    
    def test_anthropic_response(self):
        mock_content = MagicMock()
        mock_content.text = "Test response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        response = AnthropicResponse(mock_content)
        
        chat_message = response.to_chat_message()
        self.assertEqual(chat_message["role"], "assistant")
        self.assertEqual(chat_message["content"], mock_content.text)
    
    def test_tool_response(self):
        mock_response = MagicMock()
        mock_content_text = MagicMock()
        mock_content_text.type = "text"
        mock_content_text.text = "Thinking about using a tool"
        
        mock_content_tool = MagicMock()
        mock_content_tool.type = "tool_use"
        mock_content_tool.id = "call_123"
        mock_content_tool.name = "test_tool"
        mock_content_tool.input = {"arg": "value"}
        
        mock_response.content = [mock_content_text, mock_content_tool]
        
        response = AnthropicToolResponse(mock_response)
        chat_message = response.to_chat_message()
        
        self.assertEqual(chat_message["role"], "assistant")
        self.assertEqual(len(chat_message["content"]), 2)
        self.assertEqual(chat_message["content"][0]["type"], "text")
        self.assertEqual(chat_message["content"][0]["text"], "Thinking about using a tool")
        self.assertEqual(chat_message["content"][1]["type"], "tool_use")
        self.assertEqual(chat_message["content"][1]["id"], "call_123")
    
    def test_tool_result_response(self):
        response = AnthropicToolResultResponse("call_123", {"result": "success"})
        chat_message = response.to_chat_message()
        
        self.assertEqual(chat_message["role"], "user")
        self.assertEqual(chat_message["content"][0]["type"], "tool_result")
        self.assertEqual(chat_message["content"][0]["tool_use_id"], "call_123")
        self.assertEqual(chat_message["content"][0]["content"], '{"result": "success"}')

class TestAnthropicLLM(unittest.TestCase):
    def setUp(self):
        self.config = AnthropicGPTConfig(
            api_key="test_key",
            model="claude-3-opus-20240229"
        )
        self.llm = Anthropic(self.config)

    def test_get_chat_response(self):
        mock_content = MagicMock()
        mock_content.text = "Mocked response from Anthropic"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        
        with patch.object(self.llm.client.messages, 'create', return_value=mock_response) as mock_create:
            messages = [AnthropicMessage("Hello", "user")]
            response = self.llm.get_chat_response(messages)
            
            self.assertIsInstance(response, AnthropicResponse)
            self.assertEqual(response.response.text, "Mocked response from Anthropic")
            mock_create.assert_called_once()
    
    def test_format_user_message(self):
        result = self.llm.format_user_message("Hello")
        self.assertEqual(result.role, "user")
        self.assertEqual(result.message, "Hello")

    def test_process_tool_calls(self):
        # Create a proper mock response object
        mock_content_text = MagicMock()
        mock_content_text.type = "text"
        mock_content_text.text = "Thinking about using a tool"
        
        mock_content_tool = MagicMock()
        mock_content_tool.type = "tool_use"
        mock_content_tool.id = "call_123"
        mock_content_tool.name = "test_tool"
        mock_content_tool.input = {"arg": "value"}
        
        mock_response = MagicMock()
        mock_response.content = [mock_content_text, mock_content_tool]
        
        tool_response = AnthropicToolResponse(mock_response)
        
        with patch('src.gdllm.implementations.anthropic.tool.AnthropicToolProvider.use_tool') as mock_use_tool:
            mock_use_tool.return_value = {"result": "success"}
            results = self.llm.process_tool_calls(tool_response)
            
            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], AnthropicToolResultResponse)
            self.assertEqual(results[0].id, 'call_123')
            mock_use_tool.assert_called_once_with('test_tool', {'arg': 'value'})

    def test_check_tool_use(self):
        regular_response = AnthropicResponse(MagicMock())
        tool_response = AnthropicToolResponse(MagicMock())
        
        self.assertFalse(self.llm.check_tool_use(regular_response))
        self.assertTrue(self.llm.check_tool_use(tool_response))

class TestAnthropicToolProvider(unittest.TestCase):
    @patch('src.gdllm.implementations.anthropic.tool.use_tool')
    def test_use_tool(self, mock_use_tool):
        mock_use_tool.return_value = {"result": "success"}
        
        result = AnthropicToolProvider.use_tool("test_tool", {"arg": "value"})
        self.assertEqual(result, {"result": "success"})
        mock_use_tool.assert_called_once_with("test_tool", {"arg": "value"})

    def test_parse_tools(self):
        class TestTool(AbstractTool):
            def __init__(self, name):
                self.name = name
            def use(self, args):
                return {"result": "success"}

        register_tool(TestTool)
        
        tools = AnthropicToolProvider.parse_tools(["TestTool"])
        print(tools)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]['name'], 'TestTool')

if __name__ == '__main__':
    unittest.main()