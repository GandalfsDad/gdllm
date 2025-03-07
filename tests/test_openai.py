import unittest
from unittest.mock import MagicMock, patch
import json
import openai
from pydantic import BaseModel

from src.gdllm.implementations.openai.config import OpenAIGPTConfig, OpenAIReasoningConfig
from src.gdllm.implementations.openai.message import OpenAIMessage, OpenAIResponse, OpenAIToolResponse, OpenAIToolResultResponse
from src.gdllm.implementations.openai.llm import OpenAI
from src.gdllm.implementations.openai.tool import OpenAIToolProvider
from src.gdllm.abstract.tool import AbstractTool

class TestOpenAIConfig(unittest.TestCase):
    def test_gpt_config(self):
        config = OpenAIGPTConfig(
            api_key="test_key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=100
        )
        
        self.assertEqual(config.provider, "OpenAI")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.model, "gpt-4")
        
        args = config.get_call_args()
        self.assertEqual(args["temperature"], 0.5)
        self.assertEqual(args["max_tokens"], 100)
    
    def test_reasoning_config(self):
        config = OpenAIReasoningConfig(
            api_key="test_key",
            model="gpt-4",
            reasoning_effort="high"
        )
        
        args = config.get_call_args()
        self.assertEqual(args["reasoning_effort"], "high")

class TestOpenAIMessage(unittest.TestCase):
    def test_user_message(self):
        message = OpenAIMessage("Hello", "user")
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
    
    def test_openai_response(self):
        mock_response = MagicMock()
        mock_response.message.content = "Test response"
        response = OpenAIResponse(mock_response)
        
        chat_message = response.to_chat_message()
        self.assertEqual(chat_message["role"], "assistant")
        self.assertEqual(chat_message["content"], "Test response")
    
    def test_tool_response(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        mock_response = MagicMock()
        mock_response.message.tool_calls = [mock_tool_call]
        
        response = OpenAIToolResponse(mock_response)
        chat_message = response.to_chat_message()
        
        self.assertEqual(chat_message["role"], "assistant")
        self.assertEqual(len(chat_message["tool_calls"]), 1)
        self.assertEqual(chat_message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(chat_message["tool_calls"][0]["function"]["name"], "test_tool")
    
    def test_tool_result_response(self):
        response = OpenAIToolResultResponse("call_123", {"result": "success"})
        chat_message = response.to_chat_message()
        
        self.assertEqual(chat_message["role"], "tool")
        self.assertEqual(chat_message["tool_call_id"], "call_123")
        self.assertEqual(chat_message["content"], '{"result": "success"}')

class TestOpenAILLM(unittest.TestCase):
    def setUp(self):
        self.config = OpenAIGPTConfig(
            api_key="test_key",
            model="gpt-4"
        )
        self.llm = OpenAI(self.config)

    def test_get_chat_response(self):
        fake_message = MagicMock()
        fake_message.content = "Mocked response from OpenAI"
        fake_choice = MagicMock(message=fake_message)
        fake_response = MagicMock(choices=[fake_choice])
        
        # Patch the 'create' method on the completions attribute of your client.
        with patch.object(self.llm.client.chat.completions, 'create', return_value=fake_response) as mock_create:
            # Execute the test
            prompt="Hello"
            messages = [OpenAIMessage(prompt, "user")]
            response = self.llm.get_chat_response(messages)
            
            # Assertions
            self.assertIsInstance(response, OpenAIResponse)
            self.assertEqual(response.to_chat_message()['content'], "Mocked response from OpenAI")

            mock_create.assert_called_once_with(
                model=self.llm.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm.config.max_tokens,
                temperature=self.llm.config.temperature
            )
        
        
    def test_format_user_message(self):
        message = self.llm.format_user_message("Hello")
        self.assertIsInstance(message, OpenAIMessage)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.message, "Hello")

    @patch('src.gdllm.implementations.openai.tool.OpenAIToolProvider.use_tool')
    def test_process_tool_calls(self, mock_use_tool):
        mock_use_tool.return_value = {"result": "success"}
        
        # Create a proper mock response object
        mock_tool_call = MagicMock()
        mock_tool_call.id = 'call_123'
        mock_tool_call.function.name = 'test_tool'
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        
        mock_response = MagicMock()
        mock_response.message = mock_message
        
        tool_response = OpenAIToolResponse(mock_response)
        
        results = self.llm.process_tool_calls(tool_response)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], OpenAIToolResultResponse)
        self.assertEqual(results[0].id, 'call_123')
        
        mock_use_tool.assert_called_once_with('test_tool', {'arg': 'value'})

    def test_check_tool_use(self):
        regular_response = OpenAIResponse(MagicMock())
        tool_response = OpenAIToolResponse(MagicMock())
        
        self.assertFalse(self.llm.check_tool_use(regular_response))
        self.assertTrue(self.llm.check_tool_use(tool_response))

    def test_structured_output(self):
        class TestOutput(BaseModel):
            name: str
            age: int
        
        mock_parsed = TestOutput(name="Test", age=25)
        fake_message = MagicMock(parsed=mock_parsed)
        fake_choice = MagicMock(message=fake_message)
        fake_response = MagicMock(choices=[fake_choice], usage=MagicMock(prompt_tokens=0, completion_tokens=0))

        # Patch the 'parse' method on the completions attribute of your client
        with patch.object(self.llm.client.beta.chat.completions, 'parse', return_value=fake_response) as mock_parse:
            # Execute the test
            prompt = "Get me a person"
            result = self.llm.structured_output(prompt, TestOutput)
            
            # Assertions
            self.assertEqual(result.name, "Test")
            self.assertEqual(result.age, 25)
            
            mock_parse.assert_called_once_with(
                model=self.llm.config.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=TestOutput,
                temperature=self.llm.config.temperature,
                max_tokens=self.llm.config.max_tokens
            )

class TestOpenAIToolProvider(unittest.TestCase):
    @patch('src.gdllm.implementations.openai.tool.use_tool')
    def test_use_tool(self, mock_use_tool):
        mock_use_tool.return_value = {"result": "success"}
        
        result = OpenAIToolProvider.use_tool("test_tool", {"arg": "value"})
        self.assertEqual(result, {"result": "success"})
        mock_use_tool.assert_called_once_with("test_tool", {"arg": "value"})

    @patch('src.gdllm.implementations.openai.tool.get_tools')
    @patch('src.gdllm.implementations.openai.tool.pydantic_function_tool')
    def test_parse_tools(self, mock_pydantic_function_tool, mock_get_tools):
        class TestTool(AbstractTool):
            name: str
            description: str
            
            def tool_call(self):
                return {"result": "test"}
            
        test_tool = TestTool(name="test_tool", description="A test tool")
        mock_get_tools.return_value = [test_tool]
        mock_pydantic_function_tool.return_value = {"name": "test_tool", "description": "A test tool"}
        
        tools = OpenAIToolProvider.parse_tools(["test_tool"])
        
        self.assertEqual(len(tools), 1)
        mock_get_tools.assert_called_once_with(["test_tool"])
        mock_pydantic_function_tool.assert_called_once_with(test_tool)
        self.assertEqual(tools[0], {"name": "test_tool", "description": "A test tool"})

if __name__ == '__main__':
    unittest.main()