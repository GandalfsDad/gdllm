import unittest
from unittest.mock import MagicMock, patch
import sys
from io import StringIO

from src.gdllm.abstract.llm import AbstractLLM, AbstractToolUser
from src.gdllm.abstract.message import AbstractMessage
from src.gdllm.abstract.tool import AbstractTool
from src.gdllm.util.conversation import Conversation
from src.gdllm.util.tool_registry import register_tool, get_tools


class MockMessage(AbstractMessage):
    def __init__(self, content, role):
        self.content = content
        self.role = role
        
    def to_chat_message(self):
        return {"role": self.role, "content": self.content}
    
    def print(self):
        print(f"Role: {self.role}\nContent: {self.content}")


class MockTool(AbstractTool):
    name:str  = "mock_tool"
    description: str = "A mock tool for testing"
    
    def tool_call(self):
        return {"result": f"Tool called with {self.param1} and {self.param2}"}


class MockLLM(AbstractLLM):
    def __init__(self):
        self.messages_history = []
    
    def new_conversation(self):
        return []
    
    def format_user_message(self, message):
        return MockMessage(message, "user")
    
    def get_chat_response(self, messages):
        # Simple logic to create a response based on the latest message
        last_message = messages[-1]
        if isinstance(last_message, MockMessage):
            content = f"Response to: {last_message.content}"
            return MockMessage(content, "assistant")
        return MockMessage("Default response", "assistant")


class MockToolUserLLM(MockLLM, AbstractToolUser):
    def __init__(self, should_use_tools=False):
        super().__init__()
        self.should_use_tools = should_use_tools
        self.tool_call_count = 0
        self.max_tool_calls = 2  # To prevent infinite loops in tests
        # Register the mock tool for testing
        register_tool(MockTool)
    
    def check_tool_use(self, message):
        # For testing, we alternate between tool use and regular responses
        if self.should_use_tools and self.tool_call_count < self.max_tool_calls:
            self.tool_call_count += 1
            return True
        return False
    
    def process_tool_calls(self, message):
        # Create a tool response with a standardized format
        tool_params = {"param1": "test_value", "param2": 42}
        tool_result = {"result": f"Tool called with {tool_params['param1']} and {tool_params['param2']}"}
        
        # In a real implementation, this would use the tool_registry
        return [MockMessage(str(tool_result), "tool")]
    
    def get_chat_response(self, messages):
        if self.check_tool_use(messages[-1]) and self.tool_call_count == 1:
            return MockMessage("I need to use a tool", "assistant")
        elif self.tool_call_count >= 1:
            return MockMessage("Tool has been used", "assistant")
        return super().get_chat_response(messages)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.regular_llm = MockLLM()
        self.tool_user_llm = MockToolUserLLM()
        
        # Capture stdout for testing print outputs
        self.stdout = StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout
    
    def tearDown(self):
        # Restore stdout
        sys.stdout = self.original_stdout
    
    def test_conversation_initialization(self):
        conversation = Conversation(self.regular_llm)
        self.assertEqual(conversation.llm, self.regular_llm)
        self.assertFalse(conversation.output_user_input)
        self.assertTrue(conversation.output_system_response)
        self.assertFalse(conversation.output_tool_use)
        self.assertEqual(conversation.messages, [])
    
    def test_chat_without_tools(self):
        conversation = Conversation(self.regular_llm)
        conversation.chat("Hello")
        
        # Check that messages were properly added to history
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[0].content, "Hello")
        self.assertEqual(conversation.messages[0].role, "user")
        self.assertEqual(conversation.messages[1].content, "Response to: Hello")
        self.assertEqual(conversation.messages[1].role, "assistant")
        
        # Check output was printed (the assistant's response)
        self.assertIn("Response to: Hello", self.stdout.getvalue())
    
    def test_chat_no_output(self):
        conversation = Conversation(self.regular_llm, output_system_response=False)
        conversation.chat("Hello")
        
        # Check that response was not printed
        self.assertEqual(self.stdout.getvalue(), "")
    
    def test_chat_with_user_output(self):
        conversation = Conversation(self.regular_llm, output_user_input=True)
        conversation.chat("Hello")
        
        # Check that both user message and response were printed
        output = self.stdout.getvalue()
        self.assertIn("Role: user", output)
        self.assertIn("Content: Hello", output)
        self.assertIn("Response to: Hello", output)
    
    def test_check_tool_use(self):
        # Regular LLM without tool use capability
        conversation = Conversation(self.regular_llm)
        conversation.messages = [MockMessage("Test", "assistant")]
        self.assertFalse(conversation.check_tool_use())
        
        # LLM with tool use capability but not using tools
        tool_conversation = Conversation(self.tool_user_llm)
        tool_conversation.messages = [MockMessage("Test", "assistant")]
        self.assertFalse(tool_conversation.check_tool_use())
        
        # LLM with tool use capability and using tools
        tool_llm = MockToolUserLLM(should_use_tools=True)
        tool_conversation = Conversation(tool_llm)
        tool_conversation.messages = [MockMessage("Test", "assistant")]
        self.assertTrue(tool_conversation.check_tool_use())
    
    def test_print_chat(self):
        conversation = Conversation(self.regular_llm)
        conversation.messages = [
            MockMessage("Hello", "user"),
            MockMessage("Hi there!", "assistant")
        ]
        
        conversation.print_chat()
        
        output = self.stdout.getvalue()
        self.assertIn("Role: user", output)
        self.assertIn("Content: Hello", output)
        self.assertIn("Role: assistant", output)
        self.assertIn("Content: Hi there!", output)   

if __name__ == '__main__':
    unittest.main()