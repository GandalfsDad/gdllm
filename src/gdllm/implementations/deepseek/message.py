import json
from abc import ABC, abstractmethod

from ...abstract import AbstractMessage

class AbstractDeepSeekMessage(AbstractMessage, ABC):
    @abstractmethod
    def to_chat_message(self) -> dict:
        pass

class DeepSeekMessage(AbstractDeepSeekMessage):
    def __init__(self, message, role):
        self.message = message
        self.role = role
    
    def to_chat_message(self) -> dict:
        return {"role": self.role, "content": self.message}
    
    def print(self):
        print("Role: " + self.role + "\nContent: " + self.message)

class DeepSeekResponse(AbstractDeepSeekMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", "content": self.response.message.content}
    
    def print(self):
        print("Role: assistant\nContent: " + self.response.message.content)

class DeepSeekToolResponse(AbstractDeepSeekMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", 
                "tool_calls": self.response.message.tool_calls,
                "type" : 'function'}
    
    def print(self):
        print("Role: assistant\nTool calls: " + str(self.response.message.tool_calls))
    
class DeepSeekToolResultResponse(AbstractDeepSeekMessage):
    def __init__(self, id, result):
        self.id = id
        self.result = result
    
    def to_chat_message(self) -> dict:
        return {"role": "tool", 
                "tool_call_id": self.id,
                "content": json.dumps(self.result)}
    
    def print(self):
        print("Role: tool\nTool call ID: " + self.id + "\nContent: " + json.dumps(self.result))