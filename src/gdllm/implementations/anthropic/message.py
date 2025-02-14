import json
from abc import ABC, abstractmethod

from ...abstract import AbstractMessage

class AbstractAnthropicMessage(AbstractMessage, ABC):
    @abstractmethod
    def to_chat_message(self) -> dict:
        pass

class AnthropicMessage(AbstractAnthropicMessage):
    def __init__(self, message, role):
        self.message = message
        self.role = role
    
    def to_chat_message(self) -> dict:
        return {"role": self.role, "content": self.message}
    
    def print(self):
        return "Role: " + self.role + "\nContent: " + self.message

class AnthropicResponse(AbstractAnthropicMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", "content": self.response.text}
    
    def print(self):
        return "Role: assistant\nContent: " + self.response.text
    
class AnthropicToolResponse(AbstractAnthropicMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", 
                "content": [
                    {
                        "type":"text",
                        "text": self.response.content[0].text
                    },
                    {
                        "type":"tool_use",
                        "id": self.response.content[1].id,
                        "name": self.response.content[1].name,
                        "input": self.response.content[1].input
                    }
                ]
                }
    
    def print(self):
        return print("Role: assistant\nTool use: " + str(self.response.content[1]))

class AnthropicToolResultResponse(AbstractAnthropicMessage):
    def __init__(self, id, result):
        self.id = id
        self.result = result
    
    def to_chat_message(self) -> dict:
        return {"role": "user", 
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": self.id,
                    "content":json.dumps(self.result)
                }]
                }
    
    def print(self):
        return print("Role: user\nTool result: " + str(self.result))