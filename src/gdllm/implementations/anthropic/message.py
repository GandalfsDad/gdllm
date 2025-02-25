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
        print("Role: " + self.role + "\nContent: " + self.message)

class AnthropicResponse(AbstractAnthropicMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", "content": self.response.text}
    
    def print(self):
        print("Role: assistant\nContent: " + self.response.text)

class AnthropicReasoningResponse(AbstractAnthropicMessage):
    def __init__(self, response):
        self.response = response

    def to_chat_message(self) -> dict:
        return {"role": "assistant", 
                "content": [{"type":"thinking", "thinking": content.thinking, "signature": content.signature} if content.type == "thinking" else 
                            {"type":"redacted_thinking", "data": content.data} if content.type == "redacted_thinking" else
                            {"type":"text", "text": content.text} for content in self.response.content]
                }
    
    def print(self):
        for content in self.response.content:
            if content.type == "text":
                print("Role: assistant\nContent: " + content.text)
            elif content.type == "thinking":
                print("Role: assistant\nThinking: " + content.thinking)
            elif content.type == "redacted_thinking":
                print("Role: assistant\nRedacted thinking!")
            else:
                raise Exception("Unknown content type")
    
class AnthropicToolResponse(AbstractAnthropicMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "assistant", 
                "content": 
                    [ {"type":"text", "text": content.text} if content.type == "text" else 
                     {"type":"thinking", "thinking": content.thinking, "signature": content.signature} if content.type == "thinking" else
                     {"type":"redacted_thinking", "data": content.data} if content.type == "redacted_thinking" else
                     {"type":"tool_use", "id": content.id, "name": content.name, "input": content.input} 
                     for content in self.response.content
                    ]
                }
    
    def print(self):
        for content in self.response.content:
            if content.type == "text":
                print("Role: assistant\nContent: " + content.text)
            elif content.type == "thinking":
                print("Role: assistant\nThinking: " + content.thinking)
            elif content.type == "redacted_thinking":
                print("Role: assistant\nRedacted thinking!")
            elif content.type == "tool_use":
                print("Role: assistant\nTool use: " + str(content))
            else:
                raise Exception("Unknown content type")

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
        print("Role: user\nTool result: " + str(self.result))