import json
from abc import ABC, abstractmethod

from ...abstract import AbstractMessage

class AbstractGoogleMessage(AbstractMessage, ABC):
    @abstractmethod
    def to_chat_message(self) -> dict:
        pass

class GoogleMessage(AbstractGoogleMessage):
    def __init__(self, message, role):
        self.message = message
        self.role = role
    
    def to_chat_message(self) -> dict:
        return {"role": self.role, "parts": [self.message]}
    
    def print(self):
        return "Role: " + self.role + "\Parts: " + self.message

class GoogleResponse(AbstractGoogleMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "model", "parts": self.response.content.parts}
    
    def print(self):
        return "Role: model\nParts: " + str(self.response.content.parts)
    
class GoogleToolResponse(AbstractGoogleMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return {"role": "model", 
                "parts": self.response.content.parts,
                }
    
    def print(self):
        return "Role: model\nParts: " + str(self.response.content.parts)
    
class GoogleToolResultResponse(AbstractGoogleMessage):
    def __init__(self, func, result):
        self.func = func
        self.result = result
    
    def to_chat_message(self) -> dict:
        return {
            "role": "model",
            "parts": [{
                "function_response": {
                    "name": self.func,
                    "response": {"result":self.result}
                }
            }]
        }
    
    def print(self):
        return "Role: model\nFunction: " + self.func + "\nResult: " + str(self.result)