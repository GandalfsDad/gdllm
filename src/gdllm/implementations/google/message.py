import json
from abc import ABC, abstractmethod

from ...abstract import AbstractMessage

from google.genai import types as google_types

class AbstractGoogleMessage(AbstractMessage, ABC):
    @abstractmethod
    def to_chat_message(self) -> dict:
        pass

class GoogleMessage(AbstractGoogleMessage):
    def __init__(self, message, role):
        self.message = message
        self.role = role
    
    def to_chat_message(self) -> dict:
        return google_types.Content(role= self.role, parts=[google_types.Part(text=self.message)])
    
    def print(self):
        print("Role: " + self.role + "\nParts: " + self.message)

class GoogleResponse(AbstractGoogleMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return google_types.Content(role= 'model', parts=[google_types.Part(text=self.response.content.parts[0].text)])
    
    def print(self):
        print("Role: model\nParts: " + self.response.content.parts[0].text)
    
class GoogleToolResponse(AbstractGoogleMessage):
    def __init__(self, response):
        self.response = response
    
    def to_chat_message(self) -> dict:
        return google_types.Content(role= 'model', parts=[google_types.Part(function_call=self.response.content.parts[0].function_call)])

    def print(self):
        print("Role: model\nParts: " + str(self.response.content.parts[0].function_call))
    
class GoogleToolResultResponse(AbstractGoogleMessage):
    def __init__(self, func, result):
        self.func = func
        self.result = result
    
    def to_chat_message(self) -> dict:
        return google_types.Content(role= 'model',parts=[google_types.Part(function_response=google_types.FunctionResponse(name=self.func, response={"result":self.result}))])
    
    def print(self):
        print("Role: model\nFunction: " + self.func + "\nResult: " + str(self.result))