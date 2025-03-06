from ...abstract import AbstractConfig
from .tool import GoogleToolProvider

from typing import List, Optional
from abc import ABC, abstractmethod

class GoogleConfig(AbstractConfig, ABC):
    provider: str = 'Google'
    api_key: str
    model: str
    tools: List[str] = []
    system_message: Optional[str] = None

    @abstractmethod
    def get_call_args(self) -> dict:
        pass

    @abstractmethod
    def tool_use_available(self) -> bool:
        pass

    @abstractmethod
    def structured_output_available(self) -> bool:
        pass


class GoogleGPTConfig(GoogleConfig):
    temperature: float = 0.5
    max_output_tokens: int = 1024

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens
        }

        if self.tools:
            args["tools"]= GoogleToolProvider.parse_tools(self.tools)
            args["automatic_function_calling"]= {"disable": True, "maximum_remote_calls": 0}
        
        if self.system_message:
            args["system_instruction"]= self.system_message

        return args
    
    def tool_use_available(self) -> bool:
        return True
    
    def structured_output_available(self) -> bool:
        return True

class GoogleReasoningConfig(GoogleConfig):

    def get_call_args(self) -> dict:
        args = {}

        if self.tools:
            args["automatic_function_calling"] = {"disable": True, "maximum_remote_calls": 0}

        if self.system_message:
            args["system_instruction"]= self.system_message

        return args
            
        
    def tool_use_available(self) -> bool:
        return False
    
    def structured_output_available(self) -> bool:
        return False