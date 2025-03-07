from ...abstract import AbstractConfig
from .tool import OpenAIToolProvider

from typing import List, Optional
from abc import ABC, abstractmethod

class OpenAIConfig(AbstractConfig, ABC):
    provider: str = 'OpenAI'
    base_url: str = 'https://api.openai.com/v1'
    api_key: str
    model: str
    tools: List[str] = []
    system_message: Optional[str] = None
    system_name: str = 'system'

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class OpenAIGPTConfig(OpenAIConfig):
    temperature: float = 0.7
    max_tokens: int = 1024
    system_name: str = 'system'

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if self.tools:    
            args['tools'] = OpenAIToolProvider.parse_tools(self.tools)
        
        return args
    
class OpenAIReasoningConfig(OpenAIConfig):
    reasoning_effort: str = 'medium'
    system_name: str = 'developer'

    def get_call_args(self) -> dict:
        args = {
            "reasoning_effort": self.reasoning_effort
        }
        
        if self.tools:    
            args['tools'] = OpenAIToolProvider.parse_tools(self.tools)
        
        return args