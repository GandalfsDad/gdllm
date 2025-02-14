from ...abstract import AbstractConfig
from ...abstract import AbstractToolProvider

from typing import List, Any, Optional
from abc import ABC, abstractmethod

class OpenAIConfig(AbstractConfig, ABC):
    provider: str = 'OpenAI'
    base_url: str = 'https://api.openai.com/v1'
    tools: List[Any] = []

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class OpenAIGPTConfig(OpenAIConfig):
    temperature: float = 0.7
    max_tokens: int = 1024
    tool_provider: Optional[AbstractToolProvider]

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if self.tool_provider:    
            args.update(self.tool_provider.parse_tools())
        
        return args
    
class OpenAIReasoningConfig(OpenAIConfig):
    reasoning_effort: str = 'medium'
    tool_provider: Optional[AbstractToolProvider]

    def get_call_args(self) -> dict:
        args = {
            "reasoning_effort": self.reasoning_effort
        }
        
        if self.tool_provider:    
            args.update(self.tool_provider.parse_tools())
        
        return args