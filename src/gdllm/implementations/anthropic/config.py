from ...abstract import AbstractConfig
from .tool import AnthropicToolProvider

from typing import List, Optional
from abc import ABC, abstractmethod

class AnthropicConfig(AbstractConfig, ABC):
    provider: str = 'Anthropic'
    api_key: str
    model: str
    tools: List[str] = []
    system_message: Optional[str] = None

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class AnthropicGPTConfig(AnthropicConfig):
    temperature: float = 0.7
    max_tokens: int = 1024

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        if self.tools:
            args["tools"] =  AnthropicToolProvider.parse_tools(self.tools)
        
        return args
    
class AnthropicReasoningConfig(AnthropicConfig):
    max_tokens: int = 2046 # Must be larger than thinking_budget
    thinking_enabled: bool = True
    thinking_budget: int = 1024

    def get_call_args(self) -> dict:
        args = {
            "max_tokens": self.max_tokens,
            "thinking" : {
                "type": "enabled" if self.thinking_enabled else "disabled",
                **({"budget_tokens": self.thinking_budget} if self.thinking_enabled else {})
            }
        }

        if self.tools:
            args["tools"] =  AnthropicToolProvider.parse_tools(self.tools)
        
        return args
