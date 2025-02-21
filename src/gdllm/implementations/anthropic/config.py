from ...abstract import AbstractConfig
from .tool import AnthropicToolProvider

from typing import List
from abc import ABC, abstractmethod

class AnthropicConfig(AbstractConfig, ABC):
    provider: str = 'Anthropic'
    api_key: str
    model: str
    tools: List[str] = []

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
