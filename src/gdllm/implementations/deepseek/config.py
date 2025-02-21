from ...abstract import AbstractConfig
from .tool import DeepSeekToolProvider

from typing import List
from abc import ABC, abstractmethod

class DeepSeekConfig(AbstractConfig, ABC):
    provider: str = 'DeepSeek'
    base_url: str = 'https://api.deepseek.com'
    api_key: str
    model: str
    tools: List[str] = []

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class DeepSeekGPTConfig(DeepSeekConfig):
    temperature: float = 0.7
    max_tokens: int = 1024

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.tools:
            args['tools'] = DeepSeekToolProvider.parse_tools(self.tools)
        return args

class DeepSeekReasoningConfig(DeepSeekConfig):

    def get_call_args(self) -> dict:
        return {
        }