from ...abstract import AbstractConfig
from .tool import DeepSeekToolProvider

from typing import List, Optional
from abc import ABC, abstractmethod

class DeepSeekConfig(AbstractConfig, ABC):
    provider: str = 'DeepSeek'
    base_url: str = 'https://api.deepseek.com'
    api_key: str
    model: str
    tools: List[str] = []
    system_message: Optional[str] = None
    system_name: Optional[str] = None

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class DeepSeekGPTConfig(DeepSeekConfig):
    temperature: float = 0.7
    max_tokens: int = 1024
    system_name: Optional[str] = 'system'

    def get_call_args(self) -> dict:
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.tools:
            args['tools'] = DeepSeekToolProvider.parse_tools(self.tools)
        return args

class DeepSeekReasoningConfig(DeepSeekConfig):
    system_name: Optional[str] = 'system'

    def get_call_args(self) -> dict:
        args = {}
        return args