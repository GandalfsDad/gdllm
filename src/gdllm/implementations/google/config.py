from ...abstract import AbstractConfig
from .tool import GoogleToolProvider

from typing import List
from abc import ABC, abstractmethod

class GoogleConfig(AbstractConfig, ABC):
    provider: str = 'Google'
    api_key: str
    model: str
    tools: List[str] = []

    @abstractmethod
    def get_call_args(self) -> dict:
        pass


class GoogleGPTConfig(GoogleConfig):

    def get_call_args(self) -> dict:
        if self.tools:
            return {
                "tools": GoogleToolProvider.parse_tools(self.tools),
                "automatic_function_calling": {"disable": True, "maximum_remote_calls": 0}
            }
        else:
            return {}

class GoogleAIReasoningConfig(GoogleConfig):

    def get_call_args(self) -> dict:
        if self.tools:
            return {
                "tools": GoogleToolProvider.parse_tools(self.tools),
                "automatic_function_calling": {"disable": True, "maximum_remote_calls": 0}
            }