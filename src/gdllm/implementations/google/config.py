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

        return args

class GoogleReasoningConfig(GoogleConfig):

    def get_call_args(self) -> dict:
        if self.tools:
            return {
                "tools": GoogleToolProvider.parse_tools(self.tools),
                "automatic_function_calling": {"disable": True, "maximum_remote_calls": 0}
            }