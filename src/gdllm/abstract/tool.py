from abc import ABC, abstractmethod
from typing import Any

class AbstractToolProvider(ABC):
    @abstractmethod
    def use_tool(self, tool_name: str, args: dict) -> Any:
        pass

    @abstractmethod
    def parse_tools(self, tools: list) -> list:
        pass

class AbstractTool(ABC):
    @abstractmethod
    def tool_call(self) -> Any:
        pass