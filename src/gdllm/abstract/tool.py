from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class AbstractToolProvider(ABC):
    
    @classmethod
    @abstractmethod    
    def use_tool(cls, tool_name: str, args: dict) -> Any:
        pass

    @classmethod
    @abstractmethod
    def parse_tools(cls, tools: list) -> list:
        pass

class AbstractTool(ABC, BaseModel):
    @abstractmethod
    def tool_call(self) -> Any:
        pass