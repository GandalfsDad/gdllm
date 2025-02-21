from ...abstract import AbstractToolProvider
from ...util import use_tool, get_tools

from typing import Any, List
from openai import pydantic_function_tool

class OpenAIToolProvider(AbstractToolProvider):

    @classmethod
    def use_tool(cls, tool_name: str, args: dict) -> Any:
        return use_tool(tool_name, args)

    @classmethod
    def parse_tools(cls, tools: List[str]) -> list:
        tool_objects = get_tools(tools)
        return [pydantic_function_tool(tool) for tool in tool_objects]