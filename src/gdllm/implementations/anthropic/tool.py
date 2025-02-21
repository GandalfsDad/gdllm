from ...abstract import AbstractToolProvider
from ...util import use_tool, get_tools

from typing import Any, List
from openai import pydantic_function_tool

class AnthropicToolProvider(AbstractToolProvider):

    @classmethod
    def use_tool(cls, tool_name: str, args: dict) -> Any:
        return use_tool(tool_name, args)

    @classmethod
    def parse_tools(cls, tools: List[str]) -> list:
        tool_objects = get_tools(tools)
        oai_tools = [pydantic_function_tool(tool) for tool in tool_objects]

        anthropic_tools = []
        for tool in oai_tools:
            anthropic_version = {
                "name": tool['function']['name'],
                **({"description": tool['function']['description']} if "description" in tool['function'] else {}),
                "input_schema": {
                    "type": "object",
                    "properties": 
                    {param: {"type": value['type'], "description": value['title']} for param, value in tool['function']['parameters']['properties'].items()},
                    "required": tool['function']['parameters']['required']
                }
            }
            anthropic_tools.append(anthropic_version)
    
        return anthropic_tools