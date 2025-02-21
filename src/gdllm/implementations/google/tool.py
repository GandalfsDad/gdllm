from ...abstract import AbstractToolProvider
from ...util import use_tool, get_tools

from typing import Any, List
from openai import pydantic_function_tool
from google.genai import types as google_types

class GoogleToolProvider(AbstractToolProvider):

    @classmethod
    def use_tool(cls, tool_name: str, args: dict) -> Any:
        return use_tool(tool_name, args)

    @classmethod
    def parse_tools(cls, tools: List[str]) -> Any:
        tool_objects = get_tools(tools)
        oai_tools = [pydantic_function_tool(tool) for tool in tool_objects]

        google_tools = []
        for tool in oai_tools:
            google_version = google_types.FunctionDeclaration(
                name = tool['function']['name'],
                description=tool['function']['description'] if "description" in tool['function'] else None,
                parameters= google_types.Schema(
                    type = "OBJECT",
                    properties =  {
                        param:google_types.Schema(
                            type = value['type'].upper(), 
                            description=value['title']) for param, value in tool['function']['parameters']['properties'].items()},
                    required= tool['function']['parameters']['required']
                )
            )
            
            tool_object = google_types.Tool(function_declarations=[google_version])
            google_tools.append(tool_object)

        return google_tools