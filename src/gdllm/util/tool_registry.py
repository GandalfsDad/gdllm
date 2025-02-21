from typing import Dict, List

from ..abstract.tool import AbstractTool

class ToolRegistry(Dict[str, AbstractTool]):

    def register_tool(self, tool: AbstractTool):
        self[tool.__name__] = tool

    def register_tools(self, tools: List[AbstractTool]):
        for tool in tools:
            self.register_tool(tool)

TOOL_REGISTRY: ToolRegistry = ToolRegistry()

def register_tool(tool: AbstractTool) -> None:
    TOOL_REGISTRY.register_tool(tool)

def register_tools(tools: List[AbstractTool]) -> None:
    TOOL_REGISTRY.register_tools(tools)

def get_tools(tools: List[str] = None) -> List[AbstractTool]:
    if not tools:
        return [tool for _, tool in TOOL_REGISTRY.items()]
    else:
        return [tool for name, tool in TOOL_REGISTRY.items() if name in tools]
    
def use_tool(func, args):
    if (tool_class := TOOL_REGISTRY.get(func)):
        tool_instance = tool_class(**args)
        tool_response = tool_instance.tool_call()

        return tool_response
    else:
        raise ValueError(f"Unregistered tool function: {func}")