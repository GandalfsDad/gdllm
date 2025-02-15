from ...abstract import AbstractLLM, AbstractToolUser, AbstractStructuredOutputer
from .config import DeepSeekConfig
from .message import AbstractDeepSeekMessage, DeepSeekToolResponse, DeepSeekResponse, DeepSeekMessage, DeepSeekToolResultResponse
from .tool import DeepSeekToolProvider

import json
from typing import List, TypeVar, Any

from openai import OpenAI as BaseOpenAI

T = TypeVar('T')

class DeepSeek(AbstractLLM, AbstractToolUser):
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.client = BaseOpenAI(api_key=config.api_key, base_url=config.base_url)
    
    def get_chat_response(self, messages: List[AbstractDeepSeekMessage]) -> AbstractDeepSeekMessage:
        parsed_messages = [message.to_chat_message() for message in messages]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=parsed_messages,
            **self.config.get_call_args()
        )
        return self.process_response(response.choices[0])
    
    def process_response(self, response: Any) -> AbstractDeepSeekMessage:
        if response.finish_reason=='tool_calls':
            return DeepSeekToolResponse(response)
        else:
            return DeepSeekResponse(response)
    
    def format_user_message(self, message: str) -> Any:
        return DeepSeekMessage(role= "user", message= message)
    
    def process_tool_calls(self, tool_call_response: DeepSeekToolResponse) -> List[DeepSeekToolResultResponse]:
        results = []
        for tool_call in tool_call_response.response.message.tool_calls:
            func, args = tool_call.function.name, json.loads(tool_call.function.arguments)
            result = DeepSeekToolProvider.use_tool(func, args)

            results.append(DeepSeekToolResultResponse(tool_call.id, result))
        return results
    
    def check_tool_use(self, message: AbstractDeepSeekMessage) -> bool:
        return type(message) is DeepSeekToolResponse
