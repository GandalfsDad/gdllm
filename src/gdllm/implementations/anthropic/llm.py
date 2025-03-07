from ...abstract import AbstractLLM, AbstractMessage, AbstractToolUser, AbstractTokenCounter
from .config import AnthropicConfig, AnthropicReasoningConfig
from .message import AbstractAnthropicMessage, AnthropicResponse, AnthropicMessage, AnthropicToolResponse, AnthropicToolResultResponse, AnthropicReasoningResponse
from .tool import AnthropicToolProvider

import json
from typing import List, TypeVar, Any

import anthropic

T = TypeVar('T')

class Anthropic(AbstractLLM, AbstractToolUser, AbstractTokenCounter):
    def __init__(self, config: AnthropicConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)

        self.__tokens = {
            "input": 0,
            "output": 0,
            "cached_read": 0,
            "cached_write": 0
        }
    
    def get_chat_response(self, messages: List[AbstractAnthropicMessage]) -> Any:
        parsed_messages = [message.to_chat_message() for message in messages]

        response = self.client.messages.create(
            model=self.config.model,
            **({"system": self.config.system_message} if self.config.system_message else {}),
            messages=parsed_messages,
            **self.config.get_call_args()
        )

        self.update_token_usage(response.usage)
        
        return self.process_response(response)
    
    def process_response(self, response: Any) -> AbstractAnthropicMessage:
        if response.stop_reason=='tool_use':
            return AnthropicToolResponse(response)
        elif self.config is AnthropicReasoningConfig:
            return AnthropicReasoningResponse(response)
        else:
            return AnthropicResponse(response.content[0])
        
    def new_conversation(self) -> List[AbstractAnthropicMessage]:
        return []
    
    def format_user_message(self, message: str) -> Any:
        return AnthropicMessage(role="user", message= message)
    
    def process_tool_calls(self, tool_call_response: AbstractAnthropicMessage) -> List[AbstractAnthropicMessage]:
        results = []

        for content in tool_call_response.response.content:
            if content.type == 'tool_use':
                func, args = content.name, content.input
                result = AnthropicToolProvider.use_tool(func, args)
                results.append(AnthropicToolResultResponse(content.id, result))

        return results
    
    def check_tool_use(self, message: AbstractAnthropicMessage) -> bool:
        return type(message) is AnthropicToolResponse
    
    def update_token_usage(self, usage):
        self.__tokens['input'] += usage.input_tokens
        self.__tokens['output'] += usage.output_tokens
        self.__tokens['cached_read'] += usage.cache_read_input_tokens
        self.__tokens['cached_write'] += usage.cache_creation_input_tokens

    def token_usage(self) -> dict:
        return self.__tokens
    