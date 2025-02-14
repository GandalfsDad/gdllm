from ...abstract import AbstractLLM, AbstractMessage, AbstractToolUser, AbstractStructuredOutputer
from .config import AnthropicConfig
from .message import AbstractAnthropicMessage, AnthropicResponse, AnthropicMessage, AnthropicToolResponse, AnthropicToolResultResponse
from .tool import AnthropicToolProvider

import json
from typing import List, TypeVar, Any

import anthropic

T = TypeVar('T')

class Anthropic(AbstractLLM, AbstractToolUser):
    def __init__(self, config: AnthropicConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
    
    def get_chat_response(self, messages: List[AbstractAnthropicMessage]) -> Any:
        parsed_messages = [message.to_chat_message() for message in messages]

        response = self.client.messages.create(
            model=self.config.model,
            messages=parsed_messages,
            **self.config.get_call_args()
        )
        
        return self.process_response(response)
    
    def process_response(self, response: Any) -> AbstractAnthropicMessage:
        if response.stop_reason=='tool_use':
            return AnthropicToolResponse(response)
        else:
            return AnthropicResponse(response.content[0])
    
    def format_user_message(self, message: str) -> Any:
        return {"role": "user", "content": message}
    
    def process_tool_calls(self, tool_call_response: AbstractAnthropicMessage) -> List[AbstractAnthropicMessage]:
        results = []

        for content in tool_call_response['content']:
            if content['type'] == 'tool_use':
                func, args = content['name'], content['input']
                result = AnthropicToolProvider.use_tool(func, args)
                results.append(AnthropicToolResultResponse(content['id'], result))

        return results
    
    def check_tool_use(self, message: AbstractAnthropicMessage) -> bool:
        return type(message) is AnthropicToolResponse
    