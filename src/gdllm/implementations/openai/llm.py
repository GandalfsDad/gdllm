from ...abstract import AbstractLLM, AbstractMessage, AbstractToolUser, AbstractStructuredOutputer, AbstractTokenCounter
from .config import OpenAIConfig
from .message import AbstractOpenAIMessage, OpenAIToolResponse, OpenAIResponse, OpenAIMessage, OpenAIToolResultResponse
from .tool import OpenAIToolProvider

import json
from typing import List, TypeVar, Any, Optional

from openai import OpenAI as BaseOpenAI

T = TypeVar('T')


class OpenAI(AbstractLLM, AbstractToolUser, AbstractStructuredOutputer, AbstractTokenCounter):
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = BaseOpenAI(api_key=config.api_key, base_url=config.base_url)

        self.__tokens = {
            "input": 0,
            "output": 0,
            "cached_input": 0,
            "reasoning": 0
        }

    def get_chat_response(self, messages: List[AbstractOpenAIMessage]) -> AbstractOpenAIMessage:
        parsed_messages = [message.to_chat_message() for message in messages]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=parsed_messages,
            **self.config.get_call_args()
        )

        self.update_token_usage(response.usage)

        return self.process_response(response.choices[0])
    
    def process_response(self, response: Any) -> AbstractOpenAIMessage:
        if response.finish_reason=='tool_calls':
            return OpenAIToolResponse(response)
        else:
            return OpenAIResponse(response)
        
    def new_conversation(self) -> List[AbstractOpenAIMessage]:
        return [self.format_system_message(self.config.system_message)] if self.config.system_message else []

    def format_user_message(self, message: str) -> AbstractMessage:
        return OpenAIMessage(message, "user")
    
    def format_system_message(self, message: str) -> AbstractMessage:
        return OpenAIMessage(message, self.config.system_name)

    def process_tool_calls(self, tool_call_response: OpenAIToolResponse) -> List[OpenAIToolResultResponse]:
        results = []
        for tool_call in tool_call_response.response.message.tool_calls:
            func, args = tool_call.function.name, json.loads(tool_call.function.arguments)
            result = OpenAIToolProvider.use_tool(func, args)

            results.append(OpenAIToolResultResponse(tool_call.id, result))
        return results

    def check_tool_use(self, message: AbstractMessage) -> bool:
        return type(message) is OpenAIToolResponse

    def structured_output(self, message: str, output_type: T, system_message_override: Optional[str] = None) -> T:
        messages = []
        if system_message_override:
            messages.append(self.format_system_message(system_message_override))
        elif self.config.system_message:
            messages.append(self.format_system_message(self.config.system_message))
        else:
            pass

        messages.append(self.format_user_message(message))

        parsed_messages = [message.to_chat_message() for message in messages]

        structured_response = self.client.beta.chat.completions.parse(
            model = self.config.model,
            messages = parsed_messages,
            response_format=output_type,
            **self.config.get_call_args()
        )

        self.update_token_usage(structured_response.usage)

        return structured_response.choices[0].message.parsed
    
    def update_token_usage(self, usage):
        
        self.__tokens["input"] += usage.prompt_tokens

        if hasattr(usage, "completion_tokens"):
            self.__tokens["output"] += usage.completion_tokens

        if hasattr(usage, "prompt_tokens_details"):
            self.__tokens["cached_input"] += usage.prompt_tokens_details.cached_tokens
        
        if hasattr(usage, "completion_tokens_details"):
            self.__tokens["reasoning"] += usage.completion_tokens_details.reasoning_tokens
    
    def token_usage(self) -> dict:
        return self.__tokens