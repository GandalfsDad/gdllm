from ...abstract import AbstractLLM, AbstractMessage, AbstractToolUser, AbstractStructuredOutputer
from .config import OpenAIConfig
from .message import AbstractOpenAIMessage, OpenAIToolResponse, OpenAIResponse, OpenAIMessage, OpenAIToolResultResponse
from .tool import OpenAIToolProvider

import json
from typing import List, TypeVar, Any

from openai import OpenAI as BaseOpenAI

T = TypeVar('T')


class OpenAI(AbstractLLM, AbstractToolUser, AbstractStructuredOutputer):
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = BaseOpenAI(api_key=config.api_key, base_url=config.base_url)

    def get_chat_response(self, messages: List[AbstractOpenAIMessage]) -> AbstractOpenAIMessage:
        parsed_messages = [message.to_chat_message() for message in messages]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=parsed_messages,
            **self.config.get_call_args()
        )

        return self.process_response(response.choices[0])
    
    def process_response(self, response: Any) -> AbstractOpenAIMessage:
        if response.finish_reason=='tool_calls':
            return OpenAIToolResponse(response)
        else:
            return OpenAIResponse(response)

    def format_user_message(self, message: str) -> AbstractMessage:
        return OpenAIMessage(message, "user")

    def process_tool_calls(self, tool_call_response: OpenAIToolResponse) -> List[OpenAIToolResultResponse]:
        results = []
        for tool_call in tool_call_response.response.message.tool_calls:
            func, args = tool_call.function.name, json.loads(tool_call.function.arguments)
            result = OpenAIToolProvider.use_tool(func, args)

            results.append(OpenAIToolResultResponse(tool_call.id, result))
        return results

    def check_tool_use(self, message: AbstractMessage) -> bool:
        return type(message) is OpenAIToolResponse

    def structured_output(self, message: str, output_type: T) -> T:
        structured_response = self.client.beta.chat.completions.parse(
            model = self.config.model,
            messages = [self.format_user_message(message).to_chat_message()],
            response_format=output_type,
            **self.config.get_call_args()
        )

        return structured_response.choices[0].parsed