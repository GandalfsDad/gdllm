from ...abstract import AbstractLLM, AbstractToolUser, AbstractStructuredOutputer, AbstractTokenCounter
from .config import GoogleConfig
from .message import AbstractGoogleMessage, GoogleToolResponse, GoogleResponse, GoogleMessage, GoogleToolResultResponse
from .tool import GoogleToolProvider

from typing import List, TypeVar, Any

from google import genai as google_genai

T = TypeVar('T')

class Google(AbstractLLM, AbstractToolUser, AbstractStructuredOutputer, AbstractTokenCounter):
    def __init__(self, config: GoogleConfig):
        self.config = config
        self.client = google_genai.Client(api_key=self.config.api_key)

        self.__tokens = {
            "input": 0,
            "output": 0,
            "cached_input": 0
        }
        
    def get_chat_response(self, messages: List[AbstractGoogleMessage]) -> AbstractGoogleMessage:
        parsed_messages = [message.to_chat_message() for message in messages]

        chat = self.client.chats.create(
            model=self.config.model, 
            **({'history':parsed_messages[:-1]} if len(parsed_messages) > 1 else {}),
            config= self.config.get_call_args()
        )

        response = chat.send_message(parsed_messages[-1])

        self.update_token_usage(response.usage_metadata)

        return self.process_response(response.candidates[0])
    
    def process_response(self, response: Any) -> AbstractGoogleMessage:
        if response.content.parts[0].function_call is not None:
            return GoogleToolResponse(response)
        else:
            return GoogleResponse(response)
        
    def new_conversation(self) -> List[AbstractGoogleMessage]:
        return []
    
    def format_user_message(self, message: str) -> GoogleMessage:
        return GoogleMessage(role='user', message=message)
    
    def process_tool_calls(self, tool_call_response: GoogleToolResponse) -> List[GoogleToolResultResponse]:
        if self.config.tool_use_available():
            results = []
            for tool_call in tool_call_response.response.content.parts:
                #sometimes tool call brins out an extrs text response (discarded)
                if tool_call.function_call is None:
                    continue

                func = tool_call.function_call.name
                args = {k:v for k,v in tool_call.function_call.args.items()}
                result = GoogleToolProvider.use_tool(func, args)

                results.append(GoogleToolResultResponse(func, result))
            return results
        else:
            raise Exception("Tool use is not available for this model")
    
    def check_tool_use(self, message: AbstractGoogleMessage) -> bool:
        return type(message) is GoogleToolResponse
    
    def structured_output(self, message: str, output_type: T) -> T:
        if self.config.structured_output_available():
            structured_response = self.client.models.generate_content(
                model=self.config.model,
                contents=message,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': output_type,
                },
            )

            self.update_token_usage(structured_response.usage_metadata)

            return structured_response.parsed
        else:
            raise Exception("Structured output is not available for this model")
    
    def update_token_usage(self, usage: dict):
        self.__tokens["input"] += usage.prompt_token_count
        self.__tokens["output"] += usage.candidates_token_count

        if cctc := usage.cached_content_token_count:
            self.__tokens["cached_input"] += cctc

    def token_usage(self) -> dict:
        return self.__tokens