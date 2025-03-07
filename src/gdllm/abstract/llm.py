from abc import ABC, abstractmethod
from typing import List, TypeVar

from .message import AbstractMessage

# T is a generic type variable representing the return type of the structured_output method
T = TypeVar('T')

class AbstractLLM(ABC):
    @abstractmethod
    def get_chat_response(self, messages: List[AbstractMessage]) -> AbstractMessage:
        pass

    @abstractmethod
    def format_user_message(self, message: str) -> AbstractMessage:
        pass

    @abstractmethod
    def new_conversation(self) -> List[AbstractMessage]:
        pass

class AbstractToolUser(ABC):
    @abstractmethod
    def process_tool_calls(self, tool_call_response: AbstractMessage) -> List[AbstractMessage]:
        pass

    @abstractmethod
    def check_tool_use(self, message: AbstractMessage) -> bool:
        pass

class AbstractStructuredOutputer(ABC):
    @abstractmethod
    def structured_output(self, message: str, output_type: T) -> T:
        pass

class AbstractTokenCounter(ABC):
    @abstractmethod
    def token_usage(self) -> dict:
        pass