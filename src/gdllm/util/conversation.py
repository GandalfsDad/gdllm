from ..abstract.llm import AbstractLLM, AbstractToolUser

class Conversation:
    def __init__(self, llm: AbstractLLM, output_user_input: bool = False, output_system_response: bool = True, output_tool_use: bool = False) -> None:
        self.llm = llm
        self.output_user_input = output_user_input
        self.output_system_response = output_system_response
        self.output_tool_use = output_tool_use

        self.messages = llm.new_conversation()
        

    def chat(self, message: str) -> None:
        formatted_message = self.llm.format_user_message(message)

        if self.output_user_input:
            formatted_message.print()

        response = self.llm.get_chat_response(self.messages + [formatted_message])

        self.messages.extend([formatted_message,response])

        # Check if printing response is required
        if self.output_system_response:
            if self.check_tool_use():
                if self.output_tool_use:
                    response.print()
            else:
                response.print()

        while self.check_tool_use():
            tool_responses = self.llm.process_tool_calls(self.messages[-1])
            #formatted_tool_response = [tool_response.to_chat_message() for tool_response in tool_responses]
            self.messages.extend(tool_responses)

            if self.output_tool_use:
                for tool_response in tool_responses:
                    tool_response.print()

            final_response = self.llm.get_chat_response(self.messages)
            self.messages.extend([final_response])

            if self.output_system_response:
                if self.check_tool_use():
                    if self.output_tool_use:
                        final_response.print()
                else:
                    final_response.print()
            
    def check_tool_use(self) -> str:
        if isinstance(self.llm, AbstractToolUser):
            return self.llm.check_tool_use(self.messages[-1])
        else:
            return False
        
    def print_chat(self) -> None:
        for message in self.messages:
            message.print()
        
