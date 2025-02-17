import os
import inspect
import json

from typing import Literal, List
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel
from src.utils.printer import Printer

printer = Printer(identifier="AI")


# Define a Message model using Pydantic for structured validation
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    text: str
    images: List[str] = []  # List of image URLs associated with the message
    tool_calls: List[ChoiceDeltaToolCall] = []  # List of tool calls in the message
    tool_call_id: str = ""  # Identifier for tool call responses


# Base class for AI providers
class Provider:

    def append(self, messages: list[Message]):
        """Append messages to the conversation history."""
        pass

    def complete(self, model: str):
        """Generate a response based on the current conversation."""
        pass

    def stream(self, model: str):
        """Stream responses from the model."""
        pass

    def set_tools(self, tools: list[callable]):
        """Set available tools for function calling."""
        pass

    async def process_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]):
        """Process tool calls asynchronously."""
        pass


# OpenAI-specific implementation of the Provider class
class _OpenAI(Provider):
    tools_map = {}  # Mapping of tool names to functions
    tools = []  # List of available tools
    messages = []  # Conversation history
    model = ""  # Model name
    config = {
        "temperature": 0.5,  # Controls randomness
        "max_tokens": 1000,  # Limits response length
        "top_p": 1,  # Nucleus sampling
        "frequency_penalty": 0,  # Penalizes frequent tokens
        "presence_penalty": 0,  # Encourages topic diversity
    }

    def __init__(self, api_key: str, base_url: str = None):
        """Initialize OpenAI client with API key."""
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def append(self, messages: list[Message]):
        """Convert and append messages to the conversation history."""
        messages = [
            {
                "role": message.role,
                "content": [
                    {"type": "text", "text": message.text},
                    *(
                        [
                            {"type": "image_url", "image_url": {"url": image_url}}
                            for image_url in message.images
                        ]
                        if message.images
                        else []
                    ),
                ],
                **(
                    {
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": tool_call.function,
                            }
                            for tool_call in message.tool_calls
                        ]
                    }
                    if message.tool_calls
                    else {}
                ),
                **(
                    {
                        "tool_call_id": message.tool_call_id,
                    }
                    if message.tool_call_id
                    else {}
                ),
            }
            for message in messages
        ]
        self.messages.extend(messages)
        return self.messages

    def text_to_speech(self, text: str, voice: str, file_path: str):
        """Convert text to speech and save it to a file."""
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        response.write_to_file(file_path)

    async def complete(self, model: str):
        """Generate a response from the model and handle tool calls."""
        response = self.client.chat.completions.create(
            model=model,
            tools=self.tools,
            messages=self.messages,
            **self.config,
        )
        message_text = response.choices[0].message.content
        generated_message = Message(
            role="assistant",
            text=message_text if message_text else "",
        )

        if response.choices[0].message.tool_calls:
            generated_message.tool_calls = response.choices[0].message.tool_calls

        self.append([generated_message])

        if await self.process_tool_calls(generated_message.tool_calls):
            return await self.complete(model)
        else:
            return generated_message.text

    def set_tools(self, tools: list[callable]):
        """Register tools and map them by name."""
        self.tools = [toolify(tool) for tool in tools]
        self.tools_map = {
            tool["function"]["name"]: tools[index]
            for index, tool in enumerate(self.tools)
        }

    async def process_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]):
        """Execute tool calls and append results to the conversation."""
        if len(tool_calls) > 0:
            new_messages = []
            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # Check if the function is async or sync
                if inspect.iscoroutinefunction(self.tools_map[name]):
                    result = await self.tools_map[name](**args)
                else:
                    result = self.tools_map[name](**args)

                new_messages.append(
                    Message(role="tool", text=result, tool_call_id=tool_call.id)
                )

            self.append(new_messages)
            return True
        else:
            return False

    async def stream(self, model: str):
        """Stream responses from the model, handling tool calls dynamically."""
        self.model = model
        response = self.client.chat.completions.create(
            model=model,
            tools=self.tools,
            messages=self.messages,
            stream=True,
            stream_options={"include_usage": True},
            **self.config,
        )

        final_tool_calls = {}
        generated_message = Message(role="assistant", text="")

        for chunk in response:
            if len(chunk.choices) > 0:
                if chunk.choices[0].delta.content:
                    generated_message.text += chunk.choices[0].delta.content
                    yield chunk.choices[0].delta.content

                if chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls or []:
                        index = tool_call.index

                        if index not in final_tool_calls:
                            final_tool_calls[index] = tool_call

                        final_tool_calls[
                            index
                        ].function.arguments += tool_call.function.arguments
            else:
                yield chunk

        generated_message.text = generated_message.text.strip()
        generated_message.tool_calls = [final_tool_calls[i] for i in final_tool_calls]
        self.append([generated_message])

        if await self.process_tool_calls(
            [final_tool_calls[i] for i in final_tool_calls]
        ):
            async for chunk in self.stream(self.model):
                yield chunk


# Factory class to create AI instances with different providers
class AIFactory:
    """
    Creates an AI Instance capable of answering messages.
    """

    def __init__(self, provider: str, api_key: str | None = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        if provider == "openai":
            self.ai = _OpenAI(self.api_key)

        elif provider == "ollama":
            self.ai = _OpenAI(self.api_key, "http://localhost:11434/v1/")
        else:
            raise ValueError(f"Provider {provider} not supported")

    def add_messages(self, messages: List[Message]):
        self.ai.append(messages)

    def complete(
        self, model: str, tools: List[callable] = [], messages: List[Message] = []
    ):
        """Generate a response using the specified model and tools."""
        self.ai.set_tools(tools)
        self.ai.append(messages)
        return self.ai.complete(model)

    async def stream(
        self, model: str, tools: List[callable] = [], messages: List[Message] = []
    ):
        """Stream responses using the specified model and tools."""
        self.ai.set_tools(tools)
        self.ai.append(messages)
        async for chunk in self.ai.stream(model):
            yield chunk

    def text_to_speech(self, text: str, voice: str, file_path: str):
        """Convert text to speech and save it to a file."""
        self.ai.text_to_speech(text, voice, file_path)



def toolify(func):
    """
    Convert a Python function into a valid OpenAI function calling schema. For more information on how to use this, see the OpenAI API documentation.
    Function calling docs: https://platform.openai.com/docs/guides/function-calling
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        Literal: "enum",
    }

    signature = inspect.signature(func)
    parameters = {
        param.name: {"type": type_map.get(param.annotation, "string")}
        for param in signature.parameters.values()
    }
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
