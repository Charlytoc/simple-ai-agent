import os
import inspect
import json

from typing import Literal, List
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]

    text: str
    images: List[str] = []
    tool_calls: List[ChoiceDeltaToolCall] = []
    tool_call_id: str = ""


class Provider:

    def __init__(self, api_key: str):
        self.api_key = api_key

    def append(self, messages: list[Message]):
        pass

    def complete(self, model: str):
        pass

    def stream(self, model: str):
        pass

    def set_tools(self, tools: list[callable]):
        pass

    async def process_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]):
        pass


class _OpenAI(Provider):

    tools_map = {}
    tools = []
    messages = []
    model = ""
    config = {
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    def __init__(self, api_key: str):

        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def append(self, messages: list[Message]):
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
        # Correct streaming way! :)
        # with self.client.audio.speech.with_streaming_response.create(
        #     model="tts-1",
        #     voice=voice,
        #     input=text,
        # ) as response:
        #     response.stream_to_file(file_path)

        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        response.write_to_file(file_path)

    async def complete(self, model: str):

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
        self.tools = [toolify(tool) for tool in tools]
        self.tools_map = {
            tool["function"]["name"]: tools[index]
            for index, tool in enumerate(self.tools)
        }

    async def process_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]):

        if len(tool_calls) > 0:
            new_messages = []
            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
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

        await self.process_tool_calls([final_tool_calls[i] for i in final_tool_calls])

        async for chunk in self.stream(self.model):
            yield chunk


class AIFactory:
    """
    Creates an AI Instance capable of answering messages
    """

    def __init__(self, provider: str, api_key: str | None = None):
        self.provider = provider
        self.api_key = api_key
        if api_key is None:
            self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if provider == "openai":
            self.ai = _OpenAI(self.api_key)
        else:
            raise ValueError(f"Provider {provider} not supported")

    def complete(self, messages: List[Message], model: str, tools: List[callable] = []):
        self.ai.set_tools(tools)
        self.ai.append(messages)
        return self.ai.complete(model)

    async def stream(
        self, messages: List[Message], model: str, tools: List[callable] = []
    ):
        self.ai.set_tools(tools)
        self.ai.append(messages)
        async for chunk in self.ai.stream(model):
            yield chunk

    def text_to_speech(self, text: str, voice: str, file_path: str):
        self.ai.text_to_speech(text, voice, file_path)


def dict_to_message(data: dict):
    return Message(role=data["role"], text=data["text"], images=data["images"])


def toolify(func):
    """
    Convierte una función de Python en un esquema JSON válido para OpenAI function calling.
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
    parameters = {}

    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}

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
                "required": required if required else [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
