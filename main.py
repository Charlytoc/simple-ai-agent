from src.ai import AIFactory, Message
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Literal
from pydub import AudioSegment
from pydub.playback import play
from src.utils.printer import Printer, colorize

printer = Printer("main")
load_dotenv()

initial_messages = [
    Message(
        role="system",
        text="You are a helpful developer assistant capable of using tools and code in any language. Help the user with their questions and tasks using all your available tools and context.",
    ),
]


async def main():
    ai = AIFactory("openai", os.getenv("OPENAI_API_KEY"))

    async def get_weather(city: str):
        """
        Get the weather in a city
        """
        return f"The weather in {city} is sunny"

    async def list_working_directory_files():
        """
        List the files in the working directory
        """
        return "\n".join(os.listdir())

    async def read_files(files_list: List[str]):
        """
        Read the contents of different files specified in a list
        """
        file_contents = ""
        for file in files_list:
            try:
                with open(file, "r") as f:
                    file_contents += f"FILE: {file}\n{f.read()}\n\n"
            except IOError as e:
                file_contents += f"An error occurred while reading {file}: {e}\n"
        return file_contents

    async def write_file(file_name: str, file_content: str):
        """
        Write to a file, replaces the entire content of the file
        """
        try:
            with open(file_name, "w") as f:
                f.write(file_content)
            return f"File {file_name} written successfully"
        except IOError as e:
            return f"An error occurred while writing the file: {e}"

    async def talk(
        text: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    ):
        """
        Convert text to speech, the available voices are:
        - alloy
        - echo
        - fable
        - onyx
        - nova
        - shimmer
        """
        ai.text_to_speech(text, voice, "speech.mp3")
        audio = AudioSegment.from_mp3("speech.mp3")
        play(audio)
        return "Audio generated successfully"

    async def create_file(file_name: str, file_content: str):
        """
        Create a new file with the given content
        """
        try:
            with open(file_name, "w") as f:
                f.write(file_content)
            return f"File {file_name} created successfully"
        except IOError as e:
            return f"An error occurred while creating the file: {e}"

    while True:
        # The user input should be colored
        user_input = input(colorize("Enter a message: ", "blue"))
        initial_messages.append(Message(role="user", text=user_input))
        response = await ai.complete(
            messages=initial_messages,
            model="gpt-4o-mini",
            tools=[
                get_weather,
                list_working_directory_files,
                read_files,
                write_file,
                talk,
                create_file,
            ],
        )

        printer.green(response)


if __name__ == "__main__":
    asyncio.run(main())
