from src.ai import AIFactory
import os
from typing import Literal
from pydub import AudioSegment
from pydub.playback import play
from src.utils.printer import Printer

printer = Printer(identifier="TOOLS")


def get_tools(ai: AIFactory):

    async def get_weather(city: str):
        """
        Get the weather in a city
        """
        printer.blue(f"Getting weather in {city}")
        return f"The weather in {city} is sunny"

    async def list_working_directory_files():
        """
        List the files in the working directory
        """
        printer.blue("Listing working directory files")
        return "\n".join(os.listdir())

    async def read_file(file_name: str):
        """
        Read the contents of a file
        """
        printer.blue(f"Reading file: {file_name}")
        file_contents = ""
        try:
            with open(file_name, "r") as f:
                file_contents += f.read()
        except IOError as e:
            return f"An error occurred while reading {file_name}: {e}\n"

        return file_contents

    async def talk(
        text: str,
        voice: Literal[
            "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "coral", "sage"
        ],
    ):
        """
        Convert text to speech, the available voices are:
        - alloy
        - echo
        - fable
        - onyx
        - nova
        - shimmer
        - ash
        - coral
        - sage
        """
        printer.blue(f"Talking: {text} with voice: {voice}")
        ai.text_to_speech(text, voice, "speech.mp3")
        audio = AudioSegment.from_mp3("speech.mp3")
        play(audio)
        return "Audio generated successfully"

    async def write_file(file_name: str, file_content: str):
        """
        Create or update a file with the given content, encodes as UTF-8
        """
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(file_content)
            return f"File {file_name} created successfully"
        except IOError as e:
            return f"An error occurred while creating the file: {e}"

    return [
        get_weather,
        list_working_directory_files,
        read_file,
        write_file,
        talk,
    ]
