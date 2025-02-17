from src.ai import AIFactory, Message
from dotenv import load_dotenv
import asyncio
from src.utils.printer import Printer, colorize
from src.utils.tools import get_tools
import os

printer = Printer()
load_dotenv()

system_message = Message(
    role="system",
    text="You are a helpful developer assistant capable of using tools and code in any language. Help the user with their questions and tasks using all your available tools and context.",
)


PROVIDER = "openai"
MODEL = "gpt-4o-mini"


async def main():
    ai = AIFactory(PROVIDER, os.getenv("OPENAI_API_KEY"))
    ai.add_messages([system_message])

    while True:
        user_input = input(colorize("Enter a message: ", "blue"))
        ai.add_messages([Message(role="user", text=user_input)])
        async for chunk in ai.stream(
            model=MODEL,
            tools=get_tools(ai),
        ):
            if isinstance(chunk, str):
                printer.yellow(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    asyncio.run(main())
