import argparse
import logging
import os
import time
from enum import Enum

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatModel(Enum):
    TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_32k = "gpt-4-32k"


class Chatbot:
    def __init__(self, system_prompt: str, model: str, model_args: dict = {}):
        self.system_prompt = system_prompt
        self.model = model
        self.model_args = model_args

        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def __call__(self, message: str):
        self.messages.append({"role": "user", "content": message})
        result = ""
        print(f"\nAssistant: ", end="")
        for chunk in self.execute():
            chunk_message = chunk["choices"][0]["delta"].get("content", "")
            if chunk_message:
                print(chunk_message, end="")
                result += chunk_message
        print()

        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        args = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "temperature": 0,
        }
        args.update(self.model_args)
        return openai.ChatCompletion.create(**args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-prompt", type=str, default="You are a helpful assistant."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[model.value for model in ChatModel],
        default=ChatModel.TURBO.value,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    openai.util.logging.getLogger().setLevel(logging.WARNING)

    assistant = Chatbot(
        args.system_prompt, args.model, {"temperature": float(args.temperature)}
    )
    while True:
        message = input("\nYou: ")
        _ = assistant(message)
