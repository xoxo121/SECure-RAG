import asyncio
import json
import logging
import os
import sys
from random import choice, random
from time import sleep
from typing import cast

import requests
from litellm import completion
from litellm.litellm_core_utils.streaming_handler import (
    ModelResponse as LiteLLMModelResponse,
)
from litellm.types.utils import Choices

from ..schema import (
    AzureModelConfig,
    ChatBuilder,
    GeminiModelConfig,
    GPTModelConfig,
    GroqModelConfig,
    OllamaModelConfig,
    LiteLLMModelConfig,
    Model,
)

logger = logging.getLogger(__name__)


class GroqModel(Model):
    model_name = "Groq"

    def __init__(self, model_config: GroqModelConfig = GroqModelConfig()) -> None:
        self.model_name = model_config.model_name
        self.input_params = model_config.to_dict()
        self.check_api_key()

    def check_api_key(self):
        """
        Check if the API key is available in the environment.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        self.input_params["api_key"] = api_key
        if not api_key:
            print("Please set the GROQ_API_KEY environment variable with the API key.")
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """
        Generate a response for the given messages.

        Args:
            messages (list): List of message dictionaries following the chat format.
            stop_sequences (list): Optional list of stop sequences.

        Returns:
            str: The generated response content.
        """

        # Generate the response
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content


class AzureModel(Model):
    def __init__(self, model_config: AzureModelConfig = AzureModelConfig()):
        self.model_config = model_config
        self.input_params = model_config.to_dict()
        self.check_api_key()

    def check_api_key(self):
        """Check if the API key is available in the environment."""
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_API_VERSION")
        self.input_params["api_key"] = api_key
        self.input_params["api_base"] = api_base
        self.input_params["api_version"] = api_version
        if not api_key:
            print(
                "Please set the OPENAI_API_KEY environment variable with the API key."
            )
            sys.exit(1)

    async def generate(  # type: ignore
        self,
        messages: ChatBuilder,
    ):
        """Generate a response for the given messages."""
        return await asyncio.to_thread(
            completion,
            messages=messages.chat,
            **self.input_params,
        )


class GeminiModel(Model):
    model_name = "Gemini"

    def __init__(self, model_config: GeminiModelConfig = GeminiModelConfig()):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.check_api_key()
        self.input_params = model_config.to_dict()

    def check_api_key(self):
        """Check if the API key is available in the environment."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "Please set the GEMINI_API_KEY environment variable with the API key."
            )
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """Generate a response for the given messages."""
        # Generate the response
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content


class GPTModel(Model):
    def __init__(self, model_config: GPTModelConfig = GPTModelConfig()):
        self.model_config = model_config
        self.check_api_key()
        self.input_params = model_config.to_dict()
        self.model_name = model_config.model_name

    def check_api_key(self):
        """Check if the API key is available in the environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Please set the OPENAI_API_KEY environment variable with the API key."
            )
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """Generate a response for the given messages."""
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content


class OllamaModel(Model):
    def __init__(self, model_config: OllamaModelConfig = OllamaModelConfig()):
        """Initialize Ollama model with configuration.

        Args:
            model_config: Configuration for the Ollama model
        """
        self.model_config = model_config
        self.check_server()
        self.input_params = model_config.to_dict()

    def check_server(self):
        """Check if the Ollama server is running and accessible."""

        try:
            response = requests.get(f"{self.model_config.api_base}/api/version")
            if response.status_code != 200:
                print(
                    f"Cannot connect to Ollama server at {self.model_config.api_base}. "
                    "Please ensure Ollama is running."
                )
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(
                f"Cannot connect to Ollama server at {self.model_config.api_base}. "
                "Please ensure Ollama is running."
            )
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """Generate a response for the given messages."""
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content


class LiteLLMModel(Model):
    def __init__(self, model_config: LiteLLMModelConfig):
        self.model_config = model_config
        self.input_params = model_config.to_dict()

    def check_api_key(self):
        """Check if the API key is available in the environment."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Please set the OPENAI_API_KEY environment variable with the API key."
            )
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """Generate a response for the given messages."""
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content


class MockModel(Model):
    def __init__(
        self,
        model_config: GroqModelConfig = GroqModelConfig(),
        max_sleep_time: float = 0,
    ) -> None:
        """
        Mock model for testing purposes.

        Args:
            model_config (GroqModelConfig, optional): Any model config. Defaults to GroqModelConfig().
            max_sleep_time (int, optional): Sleeps in a call between 0 to `max_sleep_time` seconds. Defaults to 0.
        """
        self.model_name = f"mock/{model_config.model_name}"
        self.input_params = model_config.to_dict()
        self.max_sleep_time = max_sleep_time
        self.check_api_key()

    def check_api_key(self):
        """
        Check if the API key is available in the environment.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        self.input_params["api_key"] = api_key
        if not api_key:
            print("Please set the GROQ_API_KEY environment variable with the API key.")
            sys.exit(1)

    @Model.retry(3)
    async def generate(self, messages: ChatBuilder):
        """
        Generate a response for the given messages.

        Args:
            messages (list): List of message dictionaries following the chat format.
            stop_sequences (list): Optional list of stop sequences.

        Returns:
            str: The generated response content.
        """

        # Generate the response
        response: LiteLLMModelResponse = cast(
            LiteLLMModelResponse,
            await asyncio.to_thread(
                completion,
                messages=messages.chat,
                mock_response=choice(
                    [
                        "Hi, good to see you too",
                        "I'm sorry, I don't understand. I'm not an AI model. I'm just here as a mock message",
                        f"My system prompt is {messages.chat[0]}",
                        json.dumps(
                            {
                                "thought": "I'm not thinking",
                                "tool_calls": [
                                    {
                                        "name": "python_calculator",
                                        "args": {"query": "2+2"},
                                    }
                                ],
                                "audio": "I'm not speaking",
                            },
                            indent=2,
                        ),
                    ]
                ),
                **self.input_params,
            ),
        )  # asyncio.to_thread(completion, ...) fixes one error that causes frequent fails with acompletion
        sleep(random() * self.max_sleep_time)
        if not response:
            raise RuntimeError("No response from the model")
        choices: list[Choices] = cast(list[Choices], response.choices)
        message = choices[0].message
        if not message.content:
            raise RuntimeError("No response content from the model")
        return message.content
