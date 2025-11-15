import json
import logging
import os
from functools import wraps
from typing import Optional

from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)
load_dotenv()


class ChatBuilder:
    """A class representing a chat builder.

    Attributes:
        chat (str): The chat string.

    Methods:
        __str__(): Returns a string representation of the chat builder.
    """

    def __init__(self, max_turns=20):
        self.chat: list[dict[str, str]] = []
        self.roles: list[str] = []
        self.max_turns = max_turns

    def __str__(self):
        """Returns a string representation of the chat builder."""
        return f"{self.chat}"

    def append(self, role: str, content: str):
        if len(self.chat) >= self.max_turns:
            self.chat.pop(2)
        self.chat.append({"role": role, "content": content})
        self.roles.append(role)
        print(len(self.chat))

    def insert(self, index: int, role: str, content: str):
        self.chat.insert(index, {"role": role, "content": content})
        self.roles.insert(index, role)

    def replace(self, index: int, role: str, content: str):
        self.chat[index] = {"role": role, "content": content}
        self.roles.insert(index, role)

    def system_message(self, content: str):
        if "system" not in self.roles:
            self.insert(0, "system", content)
        else:
            self.replace(0, "system", content)

    def user_message(self, content: str):
        self.append("user", content)

    def assistant_message(self, content: str):
        self.append("assistant", content)

    def build(self):
        return self.chat

    def reset_chat(self):
        self.chat = []


class Model:
    model_name: str
    input_params: dict

    def __init__(self) -> None:
        pass

    async def generate(self, messages: ChatBuilder) -> str:
        """
        This method should be implemented by all child classes for prompt generation.
        """
        raise NotImplementedError(
            "This method is meant to be implemented by child classes."
        )

    @staticmethod
    def retry(retries: int = 3):
        """
        Decorator that retries a litellm call a number of times before raising an error.

        Args:
            retries (int, optional): Maximum retries. Defaults to 3.
        """

        def inner(func):
            @wraps(func)
            async def wrapper(self, messages, *args, **kwargs):
                errors: list[Exception] = []
                logger.debug(f"Chat History:\n{json.dumps(messages.chat, indent=2)}")
                for _ in range(retries):
                    try:
                        return await func(self, messages, *args, **kwargs)
                    except Exception as e:
                        errors.append(e)
                        logger.error(f"Error: {e}")
                raise RuntimeError(
                    f"LiteLLM API call failed after {retries} retries"
                ) from errors[-1]

            return wrapper

        return inner


class BaseTool:
    """A base class representing a tool.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        version (str): The version of the tool.
        args (list): A list of arguments for the tool.
        tool_type (str): The type of the tool.

    Methods:
        run(): Abstract method to run the tool.
        __str__(): Returns a string representation of the tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        version: str,
        args: Optional[dict[str, str]] = None,
    ):
        self.name: str = name
        self.description = description
        self.version = version
        self.args: dict[str, str] = args or {}  # argument: description(type)
        self.tool_type: str = "AUA"  # Assistant
        self.tool_output = "default"

    def run(self, *args, **kwargs):
        """Abstract method to run the tool."""
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self):
        """Returns a string representation of the tool."""
        return f"""Name of the tool : {self.name} 
     Tool_Description : {self.description}
     Tool_arguments and their description : {self.args} """

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "args": self.args,
            "tool_type": self.tool_type,
            "tool_output": self.tool_output,
        }


class ModelConfig:
    """A class representing the configuration of a model.

    Attributes:
        name (str): The name of the model.
        api_key (str): The API key for the model.
        version (str): The version of the model.

    Methods:
        __str__(): Returns a string representation of the model configuration.
    """

    def __init__(
        self,
        model_name: str,
        description: str,
        version: str = "1.5",
        api_key=None,
        temperature: float = 0.1,
        timeout: Optional[float | int] = None,
        response_format: str | None = None,
    ):
        self.model_name = model_name
        self.description = description
        self.version = version
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.response_format = response_format

    def __str__(self):
        """Returns a string representation of the model configuration."""
        return f"{self.model_name} - {self.description} - {self.version}"

    def to_dict(self):
        return {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "response_format": self.response_format,
        }


class GroqModelConfig(ModelConfig):
    """A class representing the configuration of the Gemini model.

    Attributes:
        model_name (str): The name of the Azure model.
        description (str): The description of the Azure model.

    Methods:
        __init__(): Initializes the Gemini model configuration.
    """

    def __init__(
        self,
        model_name: str = "groq/llama3-8b-8192",
        description: str = "Groq Llama 3.1 8B Instant",
        api_key: str | None = None,
        timeout: float | int = 10,
    ):
        super().__init__(
            model_name,
            description,
            "0.1",
            api_key or os.environ["GROQ_API_KEY"],
            timeout=timeout,
        )


class AzureModelConfig(ModelConfig):
    """A class representing the configuration of the Gemini model.

    Attributes:
        model_name (str): The name of the Azure model.
        description (str): The description of the Azure model.

    Methods:
        __init__(): Initializes the Gemini model configuration.
    """

    def __init__(
        self,
        model_name: str = "azure/...",
        description: str = "GPT-4o Mini",
        **kwargs,
    ):
        super().__init__(
            model_name, description, version=kwargs.get("version", "1.0"), **kwargs
        )


class GeminiModelConfig(ModelConfig):
    """A class representing the configuration of the Gemini model.

    Attributes:
        model_name (str): The name of the Gemini model.
        description (str): The description of the Gemini model.

    Methods:
        __init__(): Initializes the Gemini model configuration.
    """

    def __init__(
        self,
        model_name: str = "gemini/gemini-1.5-flash-latest",
        description: str = "Gemini 1.5 Models",
        **kwargs,
    ):
        super().__init__(
            model_name,
            description,
            version="1.0",
            api_key=kwargs.get("api_key", os.getenv("GEMINI_API_KEY")),
            **kwargs,
        )


class GPTModelConfig(ModelConfig):
    """A class representing the configuration of the Gemini model.

    Attributes:
        model_name (str): The name of the Gemini model.
        description (str): The description of the Gemini model.

    Methods:
        __init__(): Initializes the Gemini model configuration.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        description: str = "GPT-4o Model",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name,
            description,
            version="1.0",
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )


class LiteLLMModelConfig(ModelConfig):
    def __init__(self, model_name: str, description: str = "LiteLLM Model", **kwargs):
        super().__init__(model_name, description, version="1.0", **kwargs)


class OllamaModelConfig(ModelConfig):
    """A class representing the configuration of the Ollama model.

    Attributes:
        model_name (str): The name of the Ollama model.
        description (str): The description of the Ollama model.
        api_base (str): The base URL for the Ollama API.
    """

    def __init__(
        self,
        model_name: str = "ollama/llama3",  # Ollama model names like llama2, mistral, etc.
        description: str = "Ollama Local Model",
        api_base: str = "http://localhost:11434",  # Default Ollama server URL
        api_key: str | None = None,  # Usually not needed for Ollama local deployment
    ):
        super().__init__(model_name, description)
        self.api_base = api_base

    def to_dict(self):
        return {
            "model": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
        }


class BaseState:
    """A base class representing a state.

    Attributes:
        name (str): The name of the state.
        goal (str): The goal of the state.
        instructions (str): The instructions for the state.
        tools (list): A list of tools available in the state.

    Methods:
        __str__(): Returns a string representation of the state.
    """

    def __init__(
        self,
        name: str,
        goal: str,
        instructions: str,
        model: Model,
        tools: list[BaseTool] = [],
    ):
        self.name = name
        self.goal = goal
        self.instructions = instructions
        self.model = model
        self.tools: list[BaseTool] = tools

    def add_tool(self, tool: BaseTool):
        """Adds a tool to the state."""
        self.tools.append(tool)

    def __str__(self):
        """Returns a string representation of the state."""
        return f"""
        <state_details>
        Name: {self.name}
        Goal: {self.goal}
        Instructions: {self.instructions}
        Tools: {[str(tool) for tool in self.tools]}
    """

    def basic_info(self) -> str:
        return f"""
{self.name}
"""

    def to_dict(self):
        return {
            "name": self.name,
            "goal": self.goal,
            "instructions": self.instructions,
            "tools": [tool.to_dict() for tool in self.tools],
        }


class ToolImageOutput:
    """A class representing the output of a tool with an image.

    Attributes:
        image (PIL.Image.Image): Pointer to PIL Image.
    """

    def __init__(self, image: Image.Image):
        self.image = image
