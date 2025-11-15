import json
import logging
import re
from pathlib import Path
from typing import (
    Any,
    NotRequired,
    Optional,
    TypedDict,
    cast,
    NamedTuple,
)
import os
import warnings
import asyncio

from PIL import Image

from FinAgent.schema import ToolImageOutput

from .schema import (
    BaseState,
    BaseTool,
    ChatBuilder,
)

from .guardrails.guardrail_api import FinGuard
from .ToolHandler import ToolHandler, LLMToolCall
from .tools.state_change_tool import StateChangeTool
from .utils import JSONExtractionError, json_extractor_for_tool_caller
from .config.prompts import (
    SYSTEM_PROMPT_MULTI_STATE_AGENT,
    SYSTEM_PROMPT_STATELESS_AGENT,
    EXPLAINABILITY_AGENT_PROMPT,
    SYSTEM_PROMPT_META_STATE_AGENT,
    SYSTEM_PROMPT_HYFER,
)
from .config.states import (
    multi_state_agent_states,
    stateless_agent_states,
    explainability_states,
    auto_states,
    mock_states,
    meta_states,
    hyfer_states,
)


logger = logging.getLogger(__name__)

schema_path = Path(__file__).parent / "schema/response.schema.json"
response_schema = json.loads(open(schema_path).read())
guardrail = FinGuard()
guardrail_check = (os.environ["USE_GUARDRAIL"].lower() == "true")

class NonToolResponse(TypedDict):
    thought: str
    audio: str
    plan: NotRequired[str]
    queries: NotRequired[list[str]]


class CleanLLMOutput(NamedTuple):
    tool_calls: list[LLMToolCall]
    remaining_response: NonToolResponse


class ToolOutput(NamedTuple):
    tool_display_results: dict[str, ToolImageOutput]
    response: dict[str, str | list[str]]
    AUA: bool


class AgentOutput(NamedTuple):
    tool_outputs: list[ToolOutput]
    remaining_response: NonToolResponse  # can include plans


class Agent:
    state_key: str = "BaseState"
    base_prompt: str = "You are a conversational agent who helps users with their finance-related queries."
    states: dict[str, BaseState]

    def __init__(
        self,
    ) -> None:
        self.states: dict[str, BaseState] = self.states
        self.model = self.states[self.state_key].model
        logger.info(f"""
Starting Model Name: {self.model.model_name}
""")
        self.messages = ChatBuilder()
        if len(self.states) > 1:
            StateChangeTool(states=self.states)
            # Adds itself to all states
        self.set_system_prompt()
        self.tool_handler = ToolHandler(
            tools=self.states[self.state_key].tools,
            agent=self,
        )
        self.tool_counts = {
            tool.name: 0 for tool in self.states[self.state_key].tools
        }
        self.images: list[Image.Image] = []
        self.unsafe_count = 0

    def set_system_prompt(self, prompt: Optional[str] = None, state_key=None):
        self.messages.system_message(content=self.base_prompt)

    def add_user_message(self, message: str):
        if self.images:
            message = f"""{message}
The user has uploaded the following images: {', '.join((f'image_{i}' for i in range(len(self.images))))}
An image can be inserted as an argument directly - "args": {{"<arg_name>": "$image_1$"}}"""
            self.messages.user_message(message)
        else:
            self.messages.user_message(message)

    def add_assistant_message(self, message: str):
        self.messages.assistant_message(message)

    def add_input_image(self, image: Image.Image):
        self.images.append(image)

    async def get_assistant_response(self):
        answer = await self.model.generate(self.messages)

        self.add_assistant_message(answer)
        return answer

    def new_user_query(self) -> None: ...

    def extract_info(self, response: str) -> CleanLLMOutput:
        """Deal with images and potentially other placeholders

        Returns:
            tools (list) - tool_calls after json.loads
            info (dict) - remaining response
        """
        response = response.strip()  # no whitespace in JSON
        response = re.sub(
            r'"\$image_(?P<num>[0-9]+)\$"', r'"<image_\g<num>>"', response
        )
        try:
            info: dict = json_extractor_for_tool_caller(response)
            logger.debug(f"Extracted JSON: {info}")
            tools: list[LLMToolCall]
            tools = info.pop("tool_calls")
        except JSONExtractionError as e:
            logger.error(f"Error in JSON: {response}\nError: {e}")
            raise JSONExtractionError("Invalid JSON") from e
        except IndexError as e:
            logger.error(f"No JSON output {response}\nError: {e}")
            raise ValueError("No JSON output") from e
        for tool in tools:
            args: dict[str, Any] = cast(dict[str, Any], tool["args"])
            for arg, val in args.items():
                if isinstance(val, str) and (
                    img := re.match(r"<image_(?P<num>[0-9]+)>", val)
                ):
                    args[arg] = self.images[int(img.group("num"))]
        return CleanLLMOutput(tools, info)

    async def agent_loop(
        self,
    ) -> AgentOutput:
        logger.info(json.dumps(self.messages.chat, indent=2))
        self.model = self.states[self.state_key].model
        response: str = await self.get_assistant_response()
        logger.info(response)
        tool_calls, response_dict = self.extract_info(response)
        if not tool_calls:
            return AgentOutput([ToolOutput({}, {}, False)], response_dict)
        logger.info(f"{tool_calls = }")

        tool_results = await self.tool_handler.handle_tools(tool_calls)
        output = []
        for result in tool_results:
            AUA = result.get("AUA", False)
            tool_results_display: dict[str, ToolImageOutput] = {}
            tool_results_text = {}
            for key in result:
                if isinstance(out := result[key], ToolImageOutput):
                    self.add_input_image(out.image)
                    tool_results_display[key] = out
                    tool_results_text[key] = "Image shown to user"
                elif isinstance(out, str):
                    try:
                        tool_results_text[key] = json.loads(out)
                    except json.JSONDecodeError:
                        tool_results_text[key] = str(out)
                else:
                    tool_results_text[key] = str(out)
            self.add_user_message(
                json.dumps({"tool_results": tool_results_text})
            )
            output.append(
                ToolOutput(tool_results_display, tool_results_text, AUA)
            )
        
        safety_check = await (guardrail.check(response_dict["audio"], guardrail_check))
        if not safety_check["is_safe"] and self.unsafe_count <= 3:
            self.unsafe_count += 1
            self.add_user_message(f"Safety check failed. Reason: {safety_check["reason"]}")
            return await self.agent_loop()
        else:
            if not safety_check["is_safe"]:
                response_dict = {
                    "thought": "I have made a bad response",
                    "audio": "Guardrail has blocked this response"
                }
            self.unsafe_count = 0
            return AgentOutput(output, response_dict)


class ExplainabilityAgent(Agent):
    state_key = "ExplainabilityState"
    base_prompt = EXPLAINABILITY_AGENT_PROMPT
    states = explainability_states

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.set_system_prompt()

    def add_user_message(self, message: str):
        if self.images:
            message = f"""{message}
The user has uploaded the following images: {', '.join((f'image_{i}' for i in range(len(self.images))))}
An image can be inserted as an argument directly - "args": {{"<arg_name>": "$image_1$"}}"""
            self.messages.user_message(message)
        else:
            self.messages.user_message(message)

    def set_system_prompt(self, prompt: Optional[str] = None, state_key=None):
        self.state_key = state_key or self.state_key
        self.base_prompt = prompt or self.base_prompt
        self.messages.system_message(
            content=self.base_prompt.format(
                state_details=self.states[self.state_key]
            )
        )


class StatelessAgent(Agent):
    state_key: str = "MasterState"
    base_prompt = SYSTEM_PROMPT_STATELESS_AGENT
    states = stateless_agent_states

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.set_system_prompt()

    def set_system_prompt(self, prompt: Optional[str] = None, state_key=None):
        if state_key is not None and state_key != self.state_key:
            warnings.warn(
                "State key should not be set for stateless agent",
                category=RuntimeWarning,
            )
        state_key = self.state_key
        self.base_prompt = prompt or self.base_prompt
        self.messages.system_message(
            content=self.base_prompt.format(
                state_details=self.states[state_key]
            )
        )


class MultiStateAgent(Agent):
    state_key: str = "BaseState"
    base_prompt = SYSTEM_PROMPT_MULTI_STATE_AGENT
    states = multi_state_agent_states

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.set_system_prompt()

    def set_system_prompt(self, prompt: Optional[str] = None, state_key=None):
        if not state_key:
            state_key = self.state_key
        self.state_key = state_key
        self.base_prompt = prompt or self.base_prompt
        self.messages.system_message(
            content=self.base_prompt.format(
                state_details=self.states[self.state_key]
            )
        )


class AutoAgentToolCallInfo(TypedDict):
    result: Any
    args: dict[str, Any]


class AutoAgent(Agent):
    state_key: str = "AutoState"
    base_prompt = ""
    states = auto_states
    MAX_ITERATIONS = 2
    math_tools: set[str] = {"Python_Calculator", "wolfram_alpha_query"}

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.set_system_prompt()
        self.iterations: int = 0
        self.model = self.states[self.state_key].model
        self.states[self.state_key].tools = [
            tool
            for tool in self.states[self.state_key].tools
            if not isinstance(tool, StateChangeTool)
        ]
        self.tool_call_history: list[dict[str, AutoAgentToolCallInfo]] = []
        self.new_query = (
            False  # Default to False at first or query will be duplicated
        )

    @staticmethod
    def document_template(docs: list[str]) -> str:
        document_template = f"""
Here are the documents retrieved using the rag_tool (tool description is given below)

{"\n\n".join([f"retrieved text ({i+1}) -> " + doc for i,doc in enumerate(docs)]) } """
        return document_template

    @staticmethod
    def tools_template(tools: dict[str, BaseTool]) -> str:
        tools_template = f"""
        Here are the available tools and their description
        {"\n\n".join([tool.__str__() for tool_name, tool in tools.items()])}

        These are the only available tools and Do not hallucinate any tools 
        """.replace("financial_rag_tool", "rag_tool")
        return tools_template

    def initial_llm_query(
        self,
        query: str,
        math_tools: dict[str, BaseTool],
        info_tools: dict[str, BaseTool],
    ) -> str:
        llm_input = (
            f"""

Try to answer the query with the retrieved documents using rag_tool:
{self.document_template(self.docs)} 
Here are the additional tools to perform mathematical calculations in the answer
{self.tools_template(math_tools)}
                    
query : "{query}"

Answer: ?

NOTE : The answers should be strictly based on the provided information and mention the resources & excerpts you used for answering (like document number, file name, quoting the exact line)

-> If you couldn't able to answer the question based on the documents, Use the tools given below to get more information
-> If you have doubt about the technical terms in the question, use the tools to find the definition of those terms 
Call tools -> get more info -> Try to answer the question with the info

{self.tools_template(info_tools)}

Tool calling must be in the Json format , Strictly follow the Json Schema """
            + """
\n
{
    "tool_calls": [ {"name": "...", "args": {...}}, {"name": "...", "args": {...}} ]
} """
        )
        return llm_input

    def llm_query(
        self,
        query: str,
        math_tools: dict[str, BaseTool],
        info_tools: dict[str, BaseTool],
    ) -> str:
        prev_res = ""
        for tools_in_a_call in self.tool_call_history:
            for tool_name, value in tools_in_a_call.items():
                if tool_name == "financial_rag_tool" and isinstance(
                    value["result"], list
                ):
                    self.docs = value["result"]
                    continue
                prev_res += f"tool_called : {tool_name} \n args used : {value['args']} \n tool_output : {value['result']} \n\n"

        llm_prompt_with_res = f"""

Try to answer the query with the retrieved documents using rag_tool:
{self.document_template(self.docs)[:10000]} 
            
Here are the additional tools to perform mathematical calculations in the answer
{self.tools_template(math_tools)}

Here, refer to the previous tool calls made and the arguments used and the output from tool call (description of available tools are given at the end)

{prev_res}

query : "{query}"

Answer: ?

NOTE : The answers should be strictly based on the provided information and mention the resources & excerpts you used for answering (like document number, file name, quoting the exact line)"""

        info_tools_prompt = (
            f"""
-> If you couldn't able to answer the question based on the documents, Use the tools given below to get more information 
-> If you have doubt about the technical terms in the question, use the tools to find the definition of those terms
Call tools -> get more info -> Try to answer the question with the info

{self.tools_template(info_tools)}

Tool calling must be in the Json format, Strictly Follow the Json Schema """
            + """
\n
{
    "tool_calls": [{"name": "...", "args": {...}}, {"name": "...", "args": {...}} ] 
}"""
        )
        return llm_prompt_with_res + info_tools_prompt

    async def get_assistant_response(self) -> str:
        math_tools = {
            tool.name: tool
            for tool in self.states[self.state_key].tools
            if tool.name in self.math_tools
        }
        info_tools = {
            tool.name: tool
            for tool in self.states[self.state_key].tools
            if tool.name not in self.math_tools
        }
        if self.iterations == 0:
            self.messages.reset_chat()
            query = self.initial_llm_query(
                self.original_query,
                math_tools=math_tools,
                info_tools=info_tools,
            )
            self.original_query = query
            self.add_user_message(query)
            answer = await self.model.generate(self.messages)
            self.add_assistant_message(answer)
            return answer
        elif self.iterations < self.MAX_ITERATIONS:
            self.messages.reset_chat()
            query = self.llm_query(
                self.original_query,
                math_tools=math_tools,
                info_tools=info_tools,
            )
            self.add_user_message(query)
            answer = await self.model.generate(self.messages)
            self.add_assistant_message(answer)
            return answer
        else:
            self.messages.reset_chat()
            query = (
                self.llm_query(
                    self.original_query,
                    math_tools=math_tools,
                    info_tools=info_tools,
                )
                + """If you couldn't able to answer the question based on the documents and the information gathered using the tool, 
    Apologise the user and output the reason why you couldn't able to answer the query"""
            )

            self.add_user_message(query)
            answer = await self.model.generate(self.messages)
            self.add_assistant_message(answer)
            return answer

    def system_prompt(self, prompt: Optional[str] = None, state_key=None):
        if self.messages.chat[0]["role"] == "System":
            self.messages.chat.pop(0)

    async def agent_loop(self) -> AgentOutput:
        if not self.iterations:
            return await self._first_agent_loop()  # Call HyDE first
            # Agent loop is called again by the application, as AUA is True
        elif self.new_query:
            self.new_query = False
            logger.debug(f"New Query: {self.messages.chat[-1]['content']}")
            self.original_query += f"\n{self.messages.chat[-1]['content']}"

        response: str = await self.get_assistant_response()
        response = response.replace("rag_tool", "financial_rag_tool")
        logger.info(response)
        tool_calls, response_dict = self.extract_info(response)
        if not tool_calls:
            return AgentOutput([ToolOutput({}, {}, False)], response_dict)
        logger.info(f"{tool_calls = }")
        tool_call_history_item: dict[str, AutoAgentToolCallInfo] = {}
        # tool_call_history is a list, but each item is a dictionary
        # Multiple tool calls to the same tool are not permitted, only the second tool call is accepted.
        tool_results = await self.tool_handler.handle_tools(tool_calls)
        for call, result in zip(tool_calls, tool_results):
            logger.debug(f'History -> {call["name"]} : {result[call["name"]]}')
            tool_call_history_item[call["name"]] = {
                "result": result[call["name"]],
                "args": call["args"],
            }
        self.tool_call_history.append(tool_call_history_item)
        output: list[ToolOutput] = []
        if self.iterations > self.MAX_ITERATIONS:
            return AgentOutput([ToolOutput({}, {}, False)], response_dict)
        for result in tool_results:
            AUA = result.get("AUA", False)
            tool_results_display: dict[str, ToolImageOutput] = {}
            tool_results_text = {}
            for key in result:
                if isinstance(out := result[key], ToolImageOutput):
                    self.add_input_image(out.image)
                    tool_results_display[key] = out
                    tool_results_text[key] = "Image shown to user"
                elif isinstance(out, str):
                    try:
                        tool_results_text[key] = json.loads(out)
                    except json.JSONDecodeError:
                        tool_results_text[key] = str(out)
                else:
                    tool_results_text[key] = str(out)
            self.add_user_message(
                json.dumps({"tool_results": tool_results_text})
            )
            output.append(
                ToolOutput(tool_results_display, tool_results_text, AUA)
            )
        self.iterations += 1

        safety_check = await (guardrail.check(response_dict["audio"], check=guardrail_check))
        if not safety_check["is_safe"] and self.unsafe_count <= 3:
            self.unsafe_count += 1
            self.add_user_message(f"Safety check failed. Reason: {safety_check["reason"]}")
            return await self.agent_loop()
        else:
            self.unsafe_count = 0
            return AgentOutput(output, response_dict)

    async def _first_agent_loop(self) -> AgentOutput:
        logger.info("Starting AutoAgent")
        # Get to last user query
        self.original_query = self.messages.chat[-1]["content"]
        logger.debug(f"Original Query: {self.original_query}")
        tool_calls: list[LLMToolCall] = [
            {
                "name": "financial_rag_tool",
                "args": {
                    "query": self.original_query,
                    "top_k1": 20,
                    "top_k2": 10,
                    "n_similar": 2,
                },
            }
        ]
        response_dict: NonToolResponse = {
            "thought": "Calling financial_rag_tool, an advanced retriever with high quality responses",
            "audio": "Automatically calling financial_rag_tool...",
        }
        tool_results = await self.tool_handler.handle_tools(tool_calls)
        tool_results = tool_results[0]
        self.docs: list[str] = tool_results["financial_rag_tool"]
        logger.debug(f"Docs: {"\n".join(self.docs)}")
        tool_results_text = {}
        tool_results_display = {}
        for key in tool_results:
            if isinstance(out := tool_results[key], ToolImageOutput):
                self.add_input_image(out.image)
                tool_results_display[key] = out
                tool_results_text[key] = "Image shown to user"
            elif isinstance(out, str):
                try:
                    tool_results_text[key] = json.loads(out)
                except json.JSONDecodeError:
                    tool_results_text[key] = str(out)
            else:
                tool_results_text[key] = str(out)
        self.add_user_message(json.dumps({"tool_results": tool_results_text}))
        self.iterations += 1
        return AgentOutput(
            [ToolOutput(tool_results_display, tool_results_text, True)],
            response_dict,
        )
        # When this returns, the app will call the agent_loop again, as AUA is True

    def new_user_query(self) -> None:
        self.MAX_ITERATIONS += self.iterations
        self.new_query = True


class MetaStateAgent(Agent):
    state_key: str = (
        "BaseState"  # Corresponds to AutoState from AutoStateAgent
    )
    base_prompt = SYSTEM_PROMPT_META_STATE_AGENT
    states = meta_states

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.set_system_prompt()
        self.started = False

    def set_system_prompt(self, prompt: Optional[str] = None, state_key=None):
        if prompt is not None:
            warnings.warn(
                "Prompt should not be updated for MetaStateAgent externally, \
will be ignored.",
                category=RuntimeWarning,
            )
        if not state_key:
            state_key = self.state_key
        self.state_key = state_key
        if state_key == "MasterState":
            self.base_prompt = SYSTEM_PROMPT_STATELESS_AGENT
        else:
            self.base_prompt = SYSTEM_PROMPT_META_STATE_AGENT
        self.messages.system_message(
            content=self.base_prompt.format(
                state_details=self.states[self.state_key]
            )
        )

    async def agent_loop(self) -> AgentOutput:
        if not self.started:
            return await self._first_agent_loop()
        else:
            return await super().agent_loop()

    async def _first_agent_loop(self) -> AgentOutput:
        logger.info("MetaStateAgent started. Calling HyDE...")
        # Get to last user query
        self.original_query = self.messages.chat[-1]["content"]
        logger.debug(f"Original Query: {self.original_query}")
        # Calling HyDE
        tool_calls: list[LLMToolCall] = [
            {
                "name": "financial_rag_tool",
                "args": {
                    "query": self.original_query,
                    "top_k1": 20,
                    "top_k2": 10,
                    "n_similar": 2,
                },
            }
        ]
        response_dict: NonToolResponse = {
            "thought": "Calling financial_rag_tool, an advanced retriever with high quality responses",
            "audio": "Automatically calling financial_rag_tool...",
        }
        tool_results = await self.tool_handler.handle_tools(tool_calls)
        tool_results = tool_results[0]
        tool_results_text = {}
        tool_results_display = {}
        for key in tool_results:
            if isinstance(out := tool_results[key], ToolImageOutput):
                self.add_input_image(out.image)
                tool_results_display[key] = out
                tool_results_text[key] = "Image shown to user"
            elif isinstance(out, str):
                try:
                    tool_results_text[key] = json.loads(out)
                except json.JSONDecodeError:
                    tool_results_text[key] = str(out)
            else:
                tool_results_text[key] = str(out)
        self.add_user_message(json.dumps({"tool_results": tool_results_text}))
        self.started = True

        return AgentOutput(
            [ToolOutput(tool_results_display, tool_results_text, True)],
            response_dict,
        )
        # When this returns, the app will call the agent_loop again, as AUA is True


class HyFERAgent(MetaStateAgent):
    state_key = "MasterState"
    base_prompt = SYSTEM_PROMPT_HYFER
    states = hyfer_states

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_system_prompt()
        self.states[self.state_key].tools = [
            tool
            for tool in self.states[self.state_key].tools
            if not isinstance(tool, StateChangeTool)
        ]

    def set_system_prompt(self, prompt: str | None = None, state_key=None):
        if state_key is not None:
            warnings.warn(
                "State key should not be set for HyFERAgent",
                category=UserWarning,
            )
        self.base_prompt = prompt or self.base_prompt
        self.messages.system_message(
            self.base_prompt.format(
                state_details=self.states[self.state_key]
            )
        )


class MockAgent(Agent):
    state_key = "MockState"
    base_prompt = "You are a mock agent. Your state is {state_details}"
    states = mock_states

    def __init__(
        self,
    ) -> None:
        self.state_key = "MockState"
        super().__init__()
        self.set_system_prompt()

    def set_system_prompt(self, prompt: str | None = None, state_key=None):
        self.state_key = state_key or self.state_key
        self.base_prompt = prompt or self.base_prompt
        self.messages.system_message(
            content=self.base_prompt.format(
                state_details=self.states[self.state_key]
            )
        )
