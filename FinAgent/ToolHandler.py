import logging
import warnings
import asyncio
from asyncio import iscoroutinefunction
from typing import Any, Sequence, TYPE_CHECKING, TypedDict

from .schema import BaseTool
from .tools.state_change_tool import StateChangeTool

DEBUG = True
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .agents import Agent

class LLMToolCall(TypedDict):
    """
    A dictionary representing a tool call from the LLM. (name and args)
    """
    name: str
    args: dict[str, Any]
class ToolHandler:
    def __init__(
        self,
        tools: list[BaseTool],
        agent: 'Agent',
    ):
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.agent = agent

    async def call_tool(self, tool_name: str, args):
        logger.info(f"{tool_name = }\n{args = }")
        try:
            tool_result = (
                self.tools[tool_name].run(**args)
                if not iscoroutinefunction(self.tools[tool_name].run)
                else await self.tools[tool_name].run(**args)
            )
            return tool_result
        except Exception as e:
            logger.error(
                f"Error occurred while running tool: {tool_name}\n{args = }",
                exc_info=True,
            )
            error_msg = f"The following error occurred while running the tool: {tool_name} error: {e}. Please check the tool arguments or use some other tool to accomplish the task and try again."
            return error_msg
        

    async def handle_tool(self, llm_tool_call: LLMToolCall) -> dict:
        tool_results = {}
        tool_name = llm_tool_call["name"]
        args = llm_tool_call["args"]
        if tool_name not in self.tools:
            tool_results = {
                tool_name: f"Tool {tool_name} does not exist in this state.",
                "AUA": True,
            }
            return tool_results
        if args.keys() != self.tools[tool_name].args.keys():
            tool_results = {
                tool_name: f"Arguments for tool {tool_name} are incorrect.",
                "AUA": True,
            }
            logger.error(f"Arguments for tool are incorrect. {llm_tool_call}")
            return tool_results
        if isinstance(self.tools[tool_name], StateChangeTool):
            args["next_state"] = str(args["next_state"]).strip()
            if args["next_state"] not in self.agent.states:
                tool_results = {
                    tool_name: f"State {args['next_state']} does not exist.",
                    "AUA": True,
                }
                return tool_results
            self.agent.set_system_prompt(state_key=args["next_state"])
            tools = self.agent.states[args["next_state"]].tools
            self.agent.model = self.agent.states[args["next_state"]].model
            logger.info(f"State changed successfully to {args['next_state']}")
            self.tools = {tool.name: tool for tool in tools}
            tool_output = (
                f"State changed successfully to {args["next_state"]}"
            )
        else:
            tool_output = await self.call_tool(tool_name, args)  # type of tool_output is the same as tool.run

        if self.tools[tool_name].tool_type == "AUA":
            tool_results["AUA"] = True
        tool_results[tool_name] = tool_output
        return tool_results

    async def handle_tools(self, llm_tool_calls: list[LLMToolCall]) -> list[dict]:
        return await asyncio.gather(*[self.handle_tool(tool) for tool in llm_tool_calls])
