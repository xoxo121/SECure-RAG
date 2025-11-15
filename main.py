import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import cast
import logging.config
import os
from dotenv import load_dotenv
from FinAgent.agents import (
    MultiStateAgent,
    ExplainabilityAgent,
    Agent,
    StatelessAgent,
    MetaStateAgent,
    HyFERAgent,
)
from FinAgent.guardrails.guardrail_api import FinGuard
import yaml

# Load environment variables
load_dotenv()

if not os.path.exists(".logs"):
    os.mkdir(".logs")
if os.path.exists:
    logging.config.dictConfig(
        yaml.load(open("logging.yaml"), Loader=yaml.FullLoader)
    )

# Initialize logging
logger = logging.getLogger("main")

# Guardrail configuration
USE_GUARDRAIL = os.getenv("USE_GUARDRAIL", "False").lower() == "true"
guardrail = FinGuard()

# Initialize FastAPI app

app = FastAPI()
# Initialize agents
agent_set = {
    "MultiStateAgent": MultiStateAgent,
    "ExplainabilityAgent": ExplainabilityAgent,
    "StatelessAgent": StatelessAgent,
    "MetaStateAgent": MetaStateAgent,
    "HyFERAgent": HyFERAgent,
}
config = {
    "Agent": agent_set["HyFERAgent"],
}
app.state.agent = cast(Agent, config["Agent"]())
explainability_agent: Agent = ExplainabilityAgent()


# Request and Response Schemas
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    messages: list[dict]
    iterations: int


class ExplainabilityRequest(BaseModel):
    query: str


class ExplainabilityResponse(BaseModel):
    explanation: str


@app.put("/reset")
async def reset_agent():
    """
    Reset the agent and clear the chat history.
    Doesn't reset the system prompt if you have changed it.
    """
    app.state.agent = config["Agent"]()


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to handle user queries and return the agent's response.
    """
    user_input = request.query
    safety_check = await guardrail.check(user_input, check=USE_GUARDRAIL)
    if not safety_check["is_safe"]:
        return QueryResponse(
            messages=[
                {
                    "role": "Guardrail",
                    "content": f"Warning! The query contains unsafe content.\n{safety_check["reason"]}",
                }
            ],
            iterations=0,
        )
    logger.debug(f"Received query: {user_input}")
    agent = cast(Agent, app.state.agent)
    iterations = 0
    chat_history: list[dict[str, str | dict]] = []
    agent.add_user_message(user_input)
    chat_history.append({"role": "User", "content": user_input})
    AUA = False
    while iterations < 10:
        logger.debug(f"Iteration: {iterations}")

        # Call the agent loop
        assistant_and_tool_responses = await agent.agent_loop()
        chat_history.append(
            {
                "role": "Assistant",
                "content": assistant_and_tool_responses.remaining_response[
                    "audio"
                ],
            }
        )
        # Append responses
        for image, text, AUA in assistant_and_tool_responses.tool_outputs:
            if text:
                chat_history.append({"role": "Tool", "content": text})
            if image:
                warnings.warn("Image output not supported yet.")
        # Stop if no further updates are allowed
        if not AUA:
            break

        iterations += 1

    return QueryResponse(
        messages=chat_history,
        iterations=iterations,
    )


@app.post("/explain", response_model=ExplainabilityResponse)
async def explain_query(request: ExplainabilityRequest):
    """
    Endpoint to handle explanation queries using the ExplainabilityAgent.
    """
    user_exp = request.query
    app.state.agent = cast(Agent, app.state.agent)

    context = "\n".join(
        [
            str(msg["content"])
            for msg in app.state.agent.messages.chat
            if msg.get("role") == "user"
        ]
    )
    # Add user input to the explainability agent
    try:
        explainability_agent.add_user_message(
            f"Context {context} \n user_input {user_exp}"
        )
        explanation = await explainability_agent.get_assistant_response()
        return ExplainabilityResponse(explanation=explanation)
    except Exception as e:
        logger.error(
            f"Error in explanation processing: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error in processing explanation."
        ) from e
