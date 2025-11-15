import asyncio
import logging
import os
import warnings
from typing import Optional, cast, Mapping
import json

import litellm
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from FinAgent.agents import (
    MultiStateAgent,
    StatelessAgent,
    Agent,
    MockAgent,
    AutoAgent,  # noqa: F401
    MetaStateAgent,
    ExplainabilityAgent,
    HyFERAgent
)
from FinAgent.guardrails.guardrail_api import FinGuard
from FinAgent.utils.markdown_handler import escape_dollars
# from FinAgent.config.states import s

load_dotenv()
litellm.set_verbose = False  # type: ignore
litellm.logging = False
os.environ["USE_GUARDRAIL"] = os.getenv("USE_GUARDRAIL", "False")
USE_GUARDRAIL = os.environ["USE_GUARDRAIL"] == "True"
guardrail = FinGuard()

# Part of the ongoing attempt to fix the duplicate runs
logger = logging.getLogger("app")

logger.debug("--------------- Starting -------------------")

config = {"Agent": HyFERAgent}


if "initialized" not in st.session_state or not st.session_state.initialized:
    logger.debug("Initializing agent")
    st.session_state.agent = cast(Agent, config["Agent"]())
    logger.info(f"{st.session_state.agent.model.input_params}")
    st.session_state.messages = []
    st.session_state.shared_messages = st.session_state.agent.messages.chat
    st.session_state.initialized = True
    st.session_state.AUA = cast(bool, True)

agent = st.session_state.agent


def reset():
    """
    Reset the agent and clear the chat history.
    Doesn't reset the system prompt if you have changed it.
    """
    st.session_state.messages = []
    st.session_state.agent = config["Agent"]()
    st.session_state.explainability_messages = []
    st.session_state.explainability_agent = ExplainabilityAgent()


with st.sidebar:
    img_file = None

    reset_button = st.button("Reset", on_click=reset)
    if isinstance(agent, (MultiStateAgent, StatelessAgent, MockAgent)):
        new_system_prompt = st.text_area(
            "Change System Prompt", value=agent.base_prompt, height=300
        )
    else:
        new_system_prompt = agent.base_prompt
        # Remaining agents handle system prompt very differently
    state_info_container = st.container(height=200).empty()
    with state_info_container:
        state_instructions = st.code(
            json.dumps(agent.states[agent.state_key].to_dict(), indent=2),
            language="json",
        )
    thoughts = st.expander("Thoughts")

if new_system_prompt != agent.base_prompt:
    agent.set_system_prompt(new_system_prompt)


def display_chat_message(
    role: str,
    content: str | Mapping[str, str | list[str]] | Image.Image,
) -> None:
    if isinstance(content, Image.Image):
        with st.chat_message(role):
            st.image(content, use_column_width=True)
    elif role == "Assistant":
        assert isinstance(content, dict)
        content = (content).copy()
        with st.chat_message("Assistant"):
            st.markdown(escape_dollars(content.pop("audio")))
            with st.sidebar:
                with thoughts:
                    if len(content) > 1:
                        st.code(json.dumps(content, indent=2), language="json")
                    elif len(content) == 1:
                        st.text(list(content.values())[0])
    elif role == "Tool":
        if content:
            st.code(json.dumps(content, indent=2), language="json")
        
    elif role == "User":
        with st.chat_message("User"):
            st.markdown(content)
    else:
        with st.chat_message(role):
            st.text(content)
        warnings.warn(f"Unknown role: {role}")


st.title("SECure RAG")
st.write(
    "Welcome to SECure RAG, your friendly conversational assistant for Finance-related queries."
)

for user_input in st.session_state.messages:
    display_chat_message(user_input["role"], user_input["content"])


image_input: Optional[Image.Image] = Image.open(img_file) if img_file else None
user_input = st.chat_input("Please enter your query...")
if image_input:
    agent.add_input_image(image_input)  # NOTE: Must be done before query
    st.session_state.messages.append({"role": "User", "content": image_input})

if user_input:
    # Check the user input for safety using FinGuard
    agent.new_user_query()
    safety_result = asyncio.run(guardrail.check(user_input, USE_GUARDRAIL))
    if not safety_result["is_safe"]:
        with st.chat_message("Guardrail"):
            st.markdown("Warning! The query contains unsafe content.")
        st.session_state.messages.append(
            {
                "role": "Guardrail",
                "content": f"Warning! The query contains unsafe content.\n{safety_result["reason"]}",
            }
        )

    # Continue to process the input regardless of safety
    display_chat_message("User", user_input)
    st.session_state.messages.append({"role": "User", "content": user_input})
    c = 0
    agent.add_user_message(user_input)
    logger.debug("agent.add_user_message(user_input)")
    previous_state_key = "BaseState"
    # agent
    while True:
        print(f"Iteration {c}")
        logger.debug(f"Iteration: {c}")
        if c > 10:
            st.write("Too many iterations, breaking")
            break
        state_key = agent.state_key
        if st.session_state.AUA:
            logger.debug("Before agent.agent_loop()")
            try:
                with st.empty():
                    spinner_name = (
                        f"Calling {agent.model.model_name} \
                                    and its requested tools..."
                        if not (
                            (
                                agent.state_key == "AutoState"
                                and previous_state_key != agent.state_key
                            )
                            or (
                                isinstance(agent, MetaStateAgent)
                                and not agent.started
                            )
                        )
                        else "Calling HyDE..."
                    )
                    with st.spinner(spinner_name):
                        assistant_and_tool_responses = asyncio.run(
                            agent.agent_loop()
                        )
            except RuntimeError as e:
                logger.exception(f"Uncaught error {e} in agent_loop. \
                                 This may be a bug in the code.")
                st.write(f"Error: {str(e)}")
                break

            state_key = agent.state_key
            display_chat_message(
                "Assistant", assistant_and_tool_responses.remaining_response
            )
            st.session_state.messages.append(
                {
                    "role": "Assistant",
                    "content": assistant_and_tool_responses.remaining_response,
                }
            )
            for image, text, AUA in assistant_and_tool_responses.tool_outputs:
                logger.debug("After agent.agent_loop()")
                if image:
                    for img in image.values():
                        display_chat_message("Tool", img.image)
                        st.session_state.messages.append(
                            {"role": "Tool", "content": img.image}
                        )
                display_chat_message("Tool", text)
                st.session_state.messages.append(
                    {"role": "Tool", "content": text}
                )
                st.session_state.AUA = AUA
                previous_state_key = state_key
                if agent.state_key != state_key:
                    with state_info_container:
                        state_instructions = st.code(
                            json.dumps(
                                agent.states[agent.state_key].to_dict(),
                                indent=2,
                            ),
                            language="json",
                        )
                    state_key = agent.state_key
            logger.debug(f"Iteration {c} over")
            c += 1
            continue
        else:
            break
    st.session_state.AUA = True
