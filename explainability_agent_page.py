import asyncio
import streamlit as st
from FinAgent.agents import ExplainabilityAgent

# Initialize explainability agent
if "explainability_initialized" not in st.session_state:
    st.session_state.explainability_agent = ExplainabilityAgent()
    st.session_state.explainability_agent.set_system_prompt(
        prompt="""
Your task is to explain the user's answer based on the context provided above.
- The user input is answer given by agent by usage of tools and the respective tool results are also provided
- You have to explain from which parts of the tool results the answer was derived from
- Quote exact statements/facts which support the answer provided as user input
- If there are multiple statements supporting the answer, cite all
- Anything that you cite should have the source and additional identifying features such as an id or number for ease of verification
- When citing a source, quote from all relevant sources, highlighting identifying features
- If available quote the source of the statements like tool name, document name or website name
- Do not provide any new information which is not present in the tool results"""
    )
    st.session_state.explainability_messages = []
    st.session_state.explainability_initialized = True

# Title and description
st.title("Explainability Agent")
st.write(
    "Welcome to the Explainability Agent. Paste a part of the response from the Finance Agent for further clarification."
)

# Access shared messages from Finance Agent
shared_messages = st.session_state.get("shared_messages", [])
if shared_messages:
    with st.expander("See transferred messages for reference"):
        for i, msg in enumerate(shared_messages, start=1):
            if msg['role']=='system':
                continue
            if msg['role']=='assistant':
                continue
            if msg['content'].startswith("{"):
                st.markdown(f"**Message {i//2}:**")
                content='{"tool_results": '+msg['content'][18:-2]+"}"
                st.code(content,language='json')
            else:
                st.markdown(f"**Message {i//2}:** {msg['content']}")


# User input for explainability
user_input = st.text_area(
    "Paste the part of the response you want explained:",
    placeholder="Paste text here...",
)

# Handle explanation request
if user_input:
    st.session_state.explainability_messages.append(
        {"role": "User", "content": user_input}
    )
    explainability_agent = (
        st.session_state.explainability_agent
    )  # Placeholder for the actual model

    # Simulate agent response
    context = "\n".join(
        [
            str(msg["content"])  # Ensure content is a string
            for msg in shared_messages
            if msg["role"] == "user"
        ]
    )

    # Create the LLM prompt
    prompt = f"""
    Context from previous conversation:
    {context}

    User's question for clarification:
    {user_input}

    Your task is to explain the user's question using the context provided above. Make the explanation simple and concise.
    """

    # Call the LLM (using litellm)
    explainability_agent = st.session_state.explainability_agent
    explainability_agent.add_user_message(
        prompt
    )
    explanation = asyncio.run(explainability_agent.get_assistant_response())
    st.session_state.explainability_messages.append(
        {"role": "Tool", "content": explanation}
    )

    # Extract and display the response
    st.write(f"**Explanation:** {explanation}")
