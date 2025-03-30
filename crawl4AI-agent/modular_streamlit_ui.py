"""
Dynamic Streamlit UI that loads agents from configuration.
This UI adapts automatically when new agents are added to the configuration.
"""

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import the agent factory
from agent_factory import create_agent, get_available_agents, AgentDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str, agent_key: str):
    """
    Run the agent with streaming text for the user_input prompt.
    
    Args:
        user_input: The user's question or query
        agent_key: Key of the agent to use
    """
    # Initialize session state for the selected agent if not present
    if f"{agent_key}_messages" not in st.session_state:
        st.session_state[f"{agent_key}_messages"] = []
    
    # Get the messages for the current agent
    messages = st.session_state[f"{agent_key}_messages"]
    
    # Get agent configuration
    all_agents = get_available_agents()
    agent_config = all_agents[agent_key]
    
    # Determine source tags for this agent
    if "source_tags" in agent_config:
        source_tags = agent_config["source_tags"]
    else:
        source_tags = [agent_config["source_tag"]]
    
    # Get the agent
    expert = create_agent(agent_key)
    
    # Prepare dependencies
    deps = AgentDeps(
        supabase=supabase,
        openai_client=openai_client,
        source_tags=source_tags
    )

    # Run the agent in a stream
    async with expert.run_stream(
        user_input,
        deps=deps,
        message_history=messages[:-1] if messages else [],  # pass conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        messages.extend(filtered_messages)

        # Add the final response to the messages
        messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )
        
        # Update the session state
        st.session_state[f"{agent_key}_messages"] = messages


async def main():
    st.title("Documentation AI Expert Hub")
    
    # Get all available agents
    all_agents = get_available_agents()
    
    # Create two tabs: one for individual and one for combined
    tab1, tab2 = st.tabs(["Individual Experts", "Combined Experts"])
    
    with tab1:
        # Filter individual agents
        individual_agents = {k: v for k, v in all_agents.items() if k in all_agents and "source_tag" in v}
        
        # Create selection for individual agents
        individual_options = [agent["display_name"] for key, agent in individual_agents.items()]
        if individual_options:
            individual_selection = st.selectbox(
                "Select Individual Expert:",
                individual_options,
                key="individual_expert_selection"
            )
            
            # Get the key for the selected agent
            selected_individual_key = next(
                key for key, agent in individual_agents.items() 
                if agent["display_name"] == individual_selection
            )
        else:
            st.warning("No individual agents configured.")
            selected_individual_key = None
    
    with tab2:
        # Filter combined agents
        combined_agents = {k: v for k, v in all_agents.items() if k in all_agents and "source_tags" in v}
        
        # Create selection for combined agents
        combined_options = [agent["display_name"] for key, agent in combined_agents.items()]
        if combined_options:
            combined_selection = st.selectbox(
                "Select Combined Expert:",
                combined_options,
                key="combined_expert_selection"
            )
            
            # Get the key for the selected agent
            selected_combined_key = next(
                key for key, agent in combined_agents.items() 
                if agent["display_name"] == combined_selection
            )
        else:
            st.warning("No combined agents configured.")
            selected_combined_key = None
    
    # Determine which agent is active based on which tab is active
    if st.session_state.get("active_tab") is None:
        st.session_state["active_tab"] = "individual"
    
    # Add tab buttons for easy switching
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Individual Expert", type="primary" if st.session_state["active_tab"] == "individual" else "secondary"):
            st.session_state["active_tab"] = "individual"
            st.rerun()
    with col2:
        if st.button("Use Combined Expert", type="primary" if st.session_state["active_tab"] == "combined" else "secondary"):
            st.session_state["active_tab"] = "combined"
            st.rerun()
    
    # Get the selected agent key based on the active tab
    if st.session_state["active_tab"] == "individual":
        selected_agent_key = selected_individual_key
        if selected_agent_key:
            selected_agent = individual_agents[selected_agent_key]
        else:
            st.warning("Please select an individual expert.")
            return
    else:
        selected_agent_key = selected_combined_key
        if selected_agent_key:
            selected_agent = combined_agents[selected_agent_key]
        else:
            st.warning("Please select a combined expert.")
            return
    
    # Display agent info
    st.subheader(f"Active Expert: {selected_agent['display_name']}")
    st.markdown(selected_agent["description"])
    
    # Show source info
    if "source_tags" in selected_agent:
        st.markdown(f"*This expert searches across: {', '.join([tag.replace('_docs', '').title() for tag in selected_agent['source_tags']])}*")
    else:
        st.markdown(f"*This expert specializes in: {selected_agent['source_tag'].replace('_docs', '').title()}*")
    
    st.markdown("---")
    
    # Initialize message history in session state if not present
    if f"{selected_agent_key}_messages" not in st.session_state:
        st.session_state[f"{selected_agent_key}_messages"] = []

    # Display all messages from the conversation for the selected agent
    for msg in st.session_state[f"{selected_agent_key}_messages"]:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input(selected_agent["placeholder"])

    if user_input:
        # We append a new request to the conversation explicitly
        if f"{selected_agent_key}_messages" not in st.session_state:
            st.session_state[f"{selected_agent_key}_messages"] = []
            
        st.session_state[f"{selected_agent_key}_messages"].append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input, selected_agent_key)


if __name__ == "__main__":
    asyncio.run(main())