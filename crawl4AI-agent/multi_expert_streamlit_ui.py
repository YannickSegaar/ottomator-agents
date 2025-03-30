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
# Import both expert agents
from mcp_expert import mcp_expert, MCPDeps
from myfitnesspal_expert import myfitnesspal_expert, MyFitnessPalDeps

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


async def run_agent_with_streaming(user_input: str, expert_type: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    
    Args:
        user_input: The user's question or query
        expert_type: Which expert to use ('mcp' or 'myfitnesspal')
    """
    # Initialize session state for the selected expert if not present
    if f"{expert_type}_messages" not in st.session_state:
        st.session_state[f"{expert_type}_messages"] = []
    
    # Get the messages for the current expert
    messages = st.session_state[f"{expert_type}_messages"]
    
    # Select the correct expert and dependencies
    if expert_type == "mcp":
        expert = mcp_expert
        deps = MCPDeps(
            supabase=supabase,
            openai_client=openai_client
        )
    else:  # myfitnesspal
        expert = myfitnesspal_expert
        deps = MyFitnessPalDeps(
            supabase=supabase,
            openai_client=openai_client
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
        st.session_state[f"{expert_type}_messages"] = messages


async def main():
    st.title("AI Documentation Expert")
    
    # Expert selection
    expert_type = st.sidebar.radio(
        "Select Documentation Expert:",
        ["Model Context Protocol (MCP)", "MyFitnessPal API"],
        key="expert_selection"
    )
    
    # Convert selection to key
    expert_key = "mcp" if expert_type == "Model Context Protocol (MCP)" else "myfitnesspal"
    
    # Display expert info
    if expert_key == "mcp":
        st.write("Ask any question about the Model Context Protocol (MCP), its specifications, implementation details, or best practices.")
    else:
        st.write("Ask any question about the MyFitnessPal API, its endpoints, authentication methods, data models, and usage examples.")

    # Initialize message history in session state if not present
    if f"{expert_key}_messages" not in st.session_state:
        st.session_state[f"{expert_key}_messages"] = []

    # Display all messages from the conversation for the selected expert
    for msg in st.session_state[f"{expert_key}_messages"]:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    placeholder_text = "What questions do you have about the " + (
        "Model Context Protocol?" if expert_key == "mcp" else "MyFitnessPal API?"
    )
    user_input = st.chat_input(placeholder_text)

    if user_input:
        # We append a new request to the conversation explicitly
        if f"{expert_key}_messages" not in st.session_state:
            st.session_state[f"{expert_key}_messages"] = []
            
        st.session_state[f"{expert_key}_messages"].append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input, expert_key)


if __name__ == "__main__":
    asyncio.run(main())