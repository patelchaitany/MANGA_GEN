from angent import graph_init
from streamlit_cookies_manager import CookieManager
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import os

import json
import redis
import uuid
import logging
import time

cookies = CookieManager()
memory = MemorySaver()
max_retries = 1

def error(e):
    logging.error(f"An error occurred: {e}")
    st.error("An error occurred. Please try again.")
    try:
        redis_client.flushdb()
        logging.info("Redis cache cleared.")
    except Exception as e:
        logging.error(f"Error clearing Redis cache: {e}")

    st.stop()

try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True, socket_timeout=2)
        
except Exception as e:
    error(e)    


import asyncio
from search import perform_search, crawl_website

@st.cache_resource
def get_graph():
    workflow = graph_init()
    graph = workflow.compile(checkpointer=memory)
    return graph

def get_session_id():
    """Retrieve or create a unique session ID for the user."""
    try:
        query_params = st.query_params
        session_id = cookies.get("session_id",False)

        if not session_id:
            session_id = str(uuid.uuid4())

            cookies["session_id"] = session_id
        st.session_state.session_id = session_id
        return session_id
    except Exception as e:
        error(e)
if not cookies.ready():
    st.stop()

def load_chat_history(session_id):
    """Load chat history from Redis."""
    history = None
    try:
        history = redis_client.get(session_id)
    except Exception as e:
        error(e)
    return json.loads(history) if history else {}

def save_chat_history(session_id, messages):
    """Save chat history to Redis."""
    
    
    msg = json.dumps(messages)
    print(f"msg is {msg}")
    
    serializable_prompts = [
        {
            "character_or_scene_name": prompt.character_or_scene_name,
            "prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt
        }
        for prompt in st.session_state.get("stable_diffusion_prompts", [])
    ]
    
    prompts = json.dumps(serializable_prompts)
    
    data = {
        "messages": msg,
        "stable_diffusion_prompts": prompts
    }
    data = json.dumps(data)
    redis_client.set(session_id,data)



async def run_workflow(inputs, session_id, config):
    try:
        with st.status(label="Working", expanded=False, state="running") as st.session_state.status:
            st.session_state.placeholder = st.empty()
            
            # Attempt to invoke the workflow
            try:
                value = await st.session_state.workflow.ainvoke(inputs, config)
                artist_prompts = value.get("stable_diffusion_prompts",[])
                value = value['messages'][-1].content
                st.session_state.stable_diffusion_prompts = artist_prompts
               
            except KeyError as e:
                error(e)
            except Exception as e:
                error(e)
            
        rsp = {"role": "assistant", "content": value}
        st.session_state.placeholder.empty()
        st.session_state.status.update(label="**---FINISH---**", state="complete", expanded=False)
        
        st.session_state.messages.append(rsp)
        save_chat_history(session_id, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(value)
    
    except Exception as e:
        error(e)


def main():
    print(f"hello1")
    st.title("Conversational Agent Chatbot")

    # Get or create session ID
    session_id = get_session_id()

    # Load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = json.loads(load_chat_history(session_id).get("messages","[]"))
        st.session_state.stable_diffusion_prompts = json.loads(load_chat_history(session_id).get("stable_diffusion_prompts","[]"))
    
    if 'status_container' not in st.session_state:
        st.session_state.status_container = st.empty()

    if not hasattr(st.session_state, "workflow"):
        graph = get_graph()
        st.session_state.workflow = graph
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:

        # Add user input to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process input through the workflow
        config = {"configurable": {"thread_id": session_id}}
        input = {"messages": [{"role": "user", "content": user_input}]}
        asyncio.run(run_workflow(input,session_id,config))
            #for value in graph.stream({"messages": [("user", user_input)]}, config,stream_mode="values"):
            #    rsp = value['messages'][-1]
            #    st.session_state.messages.append({"role": "assistant", "content": rsp})
            #    with st.chat_message("assistant"):
            #        st.markdown(rsp)
            # Save updated chat history in Redis

    if "stable_diffusion_prompts" in st.session_state and st.session_state.stable_diffusion_prompts:
        with st.sidebar:
            st.header("Stable Diffusion Prompts")
            if isinstance(st.session_state.stable_diffusion_prompts, list):
                for prompt_data in st.session_state.stable_diffusion_prompts:
                    if isinstance(prompt_data, dict):
                        st.subheader(prompt_data.get("character_or_scene_name", "N/A"))
                        st.markdown(f"**Prompt:** {prompt_data.get('prompt', 'N/A')}")
                        st.markdown(f"**Negative Prompt:** {prompt_data.get('negative_prompt', 'N/A')}")
                    elif hasattr(prompt_data, 'character_or_scene_name'):
                        st.subheader(prompt_data.character_or_scene_name)
                        st.markdown(f"**Prompt:** {prompt_data.prompt}")
                        st.markdown(f"**Negative Prompt:** {prompt_data.negative_prompt}")
                    else:
                        st.markdown("Invalid prompt format")
                    st.divider()
            elif isinstance(st.session_state.stable_diffusion_prompts, dict):
                st.markdown(st.session_state.stable_diffusion_prompts)
            else:
                 st.markdown("Invalid prompt format")



if __name__ == "__main__":
    print(f"hello")
    main()
