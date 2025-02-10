from src.agents.angent import graph_init
from streamlit_cookies_manager import CookieManager
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import hashlib
import json
import redis
import uuid
import logging
import time
import requests
from termcolor import colored
import asyncio
from src.search_utils.search import perform_search, crawl_website
from src.search_utils.image_search import pdq_hash, DocumentIndex,perform_search_and_filter,fetch_image,check_url_validity


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


@st.cache_resource
def get_graph():
    workflow = graph_init()
    graph = workflow.compile(checkpointer=memory)
    return graph

def get_session_id():
    """Retrieve or create a unique session ID for the user."""
    try:
        query_params = st.query_params
        session_id = cookies.get("session_id", False)

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
    serializable_prompts = []
    for prompt_image in st.session_state.get("stable_diffusion_prompts", []):
        if hasattr(prompt_image, 'prompt') and hasattr(prompt_image, 'image_urls'):
            serializable_prompts.append({
                "prompt": {
                    "character_or_scene_name": prompt_image.prompt.character_or_scene_name,
                    "prompt": prompt_image.prompt.prompt,
                    "negative_prompt": prompt_image.prompt.negative_prompt,
                    "search_query": prompt_image.prompt.search_query
                },
                "image_urls": prompt_image.image_urls
            })
    prompts = json.dumps(serializable_prompts)
    data = {
        "messages": msg,
        "stable_diffusion_prompts": prompts
    }
    data = json.dumps(data)
    redis_client.set(session_id, data)

async def run_workflow(inputs, session_id, config):
    try:
        with st.status(label="Working", expanded=False, state="running") as st.session_state.status:
            st.session_state.placeholder = st.empty()
            try:
                workflow_result = await st.session_state.workflow.ainvoke(inputs, config)
                # Get the prompt_images which contains both prompts and their image URLs
                prompt_images = workflow_result.get("prompt_images", [])
                # Store the prompt_images in session state
                st.session_state.stable_diffusion_prompts = prompt_images
                # Get the last message content
                value = workflow_result['messages'][-1].content
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

def calculate_image_hash(image_bytes):
    """Calculate the hash of an image."""
    return hashlib.md5(image_bytes).hexdigest()


def load_images_from_redis():
    """Load image URLs from Redis."""
    try:
        image_urls = redis_client.lrange("image_urls", 0, -1)  # Retrieve all URLs from the list
        return image_urls
    except Exception as e:
        logging.error(f"Error loading image URLs from Redis: {e}")
        return []

async def main():
    num_columns = 3
    # Load image URLs from Redis
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Conversational Agent Chatbot")
    with col2:
        # Toggle for search functionality
        search_mode = st.toggle("Search Mode", key="search_mode_toggle", label_visibility="visible")

    # Get or create session ID
    session_id = get_session_id()

    # Load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = json.loads(load_chat_history(session_id).get("messages", "[]"))
        st.session_state.stable_diffusion_prompts = json.loads(load_chat_history(session_id).get("stable_diffusion_prompts", "[]"))

    if 'status_container' not in st.session_state:
        st.session_state.status_container = st.empty()

    if not hasattr(st.session_state, "workflow"):
        graph = get_graph()
        st.session_state.workflow = graph

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if search_mode:
        search_query = st.chat_input("Search for images...")
        if search_query:
            valid_image_urls = await perform_search_and_filter(search_query) # Now returns URLs
            #print(colored(f"valid_images URLS: {valid_image_urls}", "blue")) # Debug: Print valid_images URLs
            num_columns = 3 # Define number of columns for display
            cols = st.columns(num_columns)
            async def display_image(i, image_url, cols):
                img_bytes = await fetch_image(image_url)
                if img_bytes:
                    img = Image.open(img_bytes)
                    with cols[i % num_columns]:
                        st.image(img, use_container_width =True)

            tasks = [display_image(i, url, cols) for i, url in enumerate(valid_image_urls)]
            await asyncio.gather(*tasks)
    else:
        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            config = {"configurable": {"thread_id": session_id}}
            input = {"messages": [{"role": "user", "content": user_input}]}
            await run_workflow(input, session_id, config)

    if "stable_diffusion_prompts" in st.session_state and st.session_state.stable_diffusion_prompts:
        with st.sidebar:
            st.header("Generated Prompts & Images")
            if isinstance(st.session_state.stable_diffusion_prompts, list):
                for prompt_data in st.session_state.stable_diffusion_prompts:
                    if hasattr(prompt_data, 'prompt') and hasattr(prompt_data, 'image_urls'):
                        # Display prompt information
                        st.subheader(prompt_data.prompt.character_or_scene_name)
                        with st.expander("Show Prompt Details"):
                            st.markdown(f"**Prompt:** {prompt_data.prompt.prompt}")
                            if prompt_data.prompt.negative_prompt:
                                st.markdown(f"**Negative Prompt:** {prompt_data.prompt.negative_prompt}")
                            if prompt_data.prompt.search_query:
                                st.markdown(f"**Search Query:** {prompt_data.prompt.search_query}")
                        
                        # Display associated images
                        if prompt_data.image_urls:
                            st.markdown("### Found Images:")
                            cols = st.columns(2)
                            async def display_image(url, col):
                                try:
                                    img_bytes = await fetch_image(url)
                                    if img_bytes:
                                        img = Image.open(img_bytes)
                                        col.image(img, use_container_width=True)
                                except Exception as e:
                                    col.error(f"Failed to load image: {str(e)}")
                            
                            # Display first 4 images (2 rows of 2 images)
                            for i, url in enumerate(prompt_data.image_urls[:4]):
                                asyncio.create_task(display_image(url, cols[i % 2]))
                    else:
                        st.markdown("Invalid prompt-image pair format")
                    st.divider()
            else:
                st.markdown("Invalid prompt format")

if __name__ == "__main__":
    asyncio.run(main())
