from angent import graph_init
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
from search import perform_search, crawl_website
from test import pdq_hash, DocumentIndex


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
    redis_client.set(session_id, data)

async def run_workflow(inputs, session_id, config):
    try:
        with st.status(label="Working", expanded=False, state="running") as st.session_state.status:
            st.session_state.placeholder = st.empty()
            try:
                value = await st.session_state.workflow.ainvoke(inputs, config)
                artist_prompts = value.get("stable_diffusion_prompts", [])
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

def calculate_image_hash(image_bytes):
    """Calculate the hash of an image."""
    return hashlib.md5(image_bytes).hexdigest()

lock = asyncio.Lock()  # Create an async lock

async def perform_search_and_filter(query):
    valid_image_urls = []  # List to store valid image URLs
    unique_hashes = set()  # Set to track unique image hashes
    """Perform search and filter valid image URLs and fetch image bytes."""
    results = await perform_search(query)

    index = DocumentIndex()  # Create an instance of DocumentIndex

    async def process_item(item):
        url = item['link']
        try:
            print(colored(f"url : {url}","green"))
            response = await asyncio.to_thread(requests.get, url, allow_redirects=True, timeout=1)
            if response.status_code == 200:
                img_bytes = BytesIO(response.content)
                try:
                    img = Image.open(img_bytes)  # Verify it's a valid image
                    image_hash, perms = pdq_hash(img_bytes.getvalue())  # Get hash and permutations
                    if await index.add_document(url, perms):  # Check for uniqueness
                        valid_image_urls.append(url)  # Store URL only
                        logging.info(f"Valid and unique image URL found: {url}")
                    else:
                        logging.info(f"Duplicate image found, skipping: {url}")
                except UnidentifiedImageError:
                    logging.warning(f"Invalid image format for URL: {url}")
            else:
                logging.warning(f"Invalid image URL: {url}")
        except requests.RequestException:
            logging.warning(f"Invalid image URL: {url}")

    tasks = [process_item(item) for item in results]
    await asyncio.gather(*tasks)
    logging.info("Valid image URLs saved to Redis.")
    return valid_image_urls  # Return URLs


def check_url_validity(url):

    try:
        response = requests.head(url, allow_redirects=True, timeout=1)
        #print(colored(f"url {url} {response}","green"))
        return response.status_code == 200
    except requests.RequestException:
        return False

def load_images_from_redis():
    """Load image URLs from Redis."""
    try:
        image_urls = redis_client.lrange("image_urls", 0, -1)  # Retrieve all URLs from the list
        return image_urls
    except Exception as e:
        logging.error(f"Error loading image URLs from Redis: {e}")
        return []

async def fetch_image(url):
    """Fetch image bytes from URL asynchronously."""
    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            logging.warning(f"Failed to fetch image from {url}: Status code {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.warning(f"Error fetching image from {url}: {e}")
        return None

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
            asyncio.run(run_workflow(input, session_id, config))

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
    asyncio.run(main())
