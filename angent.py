from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, RemoveMessage,AIMessage,HumanMessage,BaseMessage,FunctionMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from typing import List,Annotated,Sequence,Dict,Any,Optional, Literal
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.utils.function_calling import format_tool_to_openai_function
import operator
from langchain.agents import Tool,create_openai_functions_agent
from langchain_core.tools import tool
import functools
from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation
import json
from langgraph.prebuilt import ToolNode 
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from typing_extensions import TypedDict
from langchain_groq import ChatGroq

from lang_work import create_agent,tool_node,summarize_conversation,delete_messages,supervisor
from graph_state import State,Responce,AgentConfig,artist,artist_responce
from agent_tool import Tools_Executor,Tool_list
from prompt import writer_prompt,critic_prompt,genral_assistance,artist_prompt
from termcolor import colored
from langchain_ollama import OllamaLLM 
from langsmith import Client
import os
from utils import calculate_levenshtein_similarity, word_by_word_levenshtein
from context_agent import get_highest_relevance_messages
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Replace hardcoded values with environment variables
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")

client = Client()

#llm = OllamaLLM(model="llama3",temperature=0)
memory = MemorySaver()
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)
model_large = llm

def create_genral_agent(llm, system_message: str, name: str, examples="", tools=None, schema=None):
    agent = create_agent(llm, system_message=system_message, examples=examples, tools=tools, schema=schema)
    
    def genral_agent_node(state) -> Command[Literal["supervisor"]]:

        messages = state["genral_assistance_"]
        messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["genral_assistance_"])
        all_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["messages"])
        conversation = [{"role": "user", "content": "Here is the conversation with user : \n"}]
        
        #print(colored(f"genral_agent_node \n ---------\n {messages}\n--------\n", "yellow"))
        #print(colored(f"genral_agent_node \n ---------\n {all_messages}\n--------\n", "red"))
        conversation = conversation + all_messages
        
        prompt_messages = [
            *messages,
            *conversation,
            {"role": "user", "content": f"user request : {state['user_request']} \n"}
        ]
        response = agent.invoke(prompt_messages)
        
        final_ans = response.final_answer
        if response.is_function_called and not final_ans:
            msg = response.responce
            rsp = response.function_call.dict()
            msg = HumanMessage(content=msg, name=name, additional_kwargs={'function_call': rsp})
        else:
            rsp = response.responce
            if final_ans:
                new_msg = f"FINAL ANSWER, {rsp}"
            else:
                new_msg = rsp
            
            msg, similarity = compare_and_update_message(new_msg, messages, name)
            if similarity > 50:
                print(f"Message similarity > 50% for {name}. Replacing old message.")
                update_agent_state(state, "genral_assistance_", msg)
                return Command(goto="supervisor",update={"name": name, "messages": [msg]})
                
        return Command(update={"name": name, "messages": [msg], "genral_assistance_": [msg]}, goto="supervisor")
    
    return genral_agent_node

def writer_agen(llm, system_message: str, name: str, examples="", tools=None, schema=None):
    agent = create_agent(llm, system_message=system_message, examples=examples, tools=tools, schema=schema)
    
    def writer_agent_node(state) -> Command[Literal["critic_"]]:
        writer_messages = state['writer']
        critic_messages = state['critic']
        writer_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["writer"])
        critic_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["critic"])

        prompt_messages = [
            *writer_messages,
            *critic_messages,
            {'role': 'user', 'content': f"user request : {state['user_request']} \n"},
        ]
        
        response = agent.invoke(prompt_messages)
        rsp = response.responce
        
        msg, word_distances = compare_and_update_message(rsp, writer_messages, name)
        if word_distances > 50:
            print(f"Message similarity > 50% for {name}. Replacing old message.")
            update_agent_state(state, "writer", msg)
            return Command(goto="critic_")
        
            
        return Command(update={"name": name, "writer": [msg]}, goto="critic_")
    
    return writer_agent_node

def critic_agen(llm, system_message: str, name: str, examples="", tools=None, schema=None):
    agent = create_agent(llm, system_message=system_message, examples=examples, tools=tools, schema=schema)

    def critic_agent_node(state) -> Command[Literal["supervisor", 'writer_', 'artist_']]:
        writer_messages = state['writer']
        critic_messages = state['critic']
        critic_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["critic"])
        writer_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["writer"])
        summary = state.get("summary", "")
        prompt_messages = [
            {'role': 'user', 'content': f"\nSummury up to this point for User Message : {summary}\n"},
            *critic_messages,
            *writer_messages,
            {'role': 'user', 'content': f"user request : {state['user_request']} \n"},
        ]
        
        response = agent.invoke(prompt_messages)
        final_ans = response.final_answer
        rsp = response.responce

        msg, word_distances = compare_and_update_message(rsp, critic_messages, name)

        if word_distances > 50:
            print(f"Message similarity > 50% for {name}. Replacing old message.")
            update_agent_state(state, "critic", msg)
            if final_ans:
                return Command(goto="artist_")
            return Command(goto="writer_")
            
        if final_ans:
            return Command(update={"name": name, "critic": [msg]}, goto="artist_")
        return Command(update={"name": name, "critic": [msg]}, goto="writer_")
    
    return critic_agent_node

def parse_prompts(state: State):
    prompts = state.stable_diffusion_prompts
    for prompt in prompts:
        print(f"Character/Scene Name: {prompt.character_or_scene_name}")
        print(f"Prompt: {prompt.prompt}")
        if prompt.negative_prompt:
            print(f"Negative Prompt: {prompt.negative_prompt}")
        print("---")
    return "Prompts Parsed and Printed"

def artist_agent(llm, system_message: str, name: str, examples="", tools=None, schema=None):
    agent = create_agent(llm, system_message=system_message, examples=examples, tools=tools, schema=schema)
    
    def artist_agent_node(state) -> Command[Literal["supervisor", "__end__"]]:
        writer_messages = state['writer']
        artist_messages = state['artist']
        #artist_messages = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["artist"])
        writer_messages_1 = get_highest_relevance_messages(state["user_request"], state, llm,agent_keys=["writer"])
        story = writer_messages[-1].content if writer_messages else ""
        prompt_messages = [
            {'role': 'user', 'content': f"Generate Stable Diffusion prompts for the following story:\n{story}"},
            *artist_messages,
            *writer_messages_1
        ]
        
        response = agent.invoke(prompt_messages)
        final_ans = response.final_answer
        rsp = response.responce
        msg = str(rsp)
        print(f"rsp : {type(rsp)}")
        msg = HumanMessage(content=msg, name=name)
        if final_ans:
            try:
                if not isinstance(rsp, list):
                    rsp = [rsp]
                return Command(
                    update={
                        "name": name,
                        "artist": [msg],
                        "stable_diffusion_prompts": rsp,
                        "messages": [writer_messages[-1]],
                    },
                    goto=END
                )
            except Exception as e:
                print(f"Error processing artist response: {e}")
                return Command(goto=END)
            
    return artist_agent_node

def compare_and_update_message(new_msg: str, history: List[BaseMessage], name: str) -> tuple[HumanMessage, float]:
    """Compare new message with last message in history and decide whether to replace"""
    if len(history) == 0:
        return HumanMessage(content=f"{name} : {new_msg}", name=name), 0
    
    
    last_msg = history[-1].content
    # Remove the agent name prefix if present
    last_msg_content = last_msg.replace(f"{name} : ", "")
    similarity = calculate_levenshtein_similarity(new_msg, last_msg_content)
    word_distances = {}
    #word_distances = word_by_word_levenshtein(new_msg, last_msg_content)
    # Return None to indicate we should keep the old message
    #print(f"Message similarity > 50% for {name}. Word-by-word distances: {word_distances}")
        
    return HumanMessage(content=f"{name} : {new_msg}", name=name), similarity

def update_agent_state(state: dict, agent_name: str, msg: HumanMessage) -> None:
    """
    Updates the state for a given agent by modifying the last message content.
    
    Args:
        state: The current state dictionary
        agent_name: Name of the agent (e.g., 'critic', 'writer', 'general_assistance')
        msg: New message to update with
    """
    agent_list = state.get(agent_name, [])
    
    # Ensure 'msg' is handled correctly
    if hasattr(msg, "content"):
        updated_agent_list = (
            agent_list[:-1] + [msg]
            if agent_list else [msg]
        )
    else:
        raise ValueError("The provided 'msg' object does not have a 'content' attribute.")

    
    state[agent_name] = updated_agent_list
    print(f"updated_agent_list :{agent_name} {state[agent_name]}")


def graph_init():

    exmples = """
    ```json
    {
    "responce":"",
    "final_answer": false,
    "is_function_called": false,
        "function_call": {"name": "add", "arguments": {"a": 2, "b": 5}}
    }

    ```
    Here Some are Invalid example :
    ```json
    {
  "responce": "The result of adding 2 and 3 is 5",
  "final_answer": true,
  "is_function_called": true,
    "function_call": {
    "name": "add",
    "arguments": {
      "a": 2,
      "b": 3
    }
    }
    }
    ```
    """


    dec_assistant = AgentConfig(name="genral_assistance",description="Answer any question about any topic.It is an personal assistant.")
    dec_writer = AgentConfig(name="writer_",description="writer a story based on user request.")
    dec_critic = AgentConfig(name="critic_",description="Give Suggestion to improve story written by writer.")
    dec_artist = AgentConfig(name="artist_", description="Generate Stable Diffusion prompts for story characters and scenes")
    
    assistant_agent = create_genral_agent(llm,system_message = genral_assistance,name="genral_assistance",examples=exmples,tools = Tool_list,schema=Responce)
    writer_agent = writer_agen(llm,system_message = writer_prompt,name="writer_",examples="",tools =[],schema=artist)
    critic_agent = critic_agen(llm,system_message = critic_prompt,name="critic_",examples="",tools =[],schema=artist)
    artist_agent_ = artist_agent(llm, system_message=artist_prompt, name="artist_", examples="", tools=[], schema=artist_responce)


    supervisor_node = supervisor(model_large,agents=[dec_assistant,dec_writer],tools = ["call_tool","summarize"])
    summarize_conversation_node = summarize_conversation(model_large)

    

    workflow = StateGraph(State)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("genral_assistance", assistant_agent)
    workflow.add_node("call_tool", tool_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("delete", delete_messages)
    workflow.add_node("writer_", writer_agent)
    workflow.add_node("critic_", critic_agent)
    workflow.add_node("artist_", artist_agent_)
    workflow.add_edge(START, "supervisor")

    return workflow
