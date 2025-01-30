from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, RemoveMessage,AIMessage,HumanMessage,BaseMessage,FunctionMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from typing import List,Annotated,Sequence,Dict,Any,Optional, Literal
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langchain_core.utils.function_calling import convert_to_openai_function
from termcolor import colored
from langchain_ollama.llms import OllamaLLM
from graph_state import State,AgentConfig,get_agent_name,get_agent_details
from agent_tool import Tool_list,Tools_Executor
from prompt import supervisor_prompt
import uuid

def create_agent(llm,system_message: str,examples="",tools=None, schema=None):
    """
    Create an agent
    """

    parser = PydanticOutputParser(pydantic_object=schema)
    format_instructions = parser.get_format_instructions()
    functions = [convert_to_openai_function(t) for t in tools] if tools else []
    prompt = ChatPromptTemplate.from_messages([
        ("system","If you finished the user request then please respond with FINAL ANSWER or set final_answer = True. \n{system_message}.\n Give Responce in JSON with this structure everthing follow the given structure and do give anything other than json.There should be no output before json and after json. \n {format_instructions}.\nThe Tool names are : {tool_name}\n.This are the Tool Details \n {functions}.\n Here is some previous Output : {examples}.\n"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    def function_parser(responce):
        content = responce.content
        return content

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(examples=examples) 
    if tools:
        prompt = prompt.partial(tool_name=", ".join([tool.name for tool in tools]))
    else:
        prompt = prompt.partial(tool_name="")
    prompt = prompt.partial(functions = functions)
    
    if isinstance(llm,GoogleGenerativeAI) or isinstance(llm,OllamaLLM):

        prompt = prompt.partial(format_instructions = format_instructions)
        agent = prompt | llm|parser

    else:

        prompt = prompt.partial(format_instructions = format_instructions)
        agent = prompt|llm|function_parser|parser 
    return agent
    


def tool_node(state) -> Command[Literal["supervisor"]]:
    messages = state['messages']
    print(colored(f"tool_node \n ---------\n {messages}\n--------\n", "green"))
    last_msg = messages[-1]
    tool_inputs = last_msg.additional_kwargs["function_call"]["arguments"]
        
    tool_name = last_msg.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool = tool_name,
        tool_input = tool_inputs
    )
    responce = Tools_Executor.invoke(action)

    function_message = FunctionMessage(role="function",content = f"Tool Output for the following input {tool_inputs} tool {tool_name} = {str(responce)}",name = action.tool)

    return Command(update = {"messages":[function_message]},goto = "supervisor")

def router(state):
    messages = state["messages"]
    print("router eorror")
    last_messages = messages[-1]
    if "function_call" in last_messages.additional_kwargs:
        return "call_tool"

    if "FINAL ANSWER" in last_messages.content:
        return "end"
    
    if len(messages) > 3:
        return "summarize"
    return "continue"


def summarize_conversation(llm):
    
    def summarize_conversation(state: State)->Command[Literal["supervisor"]]:
    # First, we summarize the conversation
        summary = state.get("summary", "")
        if  summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"
        
        messages = state["messages"]
        rsp = llm.invoke([SystemMessage(content=summary_message), *messages])
        if isinstance(rsp,AIMessage):
            response = rsp.content
        if len(messages) > 10:
            delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
        else:
            return Command(goto = "supervisor")
        return Command(update={"summary": rsp, "messages": delete_messages},goto = "supervisor")
    
    return summarize_conversation

def delete_messages(state)->Command[Literal["__end__"]]:
    messages = state["messages"]
    return Command(update={"messages": [RemoveMessage(id=m.id) for m in messages[:]]},goto = END)

def supervisor(llm,agents:List[AgentConfig],tools:List[str])->str:

    is_gemini = False
    members = get_agent_name(agents)
    members_details = get_agent_details(agents)
    if isinstance(llm,GoogleGenerativeAI) or isinstance(llm,OllamaLLM):
        is_gemini = True

    options = ["FINISH"] + members
    final = members + tools
    system_prompt = supervisor_prompt.format(members=members,details_worker=members_details)
    class Router(BaseModel):
        """Worker to route to next and describe the task to the next worker. If no workers needed, route to FINISH."""

        next: Literal[*options]
        task:str = Field(default="describe task to next worker")

    def supervisor_node(state:State) -> Command[Literal[*final, "__end__","delete"]]:
        """An LLM-based router."""
        
        messages = state["messages"]
        last_messages = messages[-1]
        if "function_call" in last_messages.additional_kwargs:
            return Command(goto="call_tool")
        
        if len(messages) > 10:
            return Command(goto="summarize")

        parser = PydanticOutputParser(pydantic_object=Router)
        format_instructions = parser.get_format_instructions()
        summary = state.get("summary", "")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": format_instructions},
            {"role": "user", "content": f"summary of conversation up to this point: {summary}\n\n"},
        ] + state["messages"]
        
        print(colored(f"supervisor_node \n ---------\n {messages}\n--------\n", "yellow"))

        response = llm.invoke(messages)
        if not is_gemini:
            response = response.content
        response =  parser.parse(response)
        goto = response.next
        print(f"goto is {goto}")
        if goto == "FINISH":
            goto = END
        if goto in members:
            if goto != "genral_assistance":
                msg = HumanMessage(content=f"supervisor : {response.task}",name = "supervisor",id = str(uuid.uuid4()))
                new_go = goto[0:-1]
                return Command(update={new_go: [msg],"user_request":response.task},goto=goto)
            if goto == "genral_assistance":
                msg = HumanMessage(content=f"supervisor : {response.task}",name = "supervisor",id = str(uuid.uuid4()))
                return Command(update={"genral_assistance_": [msg],"user_request":response.task},goto=goto)
        return Command(goto=goto)

    return supervisor_node
