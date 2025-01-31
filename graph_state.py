from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, RemoveMessage,AIMessage,HumanMessage,BaseMessage,FunctionMessage
from typing import List,Annotated,Sequence,Dict,Any,Optional, Literal
import operator
from pydantic import BaseModel, Field



class AgentConfig(BaseModel):
    name: str
    description: str

class Responce(BaseModel):
    responce : str = Field(...,description= "If function is called then it shoud empty otherwise it contain responce regarding user request")
    final_answer : bool = Field(...,description="If user request is met or for asking the question to user then it should be True")
    is_function_called: bool = Field(..., description="Indicates whether the function was called (True or False)")
    function_call: 'FunctionName' = Field(..., description="Details of the function that was called")

    class FunctionName(BaseModel):
        name: str = Field(..., description="The name of the function that was called")
        arguments: Dict[str, Any] = Field(..., description="Dictionary of parameter names and their values")

class PromptSchema(BaseModel):
    character_or_scene_name: str = Field(..., title="Character/Scene Name", description="The name of the character or scene.")
    prompt: str = Field(..., title="Prompt", description="A detailed prompt including style, lighting, camera angle, etc.")
    negative_prompt: Optional[str] = Field(None, title="Negative Prompt", description="What to avoid in the generation.")
    search_query: Optional[str] = Field(None, title="Search Query", description="The search query for the image.")

class artist(BaseModel):
    responce : str = Field(...,description= "If function is called then it shoud empty otherwise it contain responce regarding user request")
    final_answer : bool = Field(...,description="If user request is met or for asking the question to user then it should be True")

class artist_responce(BaseModel):
    responce: List[PromptSchema] = Field(...,description="List of prompts for characters and scenes")
    final_answer : bool = Field(...,description="If user request is met or for asking the question to user then it should be True")

class State(MessagesState):
    summary: str
    name: str
    context_score: Dict[str, float] = {}  # Store context scores for each agent
    user_request: str = ""  # Store the current user request
    writer: Annotated[Sequence[BaseMessage], operator.add]
    artist: Annotated[Sequence[BaseMessage], operator.add]
    critic: Annotated[Sequence[BaseMessage], operator.add]
    genral_assistance_: Annotated[Sequence[BaseMessage], operator.add]
    image_urls : List[str]
    stable_diffusion_prompts: List[PromptSchema] = []

def get_agent_name(agent:List[AgentConfig]) -> List[str]:
    name = [names.name for names in agent]
    return name

def get_agent_details(agent:List[AgentConfig]) -> List[str]:
    description = [details.description for details in agent]
    return description
