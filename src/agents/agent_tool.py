from langchain_core.tools import tool
from src.search_utils.search import perform_search
from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation
import asyncio
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def genrate_image(prompt: str) -> str:
    """Genrates an image.From The Prompt

    Args:
        prompt: prompt string
    """
    return "Genrated Image"

@tool
def search_images(query: str) -> list[str]:
    """Searches for images on the web using DuckDuckGo.

    Args:
        query: The search query.
    """
    return asyncio.run(perform_search(query=query, search_engine="duckduckgo"))


Tool_list = [add,multiply]
Tools_Executor = ToolExecutor(Tool_list)
