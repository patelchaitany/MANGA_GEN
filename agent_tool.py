from langchain_core.tools import tool

from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation

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
Tool_list = [add,multiply,genrate_image]
Tools_Executor = ToolExecutor(Tool_list)

