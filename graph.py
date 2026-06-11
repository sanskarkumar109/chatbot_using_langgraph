from typing import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages

from langchain_core.messages import AnyMessage

from langgraph.graph import (
    StateGraph,
    START
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq

from tools import TOOLS

from dotenv import load_dotenv

load_dotenv()


# ------------------
# LLM
# ------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

llm_with_tools = llm.bind_tools(
    TOOLS
)


# ------------------
# State
# ------------------

class AgentState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]


# ------------------
# Chatbot Node
# ------------------

def chatbot(state: AgentState):

    return {
        "messages": [
            llm_with_tools.invoke(
                state["messages"]
            )
        ]
    }


# ------------------
# Build Graph
# ------------------

builder = StateGraph(
    AgentState
)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    ToolNode(TOOLS)
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge(
    "tools",
    "chatbot"
)

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)