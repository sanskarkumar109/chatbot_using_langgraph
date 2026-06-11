import os

from dotenv import load_dotenv

load_dotenv()

# Wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Arxiv
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

# Tavily
from langchain_community.tools.tavily_search import TavilySearchResults

# DuckDuckGo
from langchain_community.tools import DuckDuckGoSearchRun

# Custom Tool
from langchain.tools import tool

from rag import load_retriever


# ------------------
# Wikipedia
# ------------------

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000
    )
)


# ------------------
# Arxiv
# ------------------

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000
    )
)


# ------------------
# Tavily
# ------------------

tavily_tool = TavilySearchResults(
    max_results=5
)


# ------------------
# DuckDuckGo
# ------------------

ddg_tool = DuckDuckGoSearchRun()


# ------------------
# PDF Tool
# ------------------

@tool
def pdf_search(query: str):

    """
    Search uploaded PDFs.
    """

    try:

        retriever = load_retriever()

        docs = retriever.invoke(query)

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    except Exception:

        return "No PDF indexed yet."


TOOLS = [
    wiki_tool,
    arxiv_tool,
    tavily_tool,
    ddg_tool,
    pdf_search
]