from typing import Literal, List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

from config import get_llm
from rag import get_retriever
from tools import get_tools

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# Global resources initialization for the graph nodes
llm = None
structured_llm_router = None
question_router = None
retriever = None
wiki = None

def init_graph_resources():
    global llm, structured_llm_router, question_router, retriever, wiki
    
    llm = get_llm()
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    question_router = route_prompt | structured_llm_router
    retriever = get_retriever()
    
    tools = get_tools()
    wiki = tools['wiki']

def retrieve(state):
    """Retrieve documents"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """wiki search based on the re-phrased question."""
    print("---wikipedia---")
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = Document(page_content=docs)
    return {"documents": wiki_results, "question": question}

def route_question(state):
    """Route question to wiki search or RAG."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def get_app():
    """Builds and compiles the LangGraph StateGraph."""
    init_graph_resources()
    
    workflow = StateGraph(GraphState)
    
    # Define nodes
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    
    # Define edges
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    
    return workflow.compile()
