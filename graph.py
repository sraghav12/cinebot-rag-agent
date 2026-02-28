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
    datasource: Literal["vectorstore", "wiki_search", "direct_answer"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia, a vectorstore, or provide a direct_answer for general chat or greetings.",
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
    
    system = """You are a routing assistant. Your ONLY job is to route the user's question to either 'vectorstore', 'wiki_search', or 'direct_answer'.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
If the question is about agents, prompt engineering, or adversarial attacks, route to 'vectorstore'.
If the question is a standard greeting, small talk, or a general question that does not require searching external sources, route to 'direct_answer'.
Otherwise, route to 'wiki_search'.
You must ONLY output the tool call JSON. Do NOT answer the user's question. Do NOT provide any conversational text."""
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
    """Route question to wiki search, RAG, or direct answer."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO DIRECT ANSWER---")
        return "direct_answer"

def generate(state):
    """Generate answer."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    if isinstance(documents, list):
        context = "\n\n".join(getattr(doc, 'page_content', str(doc)) for doc in documents)
    else:
        context = getattr(documents, 'page_content', str(documents))
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
        ("human", "{question}")
    ])
    
    rag_chain = prompt | llm
    response = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": response.content}

def direct_answer(state):
    """Answer directly without context."""
    print("---DIRECT ANSWER---")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly AI assistant. Answer the user's question directly and conversationally without needing external context."),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    
    return {"generation": response.content}

def get_app():
    """Builds and compiles the LangGraph StateGraph."""
    init_graph_resources()
    
    workflow = StateGraph(GraphState)
    
    # Define nodes
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)
    
    # Define edges
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
            "direct_answer": "direct_answer"
        },
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("wiki_search", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()
