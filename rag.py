from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper

# Config module for db/api access
from config import setup_environment

def get_vector_store():
    """
    Returns the configured Astra DB Vector Store
    using HuggingFaceEmbeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )
    return astra_vector_store

def build_index():
    """
    Fetches reference URLs, splits the text, and
    indexes them into the Astra DB Vector Store.
    """
    setup_environment() # Ensure DB token is set

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    astra_vector_store = get_vector_store()
    astra_vector_store.add_documents(doc_splits)
    
    print(f"Inserted {len(doc_splits)} headlines into AstraDB.")
    return VectorStoreIndexWrapper(vectorstore=astra_vector_store)

def get_retriever():
    """Returns the vector store as a Retriever."""
    return get_vector_store().as_retriever()

if __name__ == "__main__":
    # If run directly as a script, populate the index
    build_index()
