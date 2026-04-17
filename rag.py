import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore

# Config module for db/api access
from config import setup_environment, _get_secret

def get_vector_store():
    """
    Returns the configured Astra DB Vector Store using the Data API.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    token = _get_secret("ASTRA_DB_APPLICATION_TOKEN") or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    db_id = _get_secret("ASTRA_DB_ID") or os.getenv("ASTRA_DB_ID")

    # Derive the API endpoint from the database ID stored in the error URL pattern
    # Format: https://{db_id}-{region}.apps.astra.datastax.com
    api_endpoint = _get_secret("ASTRA_DB_API_ENDPOINT") or os.getenv("ASTRA_DB_API_ENDPOINT")
    if not api_endpoint:
        raise ValueError(
            "ASTRA_DB_API_ENDPOINT is missing. "
            "Add it to your secrets: https://<db-id>-<region>.apps.astra.datastax.com"
        )

    astra_vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="qa_mini_demo",
        api_endpoint=api_endpoint,
        token=token,
    )
    return astra_vector_store

def build_index():
    """
    Fetches reference URLs, splits the text, and
    indexes them into the Astra DB Vector Store.
    """
    setup_environment()

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

    print(f"Inserted {len(doc_splits)} documents into AstraDB.")
    return astra_vector_store

def get_retriever():
    """Returns the vector store as a Retriever."""
    return get_vector_store().as_retriever()

if __name__ == "__main__":
    # If run directly as a script, populate the index
    build_index()
