import os
from dotenv import load_dotenv

def _get_secret(key):
    """Get a secret from st.secrets (Streamlit Cloud) or os.environ (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

def setup_environment():
    """
    Load environment variables from .env and propagate to os.environ
    so downstream libraries can find them.
    """
    load_dotenv()

    astra_db_token = _get_secret("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_id = _get_secret("ASTRA_DB_ID")
    if not astra_db_token or not astra_db_id:
        raise ValueError("AstraDB tokens are missing. Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_ID in Streamlit secrets or .env file.")

    groq_api_key = _get_secret("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Set it in Streamlit secrets or .env file.")

    # Propagate to os.environ so downstream libraries (LangChain, etc.) can find them
    os.environ["ASTRA_DB_APPLICATION_TOKEN"] = astra_db_token
    os.environ["ASTRA_DB_ID"] = astra_db_id
    os.environ["GROQ_API_KEY"] = groq_api_key

    return groq_api_key

def get_llm():
    from langchain_groq import ChatGroq
    groq_api_key = _get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    # Using 8b model for ultra-fast low-latency responses
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
