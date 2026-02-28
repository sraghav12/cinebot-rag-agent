import os
from dotenv import load_dotenv
import cassio

def setup_environment():
    """
    Load environment variables from .env and initialize connections
    like AstraDB and LangSmith.
    """
    load_dotenv()

    # Astra DB setup
    astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_id = os.getenv("ASTRA_DB_ID")
    if not astra_db_token or not astra_db_id:
        raise ValueError("AstraDB tokens are missing in the environment.")
    
    cassio.init(token=astra_db_token, database_id=astra_db_id)

    # Groq API check
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing in the environment.")

    return groq_api_key

def get_llm():
    from langchain_groq import ChatGroq
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
