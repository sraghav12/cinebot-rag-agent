# 🍿 CineBot: Movie RAG Agent

This is a complete Python port of the LangGraph + AstraDB concepts into a fully modular project structure. This agent securely fetches external documents while handling small talk and greeting interactions. It serves as a tailored movie-expert RAG agent pointing to complex movie data records.

## Prerequisites
- **Python 3.10+**
- (Optional but recommended) Conda or `venv` virtual environment

## Setup

1. **Clone or navigate** to the repository:
   ```bash
   cd cinebot-rag-agent
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Environment Variables**:
   Copy the example environment file and fill in your keys.
   ```bash
   cp .env.example .env
   ```
   Add your `ASTRA_DB_APPLICATION_TOKEN`, `ASTRA_DB_ID`, and `GROQ_API_KEY`.

4. **Initialize Vectorstore (Optional depending on how you split data)**
   In this codebase, documents are indexed dynamically in `rag.py` upon initial load.

## Usage

### Using the Interactive UI
Run the Streamlit application for a visual, conversational chat interface:
```bash
streamlit run app.py
```

### Using the CLI
You can use the `main.py` entrypoint to talk to your agent. Depending on what you ask, it will route either to a RAG query (on the blog posts) or Wikipedia.

```bash
python main.py "What is an agent in LangChain?"
```

```bash
python main.py "Who is Shahrukh Khan?"
```

## How It Works (Technical Architecture)

This project leverages **LangGraph** to create a deterministic, state-driven workflow for Retrieval-Augmented Generation (RAG). Instead of a standard linear chain, the agent uses a **Directed Acyclic Graph (DAG)** to route questions dynamically.

Here is the breakdown of the backend flow:
1. **State Management (`graph.py`)**: The agent's memory and current process are managed via a `TypedDict` state, containing the `question`, retrieved `documents`, and final LLM `generation`.
2. **Intelligent Routing**: 
   - A `ChatGroq` LLM (using the robust **Llama-3.3-70b-versatile** model) acts as the decision engine.
   - It analyzes the user's prompt and strictly outputs a JSON decision to route the query to one of three nodes:
     - `vectorstore`: For domain-specific questions mapped to the database.
     - `wiki_search`: For general knowledge questions (using `WikipediaAPIWrapper`).
     - `direct_answer`: For greetings and conversational small talk.
3. **Retrieval (`rag.py` & `tools.py`)**: 
   - When routed to the vector store, the agent queries **DataStax AstraDB** (Cassandra), populated with chunks of text vectorized using HuggingFace's `all-MiniLM-L6-v2` embeddings.
4. **Generation Node**: 
   - Regardless of the retrieval source (Wiki or AstraDB), the gathered documents are passed to the `generate` node. The LLM synthesizes this raw context into a natural, conversational response using prompt engineering.
5. **Interactive UI (`app.py`)**: 
   - A Streamlit frontend displays the routing process in real-time, pulling the final `generation` state and presenting it interactively to the user.

## Deployment (Streamlit Community Cloud)

You can easily host this interactive UI for free using **Streamlit Community Cloud**:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app** and select your `cinebot-rag-agent` repository.
3. Set the Main file path to `app.py`.
4. Click on **Advanced Settings** before deploying.
5. In the **Secrets** text box, paste the contents of your `.env` file just like this:
   ```toml
   ASTRA_DB_APPLICATION_TOKEN="your_token_here"
   ASTRA_DB_ID="your_db_id"
   GROQ_API_KEY="your_groq_key"
   ```
6. Click **Save** and then **Deploy**!

Your movie agent will be live on a public URL to share with others!

## Acknowledgements

It was incredibly fun to work on this project! Building a conversational agent using LangGraph, structured prompt routing, and vector retrieval with AstraDB was a fantastic experience in AI engineering.
