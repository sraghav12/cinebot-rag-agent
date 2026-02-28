# AstraDB and LangGraph Agent

This is a complete Python port of the `3_Langraph_With_Astradb.ipynb` notebook into a fully modular project structure. This enables easier source control, reuse, and deployment of your agent application.

## Prerequisites
- **Python 3.10+**
- (Optional but recommended) Conda or `venv` virtual environment

## Setup

1. **Clone or navigate** to the repository:
   ```bash
   cd langgraph-astra-agent
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
