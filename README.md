# CineBot: Movie RAG Agent

**Live Demo:** [cinebot-rag-agent.streamlit.app](https://cinebot-rag-agent-mnamyragmr8qch7sfuu8tl.streamlit.app/)

A conversational AI agent that answers questions by dynamically routing them to the right knowledge source — a vector database, Wikipedia, or its own LLM reasoning — using LangGraph, AstraDB, and Groq.

---

## The Problem

Standard LLM chatbots have two failure modes:

1. **They hallucinate** when asked about specific, grounded knowledge they weren't trained on.
2. **They over-retrieve** — hitting a vector database for every question, even simple greetings or general knowledge that doesn't need it.

CineBot solves both by adding an intelligent routing layer before any retrieval happens.

---

## How It Works

Every user message goes through a three-stage pipeline:

```
User Question
      │
      ▼
┌─────────────┐
│   Router    │  ← LLM classifies the question type
└──────┬──────┘
       │
  ┌────┴────────────────┐
  │                     │                     │
  ▼                     ▼                     ▼
vectorstore          wiki_search         direct_answer
(AstraDB)           (Wikipedia)          (LLM only)
  │                     │
  └──────────┬──────────┘
             ▼
        ┌─────────┐
        │Generate │  ← LLM synthesizes context into a response
        └─────────┘
```

### Stage 1: Routing

A Groq LLM (`llama-3.1-8b-instant`) reads the question and outputs a structured JSON decision:

- `vectorstore` — domain-specific questions about the indexed content
- `wiki_search` — general knowledge questions (people, events, facts)
- `direct_answer` — greetings, small talk, meta questions

The router is constrained via `with_structured_output` so it **only** emits the routing decision — it never answers the question itself.

### Stage 2: Retrieval

**Vector Store path:** Queries DataStax AstraDB using semantic search. Documents are chunked and embedded with HuggingFace's `all-MiniLM-L6-v2` model, stored as dense vectors. The retriever finds the most semantically similar chunks to the question.

**Wikipedia path:** Calls the Wikipedia API directly for open-world factual queries.

**Direct answer path:** Skips retrieval entirely — the LLM answers from its own knowledge.

### Stage 3: Generation

Retrieved context (from either source) is injected into a prompt and passed to the LLM, which synthesizes a concise, grounded response. This keeps answers factual and traceable to source documents.

---

## Architecture

```
cinebot-rag-agent/
├── app.py          # Streamlit UI — streams LangGraph node execution live
├── graph.py        # LangGraph StateGraph — defines nodes, edges, routing logic
├── config.py       # Environment setup — reads from .env or Streamlit secrets
├── rag.py          # AstraDB vector store — embeddings + retriever
└── tools.py        # Wikipedia + Arxiv API wrappers
```

**State schema** (flows through every node):
```python
{
  "question":   str,   # original user input
  "documents":  list,  # retrieved chunks or wiki results
  "generation": str    # final LLM response
}
```

**Tech stack:**
| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM + routing | Groq (`llama-3.1-8b-instant`) |
| Vector database | DataStax AstraDB (Data API) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Web search | Wikipedia API |
| UI | Streamlit |

---

## Local Setup

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/your-username/cinebot-rag-agent
cd cinebot-rag-agent
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
```
ASTRA_DB_APPLICATION_TOKEN="AstraCS:..."
ASTRA_DB_ID="your-database-uuid"
ASTRA_DB_API_ENDPOINT="https://<db-id>-<region>.apps.astra.datastax.com"
GROQ_API_KEY="your-groq-key"
```

Run the UI:
```bash
streamlit run app.py
```

Or via CLI:
```bash
python main.py "What is prompt engineering?"
python main.py "Who directed Inception?"
python main.py "hi"
```

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo, set `app.py` as entrypoint
3. Under **Advanced Settings → Secrets**, add:

```toml
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:..."
ASTRA_DB_ID = "your-database-uuid"
ASTRA_DB_API_ENDPOINT = "https://<db-id>-<region>.apps.astra.datastax.com"
GROQ_API_KEY = "your-groq-key"
```

4. Deploy. The app reads secrets from `st.secrets` on Streamlit Cloud and from `.env` locally — no code changes needed between environments.

---

## Why LangGraph Over a Simple Chain?

A standard LangChain chain executes steps linearly — every question hits every step. LangGraph models the workflow as a **directed graph**, which means:

- Routing is a first-class decision, not an afterthought
- Each node is isolated and testable independently
- New retrieval sources (Arxiv, a SQL database, an API) can be added as nodes without touching existing logic
- The graph state is explicit and inspectable at every step

The Streamlit UI surfaces this — you can watch which node each question flows through in real time.
