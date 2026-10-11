# TechSupportAI 🛠️

**A multi-product technical support assistant powered by
Retrieval-Augmented Generation (RAG).**

TechSupportAI helps users troubleshoot hardware and software
issues by retrieving relevant information from a local knowledge
base and using a Gemini language model to generate helpful
responses.

## Features

- **Hybrid retrieval:** Combines dense vector retrieval with BM25
keyword search.
- **Parent-child retrieval:** Supports retrieving relevant
content while preserving broader document context.
- **Cross-encoder reranking:** Reorders retrieved results to
improve relevance.
- **RAG-based generation:** Uses retrieved knowledge as context
for Gemini responses.
- **FastAPI backend:** Provides an API for submitting technical
questions.
- **Streamlit interface:** Offers a conversational
troubleshooting interface.
- **Evaluation tools:** Includes retrieval, out-of-scope, and RAG
evaluation scripts.
- **LangSmith integration:** Supports tracing and debugging LLM
workflows.

## Tech Stack

- **Language:** Python
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **LLM:** Google Gemini through LangChain
- **Retrieval:** ChromaDB, BM25
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Reranking:** Cross-encoder
- **Monitoring:** LangSmith
- **Evaluation:** RAGAS and custom evaluation scripts

## Project Structure

```text
TECHSUPPORTAI/
├── backend/
│   └── app/
│       ├── api/
│       ├── core/
│       ├── evaluation/
│       ├── generation/
│       ├── ingestion/
│       ├── models/
│       └── retrieval/
├── data/
│   └── knowledge_base/
├── evaluation/
│   ├── datasets/
│   └── results/
├── frontend/
│   └── app.py
├── scripts/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TECHSUPPORTAI
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local environment file:

```bash
cp .env.example .env
```

Add your own Google AI API key and LangSmith API key to `.env`.

Never commit `.env` or publish real API keys.

### 5. Start the FastAPI backend

From the project root, run:

```bash
uvicorn backend.app.main:app --reload
```

The backend runs at:

`http://127.0.0.1:8000`

API documentation is available at:

`http://127.0.0.1:8000/docs`

Health check:

`http://127.0.0.1:8000/health`

### 6. Start the Streamlit frontend

Open a second terminal, activate the virtual environment, and
navigate to the project root.

Run:

```bash
streamlit run frontend/app.py
```

The interface will open in your browser.

## Knowledge Base

The current sample knowledge base includes troubleshooting
documents for:

- Laptops
- Routers
- Software

You can extend the knowledge base with additional relevant
technical documentation.

## Evaluation

The repository includes evaluation datasets and scripts for
retrieval quality, out-of-scope questions, and RAG evaluation.

Evaluation scores should be measured and reported from
reproducible runs; no performance claims are made here.

## Current Limitations

- The assistant's knowledge is limited by the documents available
in its knowledge base.
- Answer quality depends on retrieval relevance and the generated
response.
- API credentials and external model availability are required
for Gemini generation.
- The project is a learning and portfolio project, not a
production-ready support system.

## Future Improvements

- Expand the technical knowledge base.
- Improve retrieval evaluation and failure analysis.
- Add stronger groundedness and citation checks.
- Improve error handling and configuration of the frontend API
URL.
- Add automated tests and deployment support.

## Security

Keep API keys in your local `.env` file. The repository provides
`.env.example` with placeholder values for configuration.

## License

A license has not yet been selected. Add a license before
explicitly permitting reuse of this project.
