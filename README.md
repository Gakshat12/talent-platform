🎯 Talent Intelligence Platform

An AI-powered Talent Intelligence Platform that helps recruiters find, rank, and understand the best candidates for any Job Description.

The platform accepts a raw Job Description, converts it into structured hiring intent, retrieves relevant candidates using hybrid search, reranks them with a cross-encoder, applies deterministic business scoring, and produces recruiter-friendly explanations.

🏗️ Architecture

                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI API     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LangGraph        │
                    │    Orchestrator      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        JD Parsing       Query Expansion    Retrieval
              │                │                │
              │                │        ┌───────┴───────┐
              │                │        ▼               ▼
              │                │      BM25            FAISS
              │                │        │               │
              │                │        └───────┬───────┘
              │                │                ▼
              │                │       Metadata Filtering
              │                │                ▼
              │                │               RRF
              │                │                ▼
              │                │        CrossEncoder
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                     Business Ranker
                               │
                               ▼
                       Top 100 Candidates
                               │
                               ▼
                    Explainability Layer
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        Deterministic Explanation      LLM Top-5 Enhancement
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                        Recruiter Results

🛠️ Tech Stack
Backend
Python 3.12
FastAPI
LangGraph
Pydantic v2
Loguru
Retrieval
rank_bm25
FAISS
SentenceTransformers
all-MiniLM-L6-v2
Reciprocal Rank Fusion
Reranking
CrossEncoder
cross-encoder/ms-marco-MiniLM-L-6-v2
LLM

The application uses an OpenAI-compatible LLM client and can be configured through environment variables.

Current development setup uses the Gemini OpenAI-compatible endpoint.

Frontend
Streamlit
Pandas
Deployment
Docker
Docker Compose

⚙️ Setup
1. Clone the repository
git clone <your-repository-url>
cd talent-platform
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
🔐 Environment Variables

Create a .env file from the example:

Copy-Item .env.example .env

Configure your LLM provider:

OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

LLM_MODEL=your_available_model
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.0

Retrieval configuration:

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

BM25_TOP_K=500
DENSE_TOP_K=500
RRF_TOP_K=2000
FINAL_TOP_K=100

Scoring configuration:

WEIGHT_EVIDENCE_ALIGNMENT=0.30
WEIGHT_EXPERIENCE_FIT=0.25
WEIGHT_CREDIBILITY=0.20
WEIGHT_HIREABILITY=0.15

Paths:

FAISS_INDEX_PATH=indexes/candidates.index
BM25_INDEX_PATH=indexes/candidates.bm25.json
CANDIDATES_DATA_PATH=data/candidates.jsonl

📊 Candidate Data

Candidates are stored as JSONL:

data/candidates.jsonl

Each line represents one candidate profile.

The project has been tested with a dataset of approximately:

100,000 candidates
🧱 Build Retrieval Indexes

Run:

python -m scripts.build_index

This creates or validates:

indexes/candidates.index
indexes/candidates.bm25.json
indexes/candidates.metadata.json

The script reuses an existing compatible FAISS index instead of regenerating all embeddings.

▶️ Run the Backend

Start FastAPI:

uvicorn app.api.main:app

API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
🖥️ Run Streamlit

In another terminal:

streamlit run frontend/streamlit_app.py

Open:

http://localhost:8501

When running locally, the frontend defaults to:

http://localhost:8000

When running inside Docker, API_BASE_URL is set automatically to:

http://api:8000

🔌 API
POST /api/v1/rank

Rank candidates against a Job Description.

Request
{
  "raw_text": "We are looking for a Machine Learning Engineer with 3+ years of experience in Python, NLP, PyTorch and production ML deployment."
}
Response

The response contains:

Parsed JD information
Ranked candidates
Deterministic score breakdown
Matched skills
Missing skills
Recruiter explanations
Candidate ranking metadata
🐳 Docker

Build:

docker compose build

Run:

docker compose up

Services:

FastAPI:
http://localhost:8000

Streamlit:
http://localhost:8501

Stop:

docker compose down

The Docker setup mounts:

./data    → /app/data
./indexes → /app/indexes

so candidate data and persisted retrieval indexes remain available outside the container.

🧪 Testing

The project includes automated tests covering the main pipeline components.

Run:

pytest -q

Current validation:

34 passed
🔍 Retrieval Pipeline

Candidate retrieval happens in five stages.

Stage 1 — BM25

The system runs BM25 over the expanded Job Description queries and keeps the top sparse matches.

Stage 2 — FAISS

A semantic query embedding is generated using:

all-MiniLM-L6-v2

and searched against the persisted FAISS IndexFlatIP.

Stage 3 — Filtering

Retrieved candidates are combined and filtered using structured JD metadata such as experience and applicable constraints.

Stage 4 — RRF

Sparse and dense rankings are fused using Reciprocal Rank Fusion.

Stage 5 — CrossEncoder

The resulting candidate pool is reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The final top 100 candidates are passed to the business scoring layer.

📈 Business Scoring

The business ranking layer uses deterministic scoring based on structured candidate evidence and the parsed Job Description.

The core formula is:

final_score =
    evidence_alignment × 0.30
  + experience_fit × 0.25
  + credibility × 0.20
  + hireability × 0.15
  + 0.10
  - penalties

The final result is scaled to:

0–100

LLM output does not determine the candidate's core ranking score.

💡 Explainability Strategy

The explanation system is intentionally resilient.

Top 100 ranked candidates
        ↓
Deterministic explanation for all 100
        ↓
LLM enhancement for top 5
        ↓
LLM succeeds → enhanced explanation
LLM fails    → deterministic explanation retained

This means temporary provider failures or quota limits do not prevent candidate ranking.

⚡ Performance Design

The platform avoids rebuilding expensive retrieval resources for every request.

Persisted resources:

FAISS index
BM25 corpus

Shared in-memory resources:

Embedding model
CrossEncoder
Retrieval objects

This allows subsequent requests in the same process to reuse the retrieval infrastructure.

🛡️ Reliability

The system is designed to degrade gracefully.

Examples:

Invalid candidate records are skipped.
Missing candidate data returns an empty candidate pool.
Missing retrieval indexes can trigger controlled rebuilding.
LLM parsing failures are handled through retries/fallbacks.
LLM explanation failures fall back to deterministic explanations.
API input validation prevents empty/invalid JDs from reaching the pipeline.
📌 Example JD
We are looking for a Machine Learning Engineer with 3+ years of professional experience in Python, machine learning, NLP, PyTorch, and deploying production ML systems. The candidate should have experience building and maintaining ML pipelines, developing NLP or LLM-based applications, and exposing models through backend APIs. Experience with Docker, cloud platforms, vector databases, and model deployment is preferred.

👤 Author
Akshat Gupta