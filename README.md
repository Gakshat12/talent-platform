# 🎯 Talent Intelligence Platform

An AI-powered recruiter copilot that transforms a Job Description into a ranked, explainable shortlist of candidates.

The platform combines **LLM-based JD understanding, query expansion, hybrid BM25 + FAISS retrieval, metadata filtering, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and deterministic business scoring** to identify the strongest candidates for any role.

The system is designed to be **role-agnostic**: scoring and retrieval are driven by the structured Job Description rather than hardcoded rules for a specific role.

---

## 🚀 Key Capabilities

- **Structured JD Understanding** — extracts role, skills, experience, seniority, domain, responsibilities, and hiring intent.
- **Query Expansion** — generates multiple retrieval queries to improve candidate recall.
- **Hybrid Retrieval** — combines BM25 lexical search with FAISS semantic search.
- **Metadata Filtering** — removes candidates that do not satisfy applicable structured constraints.
- **Reciprocal Rank Fusion** — combines sparse and dense retrieval rankings.
- **CrossEncoder Reranking** — improves precision on the retrieved candidate pool.
- **Deterministic Business Scoring** — produces reproducible 0–100 candidate scores.
- **Explainable Recommendations** — generates grounded recruiter explanations.
- **LLM Resilience** — deterministic explanations remain available when the LLM provider fails or is rate-limited.
- **100K+ Candidate Support** — tested with approximately 100,000 candidate profiles.
- **FastAPI + Streamlit** — backend API and recruiter dashboard.
- **Dockerized Deployment** — reproducible containerized setup.

---

# 🏗️ Architecture

```text
                           ┌──────────────────────┐
                           │     Streamlit UI     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      FastAPI API     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      LangGraph       │
                           │     Orchestrator     │
                           └──────────┬───────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
           JD Parsing          Query Expansion        Retrieval
                                                           │
                                             ┌─────────────┴─────────────┐
                                             ▼                           ▼
                                           BM25                        FAISS
                                             │                           │
                                             └─────────────┬─────────────┘
                                                           ▼
                                                  Metadata Filtering
                                                           │
                                                           ▼
                                                          RRF
                                                           │
                                                           ▼
                                                  CrossEncoder
                                                           │
                                                           ▼
                                                  Top 100 Candidates
                                                           │
                                                           ▼
                                                 Business Ranker
                                                           │
                                                           ▼
                                               Explainability Layer
                                                           │
                                         ┌─────────────────┴────────────────┐
                                         ▼                                  ▼
                              Deterministic Explanations         LLM Top-5 Enhancement
                                         │                                  │
                                         └─────────────────┬────────────────┘
                                                           ▼
                                                    Recruiter Results

🧠 How the Pipeline Works
1. Job Description Understanding

A raw JD is converted into structured hiring intent, including:

Target role
Required skills
Preferred skills
Minimum experience
Seniority
Domain keywords
Industry
Location
Employment type
Responsibilities
2. Query Expansion

The structured JD is transformed into multiple retrieval queries.

These queries are used by BM25 to improve lexical recall across different ways a candidate profile may describe the same experience.

3. Hybrid Retrieval

The platform combines two complementary retrieval strategies:

BM25

Lexical matching for explicit skills, titles, technologies, and terminology.

FAISS

Semantic retrieval using normalized SentenceTransformer embeddings.

Expanded JD Queries
       │
       ├──────────► BM25
       │
       └──────────► FAISS
                     │
                     ▼
              Candidate Union
                     │
                     ▼
            Metadata Filtering
                     │
                     ▼
                    RRF
                     │
                     ▼
             CrossEncoder
4. CrossEncoder Reranking

The retrieved candidate pool is reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

This provides a higher-precision ordering before business scoring.

5. Deterministic Business Ranking

The final candidate ranking is generated using deterministic scoring logic driven by the parsed JD and candidate evidence.

📈 Scoring

The core scoring formula is:

final_score =
    (evidence_alignment × 0.30)
  + (experience_fit × 0.25)
  + (credibility × 0.20)
  + (hireability × 0.15)
  + 0.10
  - penalties

Final scores are scaled to 0–100.

The LLM does not determine the candidate's core ranking score.

This makes the ranking:

Deterministic
Reproducible
Auditable
Independent of LLM explanation quality
💡 Explainability

Every ranked candidate receives a deterministic explanation.

The top five candidates can additionally receive LLM-enhanced explanations:

Top 100 Ranked Candidates
          │
          ▼
Deterministic Explanation for All 100
          │
          ▼
LLM Enhancement for Top 5
          │
     ┌────┴────┐
     ▼         ▼
  Success    Failure
     │         │
     ▼         ▼
 Enhanced   Keep deterministic

If the LLM provider is unavailable, rate-limited, or returns malformed output, candidate ranking continues normally.

🛠️ Technology Stack
Backend
Python 3.12
FastAPI
LangGraph
Pydantic v2
Loguru
Retrieval
rank-bm25
FAISS
SentenceTransformers
all-MiniLM-L6-v2
Reciprocal Rank Fusion
Reranking
CrossEncoder
cross-encoder/ms-marco-MiniLM-L-6-v2
LLM

The platform uses an OpenAI-compatible LLM interface configured through environment variables.

The current development setup uses the Gemini OpenAI-compatible endpoint.

Frontend
Streamlit
Pandas
Deployment
Docker
Docker Compose
📁 Project Structure
talent-platform/
│
├── app/
│   ├── core/
│   ├── models/
│   ├── parser/
│   ├── retrieval/
│   ├── ranking/
│   ├── agents/
│   ├── langgraph/
│   └── api/
│
├── frontend/
│   └── streamlit_app.py
│
├── scripts/
│   ├── build_index.py
│   └── generate_candidates.py
│
├── tests/
│   ├── test_agents_and_business_ranker.py
│   ├── test_hybrid_retrieval.py
│   ├── test_langgraph_pipeline.py
│   ├── test_parser.py
│   ├── test_ranking_modules.py
│   └── test_retrieval_modules.py
│
├── data/
├── indexes/
├── docs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── .dockerignore
├── project_context.md
└── README.md

Large candidate datasets and generated retrieval indexes are intentionally excluded from the Git repository.

⚙️ Setup
1. Clone
git clone https://github.com/Gakshat12/talent-platform.git
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
🔐 Environment Configuration

Create the environment file.

Windows
Copy-Item .env.example .env
Linux / macOS
cp .env.example .env

Configure the LLM provider:

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

Index/data paths:

FAISS_INDEX_PATH=indexes/candidates.index
BM25_INDEX_PATH=indexes/candidates.bm25.json
CANDIDATES_DATA_PATH=data/candidates.jsonl

Never commit .env or API keys.

📊 Candidate Dataset

Candidate profiles are expected as JSONL:

data/candidates.jsonl

The project has been tested with approximately:

100,000 candidates

Large candidate datasets are intentionally excluded from the GitHub repository.

🧱 Build Retrieval Indexes

After providing the candidate dataset:

python -m scripts.build_index

The script creates or validates:

indexes/
├── candidates.index
├── candidates.bm25.json
└── candidates.metadata.json

Existing compatible FAISS and BM25 artifacts are reused where possible.

▶️ Run Locally
Start FastAPI
uvicorn app.api.main:app

API:

http://localhost:8000

Swagger:

http://localhost:8000/docs

Start Streamlit

In a second terminal:

streamlit run frontend/streamlit_app.py

Dashboard:

http://localhost:8501

The Streamlit client automatically uses:

Local:
http://localhost:8000

Docker:
http://api:8000

through the API_BASE_URL environment variable.

🔌 API
POST /api/v1/rank

Ranks candidates against a supplied Job Description.

Request
{
  "raw_text": "We are looking for a Machine Learning Engineer with 3+ years of experience in Python, NLP, PyTorch and production ML deployment."
}
Response

The API returns:

Parsed JD analysis
Ranked candidates
Final scores
Score breakdowns
Matched skills
Missing skills
Recruiter explanations
Ranking metadata
🐳 Docker

Build:

docker compose build

Run:

docker compose up

Services:

FastAPI
http://localhost:8000

Swagger
http://localhost:8000/docs

Streamlit
http://localhost:8501

Stop:

docker compose down

The Docker deployment mounts candidate data and persisted indexes from the host:

./data    → /app/data
./indexes → /app/indexes
🧪 Testing

Run the complete test suite:

pytest -q

Current validation:

34 passed

The tests cover:

Parser behavior
LangGraph routing
Retrieval components
Hybrid retrieval
Ranking logic
Agent behavior
Business scoring
⚡ Performance Design

The platform is designed to avoid unnecessary recomputation.

Persisted artifacts
FAISS index
BM25 corpus
Shared in-memory resources
Embedding model
CrossEncoder
Retrieval components

This allows subsequent requests in the same application process to reuse expensive retrieval resources.

🛡️ Reliability & Failure Handling

The platform is designed to degrade gracefully.

Examples:

Invalid candidate records are skipped.
Missing candidate data is handled without crashing the API.
Missing retrieval indexes can trigger controlled rebuilding.
LLM parsing failures are retried and handled.
LLM explanation failures fall back to deterministic explanations.
LLM quota/rate-limit failures do not change core candidate ranking.
Invalid or short JD inputs are validated before entering the pipeline.
📌 Example Job Description
We are looking for a Machine Learning Engineer with 3+ years of professional experience in Python, machine learning, NLP, PyTorch, and deploying production ML systems.

The candidate should have experience building and maintaining ML pipelines, developing NLP or LLM-based applications, and exposing models through backend APIs.

Experience with Docker, cloud platforms, vector databases, and model deployment is preferred.

The role involves designing ML solutions, training and evaluating models, improving inference performance, collaborating with software engineers and data scientists, and deploying reliable ML services to production.                                                    