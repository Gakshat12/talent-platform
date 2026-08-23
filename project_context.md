# Project Context — Talent Intelligence Platform

## Problem
Recruiters waste hours manually reviewing irrelevant candidates. Traditional ATS
systems use keyword matching — a Marketing Manager with 9 AI keywords ranks above
a genuine ML Engineer. This breaks hiring.

## Solution
AI-powered recruiter copilot that understands JD intent, retrieves candidates
semantically, verifies skill claims against career history, and explains every
recommendation with specific evidence.

## Pipeline (5 stages)
1. JD Parsing → structured intent extraction (title, skills, experience band, domain)
2. Hybrid Retrieval → BM25 + FAISS dense search + metadata filter + RRF fusion + cross-encoder rerank
3. Evidence Verification → skill claims cross-checked against career history descriptions
4. Business Scoring → weighted sum of evidence alignment, experience fit, credibility, hireability
5. Explainability → per-candidate reasoning citing actual career evidence

## Key design decisions
- Deterministic scoring decoupled from LLM agents — ranking is reproducible
- Evidence alignment is the primary signal — not keyword frequency
- Two-stage retrieval funnel — recall (BM25+FAISS) then precision (cross-encoder)
- LangGraph conditional edges — retry on low-confidence JD parse, skip evidence agent if >50 candidates
- Generalises to any JD — zero hardcoded role-specific rules

## File-by-file purpose
| File | Purpose |
|------|---------|
| app/core/config.py | Pydantic settings loaded from .env |
| app/core/logging.py | Loguru-based structured logger |
| app/core/exceptions.py | Custom domain exceptions |
| app/models/candidate.py | CandidateProfile + WorkExperience Pydantic models |
| app/models/jd.py | ParsedJobDescription + SkillRequirement + ExperienceRequirement |
| app/models/response.py | ScoreBreakdown + CandidateRankResult + APIResponse |
| app/langgraph/state.py | TalentGraphState TypedDict |
| app/parser/jd_parser.py | LLM call → ParsedJobDescription |
| app/parser/query_expansion.py | Expands JD into 5 retrieval queries |
| app/retrieval/bm25.py | BM25Okapi sparse retrieval |
| app/retrieval/embeddings.py | SentenceTransformer embedding service |
| app/retrieval/faiss_index.py | FAISS IndexFlatIP dense retrieval |
| app/retrieval/metadata_filter.py | Hard constraint filtering |
| app/retrieval/reciprocal_rank_fusion.py | RRF merge of ranked lists |
| app/retrieval/reranker.py | CrossEncoder reranking |
| app/retrieval/hybrid_retrieval.py | Full retrieval pipeline orchestrator |
| app/ranking/experience_fit.py | Years of experience scorer |
| app/ranking/evidence_alignment.py | Required skill evidence scorer |
| app/ranking/credibility.py | Skill-to-career-evidence scorer |
| app/ranking/hireability.py | Career progression scorer |
| app/ranking/penalties.py | Penalty deduction calculator |
| app/ranking/final_score.py | Weighted aggregator → ScoreBreakdown |
| app/ranking/business_ranker.py | Ranks all candidates, assigns ranks 1-N |
| app/agents/jd_agent.py | JD Understanding Agent (LangGraph node wrapper) |
| app/agents/evidence_agent.py | Evidence Verification Agent |
| app/agents/explainability_agent.py | Explanation generator |
| app/agents/supervisor.py | Supervisor agent (routes between nodes) |
| app/langgraph/nodes.py | All 6 LangGraph node functions |
| app/langgraph/edges.py | Conditional edge functions |
| app/langgraph/graph.py | Graph assembly + compile |
| app/api/routes.py | FastAPI route handlers |
| app/api/main.py | FastAPI app factory |
| frontend/streamlit_app.py | Recruiter dashboard UI |
