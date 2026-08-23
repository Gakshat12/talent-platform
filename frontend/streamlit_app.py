"""Streamlit recruiter dashboard for the Talent Intelligence Platform."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talent Intelligence Platform",
    page_icon="🎯",
    layout="wide",
)


# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
    ).strip()

    st.divider()

    st.subheader("About")

    st.markdown(
        """
        **Talent Intelligence Platform** is an AI-powered recruiter copilot that:

        - 📄 Parses job descriptions into structured intent
        - 🔍 Retrieves candidates via hybrid BM25 + dense search
        - ✅ Verifies skill claims against career history
        - 📊 Scores candidates using deterministic business logic
        - 💡 Generates recruiter-friendly explanations

        Ranking is fully deterministic — not based on keyword frequency.
        """
    )


# ── Main layout ────────────────────────────────────────────────────────────────
st.title("🎯 Talent Intelligence Platform")

st.caption(
    "AI-powered candidate ranking — backed by hybrid retrieval "
    "and evidence verification."
)

left_col, right_col = st.columns([0.4, 0.6])


# ── Left column: Input ─────────────────────────────────────────────────────────
with left_col:
    st.subheader("Job Description")

    jd_text = st.text_area(
        "Paste your job description here",
        height=320,
        placeholder=(
            "e.g. We are looking for a Senior Python Engineer with 5+ years "
            "of experience in FastAPI, PostgreSQL and cloud infrastructure..."
        ),
        key="jd_input",
    )

    find_btn = st.button(
        "🔍 Find Candidates",
        type="primary",
        use_container_width=True,
    )

    if find_btn:
        raw_jd = (jd_text or "").strip()

        if not raw_jd:
            st.error(
                "Please enter a job description before searching."
            )

        elif len(raw_jd) < 50:
            st.warning(
                f"Job description is too short ({len(raw_jd)} chars). "
                "Please provide at least 50 characters."
            )

        elif not api_url:
            st.error(
                "API Base URL cannot be empty."
            )

        else:
            with st.spinner(
                "🔄 Analysing JD and ranking candidates…"
            ):
                try:
                    response = requests.post(
                        f"{api_url.rstrip('/')}/api/v1/rank",
                        json={
                            "raw_text": raw_jd,
                            "source": "streamlit_dashboard",
                        },
                        timeout=300,
                    )

                    if response.ok:
                        payload = response.json()

                        if payload.get("success"):
                            st.session_state["ranking_result"] = (
                                payload.get("data", {})
                            )
                            st.session_state["api_error"] = None

                        else:
                            st.session_state["api_error"] = payload.get(
                                "error",
                                "Unknown API error.",
                            )
                            st.session_state["ranking_result"] = None

                    else:
                        try:
                            error_payload = response.json()

                            detail = error_payload.get(
                                "detail",
                                error_payload.get(
                                    "error",
                                    response.text,
                                ),
                            )
                        except Exception:
                            detail = response.text

                        st.session_state["api_error"] = (
                            f"API error {response.status_code}: {detail}"
                        )

                        st.session_state["ranking_result"] = None

                except requests.exceptions.ConnectionError:
                    st.session_state["api_error"] = (
                        f"Could not connect to API at {api_url}. "
                        "Is the server running?"
                    )
                    st.session_state["ranking_result"] = None

                except requests.exceptions.Timeout:
                    st.session_state["api_error"] = (
                        "Request timed out after 300 seconds. "
                        "The ranking pipeline may still be running."
                    )
                    st.session_state["ranking_result"] = None

                except requests.exceptions.RequestException as exc:
                    st.session_state["api_error"] = (
                        f"Request failed: {exc}"
                    )
                    st.session_state["ranking_result"] = None

                except ValueError as exc:
                    st.session_state["api_error"] = (
                        f"Invalid JSON response from API: {exc}"
                    )
                    st.session_state["ranking_result"] = None

                except Exception as exc:
                    st.session_state["api_error"] = (
                        f"Unexpected error: {exc}"
                    )
                    st.session_state["ranking_result"] = None


# ── Right column: Results ──────────────────────────────────────────────────────
with right_col:
    api_error = st.session_state.get(
        "api_error"
    )

    result_data = st.session_state.get(
        "ranking_result"
    )

    if api_error:
        st.error(
            f"❌ {api_error}"
        )

    elif result_data:
        job_title = result_data.get(
            "job_title",
            "N/A",
        )

        total = result_data.get(
            "total_matching_candidates",
            0,
        )

        ranked = result_data.get(
            "ranked_candidates",
            [],
        ) or []

        jd_analysis = result_data.get(
            "jd_analysis"
        ) or {}

        # ── JD Analysis ────────────────────────────────────────────────────────
        st.subheader("📋 JD Analysis")

        parsed = jd_analysis.get(
            "parsed_jd",
            {},
        ) or {}

        experience = parsed.get(
            "experience",
            {},
        ) or {}

        min_yrs = experience.get(
            "min_years",
            0,
        )

        max_yrs = experience.get(
            "max_years"
        )

        seniority = experience.get(
            "seniority_level",
            "N/A",
        )

        required_skills = [
            skill
            for skill in parsed.get("skills", [])
            if skill.get("is_required")
        ]

        preferred_skills = [
            skill
            for skill in parsed.get("skills", [])
            if not skill.get("is_required")
        ]

        meta_cols = st.columns(4)

        meta_cols[0].metric(
            "Role",
            job_title,
        )

        exp_label = (
            f"{min_yrs}–{max_yrs} yrs"
            if max_yrs is not None
            else f"{min_yrs}+ yrs"
        )

        meta_cols[1].metric(
            "Experience",
            exp_label,
        )

        meta_cols[2].metric(
            "Required Skills",
            len(required_skills),
        )

        meta_cols[3].metric(
            "Total Candidates",
            total,
        )

        if seniority and seniority != "N/A":
            st.caption(
                f"Seniority: **{seniority}**"
            )

        if preferred_skills:
            st.caption(
                "Preferred skills: "
                + ", ".join(
                    skill.get("name", "")
                    for skill in preferred_skills
                    if skill.get("name")
                )
            )

        st.divider()

        # ── CSV download ───────────────────────────────────────────────────────
        if ranked:
            rows = []

            for candidate in ranked:
                score_breakdown = candidate.get(
                    "score_breakdown",
                    {},
                ) or {}

                rows.append(
                    {
                        "Rank": candidate.get(
                            "rank"
                        ),
                        "Name": candidate.get(
                            "candidate_name",
                            "",
                        ),
                        "Final Score": score_breakdown.get(
                            "final_score",
                            0,
                        ),
                        "Evidence Alignment": score_breakdown.get(
                            "evidence_alignment",
                            0,
                        ),
                        "Experience Fit": score_breakdown.get(
                            "experience_fit",
                            0,
                        ),
                        "Credibility": score_breakdown.get(
                            "credibility",
                            0,
                        ),
                        "Hireability": score_breakdown.get(
                            "hireability",
                            0,
                        ),
                        "Matched Skills": ", ".join(
                            candidate.get(
                                "matched_skills",
                                [],
                            )
                        ),
                        "Missing Skills": ", ".join(
                            candidate.get(
                                "missing_skills",
                                [],
                            )
                        ),
                        "Explanation": candidate.get(
                            "explanation",
                            "",
                        ),
                    }
                )

            dataframe = pd.DataFrame(rows)

            csv_bytes = dataframe.to_csv(
                index=False
            ).encode("utf-8")

            safe_title = (
                str(job_title)
                .replace(" ", "_")
                .replace("/", "_")
                .lower()
            )

            st.download_button(
                label="⬇️ Download Ranked Results CSV",
                data=csv_bytes,
                file_name=f"ranked_{safe_title}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()

        # ── Top candidates ─────────────────────────────────────────────────────
        shown_count = min(
            len(ranked),
            20,
        )

        st.subheader(
            f"🏆 Top Candidates ({shown_count} shown)"
        )

        for candidate in ranked[:20]:
            rank = candidate.get(
                "rank",
                "?",
            )

            name = candidate.get(
                "candidate_name",
                "Unknown",
            )

            score_breakdown = candidate.get(
                "score_breakdown",
                {},
            ) or {}

            final_score = float(
                score_breakdown.get(
                    "final_score",
                    0.0,
                )
            )

            explanation = candidate.get(
                "explanation",
                "",
            )

            matched = candidate.get(
                "matched_skills",
                [],
            ) or []

            missing = candidate.get(
                "missing_skills",
                [],
            ) or []

            score_color = (
                "🟢"
                if final_score >= 60
                else "🟡"
                if final_score >= 40
                else "🔴"
            )

            label = (
                f"#{rank}  {name}  —  "
                f"{score_color} {final_score:.1f}/100"
            )

            try:
                expanded = int(rank) <= 3
            except (TypeError, ValueError):
                expanded = False

            with st.expander(
                label,
                expanded=expanded,
            ):
                exp_col, score_col = st.columns(
                    [0.55, 0.45]
                )

                with exp_col:
                    if explanation:
                        st.markdown(
                            "**💬 Recruiter Note**"
                        )
                        st.info(explanation)

                    if matched:
                        st.success(
                            "✅ Matched skills: "
                            + ", ".join(matched)
                        )

                    if missing:
                        st.warning(
                            "⚠️ Missing skills: "
                            + ", ".join(missing)
                        )

                with score_col:
                    st.markdown(
                        "**📊 Score Breakdown**"
                    )

                    components = [
                        (
                            "Evidence Alignment",
                            score_breakdown.get(
                                "evidence_alignment",
                                0,
                            ),
                            0.30,
                        ),
                        (
                            "Experience Fit",
                            score_breakdown.get(
                                "experience_fit",
                                0,
                            ),
                            0.25,
                        ),
                        (
                            "Credibility",
                            score_breakdown.get(
                                "credibility",
                                0,
                            ),
                            0.20,
                        ),
                        (
                            "Hireability",
                            score_breakdown.get(
                                "hireability",
                                0,
                            ),
                            0.15,
                        ),
                    ]

                    for (
                        component_name,
                        component_score,
                        component_weight,
                    ) in components:
                        component_score = float(
                            component_score or 0.0
                        )

                        st.caption(
                            f"{component_name} "
                            f"(weight {int(component_weight * 100)}%): "
                            f"{component_score:.2f}"
                        )

                        st.progress(
                            min(
                                max(
                                    component_score,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                    penalty = float(
                        score_breakdown.get(
                            "penalty_deduction",
                            0.0,
                        ) or 0.0
                    )

                    if penalty > 0.0:
                        st.caption(
                            f"Penalty deduction: −{penalty:.2f}"
                        )

    else:
        st.info(
            "👈 Enter a job description on the left and click "
            "**Find Candidates** to begin."
        )