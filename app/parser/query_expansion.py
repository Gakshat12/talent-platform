"""Query Expansion module using OpenRouter API to expand Job Descriptions into retrieval queries."""

import json

from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)

QUERY_EXPANSION_SYSTEM_PROMPT = """You are a specialized recruiter search assistant.
Given a structured Job Description, generate exactly 5 search queries for retrieving candidate profiles.

The 5 queries MUST follow these specific strategies:
1. Exact job title + top 3 required skills
2. Synonym or alternative job titles
3. Required skills-focused query
4. Key responsibility-focused query
5. Domain keywords + target seniority level query

Return ONLY a JSON array of 5 strings. No explanation, no markdown, no backticks.

Example output:
[
  "Senior Machine Learning Engineer Python PyTorch LLM",
  "Lead AI Engineer Principal ML Specialist NLP",
  "Python PyTorch Vector Search Distributed Systems",
  "Design deploy production ML models optimize inference latency",
  "Artificial Intelligence NLP Senior"
]
"""


class QueryExpander:
    """Expands structured Job Description into multiple search queries for candidate retrieval."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize QueryExpander with OpenRouter credentials and client settings.

        Args:
            api_key: OpenRouter API key. Defaults to settings.openrouter_api_key.
            base_url: OpenRouter base API URL. Defaults to settings.openrouter_base_url.
            model: Target LLM model name. Defaults to settings.llm_model.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.llm_model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _build_prompt(self, parsed_jd: ParsedJobDescription) -> str:
        """Construct user prompt from ParsedJobDescription attributes.

        Args:
            parsed_jd: Structured job description object.

        Returns:
            Formatted prompt string describing job intent for query generation.
        """
        top_required_skills = parsed_jd.required_skill_names[:3]
        return f"""Job Title: {parsed_jd.title}
Seniority Level: {parsed_jd.experience.seniority_level}
Required Skills: {', '.join(parsed_jd.required_skill_names)}
Top 3 Required Skills: {', '.join(top_required_skills)}
Domain Keywords: {', '.join(parsed_jd.domain_keywords)}
Responsibilities: {'; '.join(parsed_jd.responsibilities)}
Summary: {parsed_jd.summary}

Generate 5 alternative search queries following the 5 requested strategies. Return ONLY the JSON array of strings."""

    def _get_fallback_query(self, parsed_jd: ParsedJobDescription) -> list[str]:
        """Generate single fallback search query if LLM call or response parsing fails.

        Args:
            parsed_jd: Structured job description object.

        Returns:
            List containing one fallback query formatted as '{title} {required_skill_names}'.
        """
        skill_str = " ".join(parsed_jd.required_skill_names)
        fallback = f"{parsed_jd.title} {skill_str}".strip()
        if not fallback:
            fallback = "Candidate"
        return [fallback]

    def expand(self, parsed_jd: ParsedJobDescription) -> list[str]:
        """Generate exactly 5 alternative retrieval queries derived from parsed JD intent.

        Args:
            parsed_jd: Structured job description object.

        Returns:
            List of search query strings. Returns fallback query on LLM error or malformed response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(parsed_jd)},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            content = response.choices[0].message.content or ""
            clean_content = content.strip()

            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                    clean_content = "\n".join(lines[1:-1]).strip()

            data = json.loads(clean_content)
            if isinstance(data, list) and all(isinstance(item, str) for item in data) and len(data) > 0:
                queries = [q.strip() for q in data if q.strip()]
                if queries:
                    logger.debug(f"Generated expanded queries: {queries}")
                    return queries

            logger.warning("LLM response for query expansion was not a valid list of strings. Using fallback.")
            fallback = self._get_fallback_query(parsed_jd)
            logger.debug(f"Generated expanded queries (fallback): {fallback}")
            return fallback

        except Exception as e:
            logger.warning(f"Error during query expansion LLM call: {e}. Using fallback.")
            fallback = self._get_fallback_query(parsed_jd)
            logger.debug(f"Generated expanded queries (fallback): {fallback}")
            return fallback
