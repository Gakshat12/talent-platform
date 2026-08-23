"""Job Description parser module using the OpenRouter OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import JDParsingError
from app.core.logging import get_logger
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)


JD_PARSER_SYSTEM_PROMPT = """Parse the provided Job Description into the required schema.

Return ONLY one valid JSON object. No markdown, no code fences, no explanation.

Required fields:
- title: string
- summary: concise string
- skills: array of objects with name, is_required, min_years, category
- experience: object with min_years, max_years, seniority_level
- domain_keywords: concise array of strings
- industry: string or null
- location: string or null
- employment_type: string or null
- responsibilities: concise array of strings
- confidence_score: number from 0.0 to 1.0

Rules:
- Use only information supported by the JD.
- Default skill min_years to 0.0 when unspecified.
- Default experience min_years to 0.0 when unspecified.
- Default experience seniority_level to "mid" when unspecified.
- Keep summary concise.
- Keep responsibilities concise.
- Keep domain_keywords focused.
- Do not invent requirements.
- Always return syntactically valid JSON.
"""


class JDParser:
    """Parse raw job descriptions into validated structured job requirements."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the parser and configure its OpenRouter-compatible client.

        Args:
            api_key: OpenRouter API key. Defaults to configured settings.
            base_url: OpenRouter-compatible API base URL.
            model: LLM model name to use for parsing.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.llm_model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
        )

    def _build_prompt(self, raw_jd: str) -> str:
        """Build a compact user prompt focused on structured JSON extraction.

        Args:
            raw_jd: Raw job description supplied by the recruiter.

        Returns:
            Prompt text passed to the LLM.
        """
        return (
            "Extract the structured job requirements from this JD. "
            "Return one compact valid JSON object only. "
            "Do not add explanations or extra text.\n\n"
            f"{raw_jd.strip()}"
        )

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        """Extract one JSON object from the raw LLM response.

        Args:
            content: Raw response returned by the LLM.

        Returns:
            Parsed JSON object.

        Raises:
            json.JSONDecodeError: If no valid JSON object can be extracted.
        """
        text = (content or "").strip()

        if not text:
            raise json.JSONDecodeError(
                "LLM returned an empty response.",
                text,
                0,
            )

        # Remove markdown code fences if the model ignores the instruction.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        # Fast path: the entire response is JSON.
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

            raise json.JSONDecodeError(
                "Expected a JSON object.",
                text,
                0,
            )
        except json.JSONDecodeError:
            pass

        # Fallback: locate the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise json.JSONDecodeError(
                "No JSON object found in LLM response.",
                text,
                0,
            )

        candidate = text[start : end + 1]
        parsed = json.loads(candidate)

        if not isinstance(parsed, dict):
            raise json.JSONDecodeError(
                "Expected a JSON object.",
                candidate,
                0,
            )

        return parsed

    def _request_parse(
        self,
        raw_jd: str,
        max_tokens: int,
    ) -> str:
        """Request one structured JD parse from the configured OpenRouter model.

        Args:
            raw_jd: Raw job description.
            max_tokens: Maximum completion token budget.

        Returns:
            Raw LLM response content.

        Raises:
            JDParsingError: If the provider returns no usable completion.
        """
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": JD_PARSER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self._build_prompt(raw_jd),
                },
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens,
        }

        # Prefer JSON response mode. Some OpenRouter models may not support it,
        # so fall back to a normal completion request.
        try:
            response = self.client.chat.completions.create(
                **request_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as first_error:
            logger.debug(
                "JSON response mode unavailable; retrying standard completion: {}",
                first_error,
            )

            response = self.client.chat.completions.create(
                **request_kwargs,
            )

        if not response.choices:
            raise JDParsingError(
                "LLM returned no completion choices."
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        logger.debug(
            "JD parser response received. finish_reason={}, response_length={}",
            finish_reason,
            len(content),
        )

        if finish_reason == "length":
            logger.warning(
                "JD parser response reached the token limit."
            )

        return content

    def parse(self, raw_jd: str) -> ParsedJobDescription:
        """Parse raw JD text into a validated ParsedJobDescription.

        Args:
            raw_jd: Raw job description text.

        Returns:
            Validated ParsedJobDescription.

        Raises:
            JDParsingError: If the input, provider response, JSON, or schema is invalid.
        """
        if not raw_jd or not raw_jd.strip():
            logger.error(
                "Job description input text is empty or whitespace."
            )
            raise JDParsingError(
                "Job description text cannot be empty."
            )

        # Keep the output budget bounded. Two attempts use the same budget
        # instead of doubling the number of requested tokens.
        max_tokens = min(
            max(int(settings.llm_max_tokens), 2048),
            4096,
        )

        for attempt in range(1, 3):
            try:
                content = self._request_parse(
                    raw_jd=raw_jd,
                    max_tokens=max_tokens,
                )

                try:
                    data = self._extract_json_object(content)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "JD parser returned malformed JSON on attempt {}: {}",
                        attempt,
                        exc,
                    )

                    if attempt == 1:
                        continue

                    preview = (
                        (content or "")
                        .replace("\n", " ")
                        .strip()[:300]
                    )

                    raise JDParsingError(
                        f"Malformed JSON response from LLM: {exc}. "
                        f"Response preview: {preview}"
                    ) from exc

                try:
                    parsed_jd = ParsedJobDescription.model_validate(data)
                except Exception as exc:
                    logger.error(
                        "ParsedJobDescription validation failed on attempt {}: {}",
                        attempt,
                        exc,
                    )

                    raise JDParsingError(
                        f"Parsed Job Description validation error: {exc}"
                    ) from exc

                logger.info(
                    "Parsed job description successfully: "
                    "title='{}', skills_count={}, attempt={}",
                    parsed_jd.title,
                    len(parsed_jd.skills),
                    attempt,
                )

                return parsed_jd

            except JDParsingError:
                if attempt == 2:
                    raise

            except Exception as exc:
                logger.warning(
                    "OpenRouter JD parsing attempt {} failed: {}",
                    attempt,
                    exc,
                )

                if attempt == 2:
                    raise JDParsingError(
                        f"OpenRouter API failure during JD parsing: {exc}"
                    ) from exc

        raise JDParsingError(
            "JD parsing failed after all retry attempts."
        )