"""Scratch unit tests for JDParser and QueryExpander."""

import json
from unittest.mock import MagicMock, patch
import pytest

from app.core.exceptions import JDParsingError
from app.models.jd import ParsedJobDescription, SkillRequirement, ExperienceRequirement
from app.parser.jd_parser import JDParser
from app.parser.query_expansion import QueryExpander


def test_jd_parser_empty_input():
    parser = JDParser()
    with pytest.raises(JDParsingError) as exc_info:
        parser.parse("   ")
    assert "cannot be empty" in str(exc_info.value)


@patch("app.parser.jd_parser.OpenAI")
def test_jd_parser_success(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "title": "Senior Python Developer",
            "summary": "Building scalable backend microservices",
            "skills": [
                {"name": "Python", "is_required": True, "min_years": 5.0},
                {"name": "FastAPI", "is_required": True, "min_years": 3.0}
            ],
            "experience": {"min_years": 5.0, "seniority_level": "senior"},
            "domain_keywords": ["Backend", "Microservices"],
            "responsibilities": ["Develop REST APIs"],
            "confidence_score": 0.95
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    parser = JDParser(api_key="test-key")
    parsed_jd = parser.parse("Looking for Senior Python Developer with 5 years experience...")
    
    assert isinstance(parsed_jd, ParsedJobDescription)
    assert parsed_jd.title == "Senior Python Developer"
    assert len(parsed_jd.skills) == 2
    assert parsed_jd.required_skill_names == ["Python", "FastAPI"]
    assert parsed_jd.confidence_score == 0.95


@patch("app.parser.jd_parser.OpenAI")
def test_jd_parser_malformed_json(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Invalid JSON string"))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    parser = JDParser(api_key="test-key")
    with pytest.raises(JDParsingError) as exc_info:
        parser.parse("Python Dev JD")
    assert "Malformed JSON" in str(exc_info.value)


@patch("app.parser.jd_parser.OpenAI")
def test_jd_parser_api_error(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("API Connection Error")

    parser = JDParser(api_key="test-key")
    with pytest.raises(JDParsingError) as exc_info:
        parser.parse("Python Dev JD")
    assert "OpenRouter API failure" in str(exc_info.value)


@patch("app.parser.query_expansion.OpenAI")
def test_query_expander_success(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    expected_queries = [
        "Senior Python Developer Python FastAPI PostgreSQL",
        "Lead Backend Engineer Senior Software Engineer",
        "Python FastAPI AsyncIO Microservices",
        "Build scalable REST APIs design database schemas",
        "Backend Architecture Senior 5 years"
    ]
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(expected_queries)))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    expander = QueryExpander(api_key="test-key")
    mock_jd = ParsedJobDescription(
        title="Senior Python Developer",
        skills=[
            SkillRequirement(name="Python", is_required=True),
            SkillRequirement(name="FastAPI", is_required=True),
            SkillRequirement(name="PostgreSQL", is_required=True)
        ],
        experience=ExperienceRequirement(min_years=5.0, seniority_level="senior"),
        domain_keywords=["Backend"],
        responsibilities=["Build scalable REST APIs"]
    )
    
    queries = expander.expand(mock_jd)
    assert len(queries) == 5
    assert queries == expected_queries


@patch("app.parser.query_expansion.OpenAI")
def test_query_expander_fallback_on_error(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("OpenRouter Timeout")

    expander = QueryExpander(api_key="test-key")
    mock_jd = ParsedJobDescription(
        title="Senior Python Developer",
        skills=[
            SkillRequirement(name="Python", is_required=True),
            SkillRequirement(name="FastAPI", is_required=True)
        ]
    )
    
    queries = expander.expand(mock_jd)
    assert len(queries) == 1
    assert queries == ["Senior Python Developer Python FastAPI"]
