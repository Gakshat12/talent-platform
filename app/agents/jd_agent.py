"""JD Understanding Agent module wrapping JDParser for extraction of job requirements."""

from typing import Any

from app.core.exceptions import JDParsingError
from app.core.logging import get_logger
from app.models.jd import ParsedJobDescription
from app.parser.jd_parser import JDParser

logger = get_logger(__name__)


class JDUnderstandingAgent:
    """Agent wrapping JDParser for extracting structured intent from raw Job Descriptions."""

    def __init__(self, parser: JDParser | None = None) -> None:
        """Initialize JDUnderstandingAgent with specified or default JDParser instance.

        Args:
            parser: Optional JDParser instance. Defaults to creating a new JDParser.
        """
        self.parser = parser or JDParser()

    def run(self, raw_jd: str) -> dict[str, Any]:
        """Parse raw job description text into structured intent dictionary.

        Args:
            raw_jd: Raw text string of the job description.

        Returns:
            Dictionary payload containing parsed_jd, confidence, skill counts, and status.
        """
        try:
            parsed_jd: ParsedJobDescription = self.parser.parse(raw_jd)
            skill_count = len(parsed_jd.skills)
            required_count = len(parsed_jd.required_skills)
            confidence = parsed_jd.confidence_score

            logger.info(
                "JDUnderstandingAgent parsed JD successfully. Title: '{}', skills: {}, required: {}, confidence: {}",
                parsed_jd.title,
                skill_count,
                required_count,
                confidence,
            )

            return {
                "status": "success",
                "parsed_jd": parsed_jd,
                "confidence": confidence,
                "skill_count": skill_count,
                "required_count": required_count,
                "error": None,
            }

        except (JDParsingError, Exception) as e:
            logger.error("JDUnderstandingAgent failed to parse JD: {}", e)
            return {
                "status": "error",
                "parsed_jd": None,
                "confidence": 0.0,
                "skill_count": 0,
                "required_count": 0,
                "error": str(e),
            }
