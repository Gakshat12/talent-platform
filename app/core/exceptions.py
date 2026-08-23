"""Custom exception hierarchy for the AI Talent Intelligence Platform."""


class TalentPlatformError(Exception):
    """Base exception for all domain-specific errors in the Talent Platform."""

    def __init__(self, message: str = "An unexpected error occurred in the Talent Platform."):
        self.message = message
        super().__init__(self.message)


class JDParsingError(TalentPlatformError):
    """Raised when job description parsing or intent extraction fails."""

    def __init__(self, message: str = "Failed to parse the job description."):
        super().__init__(message)


class RetrievalError(TalentPlatformError):
    """Raised when candidate retrieval (sparse, dense, or hybrid) encounters an error."""

    def __init__(self, message: str = "Error occurred during candidate retrieval."):
        super().__init__(message)


class ScoringError(TalentPlatformError):
    """Raised when candidate evaluation or scoring fails."""

    def __init__(self, message: str = "Error occurred during candidate scoring calculation."):
        super().__init__(message)


class EvidenceVerificationError(TalentPlatformError):
    """Raised when skill claim cross-checking or evidence verification fails."""

    def __init__(self, message: str = "Error occurred during candidate evidence verification."):
        super().__init__(message)


class GraphExecutionError(TalentPlatformError):
    """Raised when a specific LangGraph node workflow fails during execution."""

    def __init__(
        self,
        message: str = "Error executing workflow graph node.",
        node_name: str = "unknown",
    ):
        self.node_name = node_name
        full_message = f"Node '{node_name}' error: {message}"
        super().__init__(full_message)


class IndexNotFoundError(TalentPlatformError):
    """Raised when required FAISS vector index file is missing or unreadable."""

    def __init__(self, message: str = "Candidate FAISS index file was not found."):
        super().__init__(message)
