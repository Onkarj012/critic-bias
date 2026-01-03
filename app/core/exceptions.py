"""
Custom exceptions for CRITIQ-BIAS.
"""


class CritiqBiasError(Exception):
    """Base exception for all CRITIQ-BIAS errors."""
    pass


class LLMAPIError(CritiqBiasError):
    """Raised when LLM API call fails."""
    
    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        self.status_code = status_code
        self.provider = provider
        super().__init__(message)


class RateLimitError(LLMAPIError):
    """Raised when rate limited by LLM provider."""
    
    def __init__(self, message: str, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__(message, status_code=429)


class ExperimentError(CritiqBiasError):
    """Raised when experiment execution fails."""
    
    def __init__(self, message: str, run_id: str | None = None, step: str | None = None):
        self.run_id = run_id
        self.step = step
        super().__init__(message)


class CacheError(CritiqBiasError):
    """Raised when cache operations fail."""
    pass


class MetricError(CritiqBiasError):
    """Raised when metric computation fails."""
    
    def __init__(self, message: str, metric_name: str | None = None):
        self.metric_name = metric_name
        super().__init__(message)


class ConfigurationError(CritiqBiasError):
    """Raised when configuration is invalid."""
    pass
