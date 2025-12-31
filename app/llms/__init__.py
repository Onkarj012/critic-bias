from app.llms.openrouter import OpenRouterClient
from app.llms.mock import MockLLMClient
from app.llms.base import BaseLLMClient


def get_llm_client(kind: str) -> BaseLLMClient:
    if kind == "openrouter":
        return OpenRouterClient()
    if kind == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM client type: {kind}")
