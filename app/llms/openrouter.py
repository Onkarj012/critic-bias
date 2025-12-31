import httpx
from typing import List

from app.llms.base import BaseLLMClient
from app.llms.types import Message, LLMResponse
from app.core.settings import settings

OPENROUTER_API_KEY: str = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL: str = "https://api.openrouter.ai/v1"

class OpenRouterClient(BaseLLMClient):
    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        *,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if seed is not None:
            payload["seed"] = seed

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers=self._headers,
                json=payload,
            )

        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]

        usage = data.get("usage", {})

        return {
            "content": choice["content"],
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "model": data.get("model", model),
        }
