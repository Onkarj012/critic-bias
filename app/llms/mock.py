from app.llms.base import BaseLLMClient
from app.llms.types import Message, LLMResponse
from typing import List
import hashlib

class MockLLMClient(BaseLLMClient):

    async def generate(
        self,
        *,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        joined = " ".join(m["content"] for m in messages)
        h = hashlib.sha256(joined.encode()).hexdigest()[:16]

        return {
            "content": f"MOCK RESPONSE: {h}",
            "input_tokens": len(joined.split()),
            "output_tokens": 16,
            "model": f"mock-{model}",
        }