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
        h = int(hashlib.sha256(joined.encode()).hexdigest(), 16)
        
        # Deterministic pseudo-random score based on hash
        score = (h % 100) / 10.0
        
        content = f"""
```json
{{
  "score": {score},
  "strengths": ["Good point {h % 5}", "Valid reasoning"],
  "weaknesses": ["Could be better {h % 3}"],
  "suggestions": ["Improve X"],
  "tone": "neutral"
}}
```
"""
        return {
            "content": content,
            "input_tokens": len(joined.split()),
            "output_tokens": 50,
            "model": f"mock-{model}",
        }