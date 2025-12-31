from abc import ABC, abstractmethod
from typing import List
from app.llms.types import Message, LLMResponse

class BaseLLMClient(ABC):

    @abstractmethod
    async def generate(
        self, 
        *,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError        