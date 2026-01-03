"""
Caching layer for CRITIQ-BIAS.

Caches LLM responses to avoid duplicate API calls and reduce costs.
Cache key is based on: model_id + prompt_content + temperature + seed
"""

import hashlib
import json
import logging
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime

from app.db.models import Prompt, Critique

logger = logging.getLogger(__name__)


def generate_cache_key(
    model_id: str,
    messages: list[dict],
    temperature: float,
    seed: int | None = None,
) -> str:
    """Generate a deterministic cache key from request parameters."""
    key_data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()


class ResponseCache:
    """
    Cache for LLM responses.
    
    Uses database to store cached responses, enabling persistence across runs.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._memory_cache: dict[str, dict] = {}  # In-memory cache for current session
    
    async def get_creator_response(
        self,
        *,
        run_id: str,
        task_id: str,
        creator_model: str,
        creator_provider: str,
    ) -> Prompt | None:
        """
        Check if we already have a prompt from this creator for this task.
        Returns the cached Prompt if found, None otherwise.
        """
        stmt = select(Prompt).where(
            and_(
                Prompt.run_id == run_id,
                Prompt.task_id == task_id,
                Prompt.creator_model == creator_model,
                Prompt.creator_provider == creator_provider,
            )
        )
        result = await self.db.execute(stmt)
        cached = result.scalar_one_or_none()
        
        if cached:
            logger.info(f"Cache HIT: prompt from {creator_provider}/{creator_model}")
        
        return cached
    
    async def get_critique_response(
        self,
        *,
        prompt_id: str,
        critic_model: str,
        critic_provider: str,
        source_visible: bool,
    ) -> Critique | None:
        """
        Check if we already have a critique for this prompt/critic/condition combo.
        Returns the cached Critique if found, None otherwise.
        """
        stmt = select(Critique).where(
            and_(
                Critique.prompt_id == prompt_id,
                Critique.critic_model == critic_model,
                Critique.critic_provider == critic_provider,
                Critique.source_visible == source_visible,
            )
        )
        result = await self.db.execute(stmt)
        cached = result.scalar_one_or_none()
        
        if cached:
            logger.info(
                f"Cache HIT: critique from {critic_provider}/{critic_model} "
                f"(visible={source_visible})"
            )
        
        return cached
    
    def get_memory(self, cache_key: str) -> dict | None:
        """Get from in-memory cache (fast, session-only)."""
        return self._memory_cache.get(cache_key)
    
    def set_memory(self, cache_key: str, response: dict) -> None:
        """Store in in-memory cache."""
        self._memory_cache[cache_key] = response
    
    def clear_memory(self) -> None:
        """Clear in-memory cache."""
        self._memory_cache.clear()
    
    @staticmethod
    def make_key(
        model_id: str,
        messages: list[dict],
        temperature: float,
        seed: int | None = None,
    ) -> str:
        """Generate cache key for a request."""
        return generate_cache_key(model_id, messages, temperature, seed)


class CacheStats:
    """Track cache hit/miss statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.saved_calls = 0
    
    def record_hit(self):
        self.hits += 1
        self.saved_calls += 1
    
    def record_miss(self):
        self.misses += 1
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def summary(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "saved_api_calls": self.saved_calls,
        }
    
    def __str__(self) -> str:
        return (
            f"Cache Stats: {self.hits} hits, {self.misses} misses "
            f"({self.hit_rate:.1%} hit rate, {self.saved_calls} API calls saved)"
        )
