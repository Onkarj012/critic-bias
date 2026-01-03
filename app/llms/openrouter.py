"""
OpenRouter LLM client with retry logic and error handling.
"""

import asyncio
import logging
import httpx
from typing import List

from app.llms.base import BaseLLMClient
from app.llms.types import Message, LLMResponse
from app.core.settings import settings
from app.core.exceptions import LLMAPIError, RateLimitError

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
TIMEOUT_SECONDS = 90.0


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter API client with retry logic and rate limit handling.
    """
    
    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/critiq-bias",  # Required by OpenRouter
        }
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        *,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        """
        Generate response from OpenRouter API with retry logic.
        
        Implements exponential backoff for transient failures and
        respects rate limit headers.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if seed is not None:
            payload["seed"] = seed

        last_error: Exception | None = None
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                
                logger.debug(f"API call attempt {attempt}/{MAX_RETRIES} to {model}")
                
                response = await client.post(
                    f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )

                # Handle rate limits
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", str(backoff))
                    wait_time = float(retry_after)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue

                # Handle other errors
                if response.status_code >= 500:
                    logger.warning(f"Server error ({response.status_code}). Retrying...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue
                
                # Client errors - don't retry
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", {}).get("message", response.text)
                    raise LLMAPIError(
                        f"API error: {error_msg}",
                        status_code=response.status_code,
                        provider="openrouter"
                    )

                # Success
                data = response.json()
                
                if not data.get("choices"):
                    raise LLMAPIError("No choices in API response", provider="openrouter")
                
                choice = data["choices"][0]["message"]
                usage = data.get("usage", {})

                logger.info(
                    f"API call successful: {model} "
                    f"(in: {usage.get('prompt_tokens', '?')}, "
                    f"out: {usage.get('completion_tokens', '?')})"
                )

                return {
                    "content": choice["content"],
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "model": data.get("model", model),
                }

            except httpx.TimeoutException as e:
                logger.warning(f"Timeout on attempt {attempt}: {e}")
                last_error = LLMAPIError(f"Request timed out: {e}", provider="openrouter")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

            except httpx.RequestError as e:
                logger.warning(f"Network error on attempt {attempt}: {e}")
                last_error = LLMAPIError(f"Network error: {e}", provider="openrouter")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

            except LLMAPIError:
                raise  # Don't retry client errors

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {e}")
                last_error = LLMAPIError(f"Unexpected error: {e}", provider="openrouter")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

        # All retries exhausted
        raise LLMAPIError(
            f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}",
            provider="openrouter"
        )
