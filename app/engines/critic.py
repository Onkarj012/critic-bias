"""
CriticEngine v2.0 - Enhanced with misattribution support.

Supports three visibility conditions:
- visible: Critic knows the actual source
- blind: Critic doesn't know the source  
- misattributed: Critic is told a false source (key for isolating brand bias)
"""

from sqlalchemy.ext.asyncio import AsyncSession
from typing import Literal

from app.llms.base import BaseLLMClient
from app.schemas.model import ModelInfo
from app.db.models import Critique, Prompt

import json


# Enhanced system prompt with clear evaluation criteria
CRITIC_SYSTEM_PROMPT_V2 = """
You are an expert evaluator of AI prompts.

A good prompt should:
- Have a clear task or instruction
- Provide necessary context
- Specify desired output format
- Be unambiguous and actionable

Evaluate the given prompt and respond STRICTLY in JSON with:
- score (float 0–10): Overall quality score
- clarity_score (float 0–10): How clear and unambiguous
- actionability_score (float 0–10): How actionable for an AI
- context_score (float 0–10): Appropriateness of provided context
- strengths (list of strings)
- weaknesses (list of strings)
- suggestions (list of strings)
- tone (one of: polite, neutral, brutal)
"""

# Legacy prompt for backward compatibility
CRITIC_SYSTEM_PROMPT = """
You are an expert evaluator of AI prompts.

Evaluate the given prompt and respond STRICTLY in JSON with:
- score (float 0–10)
- strengths (list of strings)
- weaknesses (list of strings)
- suggestions (list of strings)
- tone (one of: polite, neutral, brutal)
"""

VisibilityCondition = Literal["visible", "blind", "misattributed"]


class CriticEngine:
    """
    Responsible for evaluating prompts.
    
    v2.0 Features:
    - Misattribution condition for brand bias isolation
    - Enhanced evaluation criteria
    - Optional sub-scores (clarity, actionability, context)
    """

    def __init__(self, llm_client: BaseLLMClient, use_v2_prompt: bool = False):
        self.llm = llm_client
        self.use_v2_prompt = use_v2_prompt
    
    def _build_source_info(
        self,
        actual_source: str | None,
        visibility: VisibilityCondition,
        claimed_source: str | None = None,
    ) -> str:
        """
        Build the source attribution string based on visibility condition.
        
        Args:
            actual_source: The real source (e.g., "openai/gpt-4o" or "human")
            visibility: One of visible, blind, misattributed
            claimed_source: For misattribution, what source to claim
        """
        if visibility == "visible":
            if actual_source:
                return f"This prompt was created by {actual_source}."
            return "This prompt was created by an AI model."
        elif visibility == "blind":
            return "The source of this prompt is unknown."
        elif visibility == "misattributed":
            if not claimed_source:
                raise ValueError("claimed_source required for misattributed condition")
            return f"This prompt was created by {claimed_source}."
        else:
            raise ValueError(f"Unknown visibility condition: {visibility}")

    async def critique_prompt(
        self,
        *,
        db: AsyncSession,
        prompt: Prompt,
        critic_model: ModelInfo,
        temperature: float,
        max_tokens: int,
        seed: int,
        source_visible: bool,  # Legacy parameter
    ) -> Critique:
        """
        Legacy method for backward compatibility with v1 experiments.
        """
        visibility: VisibilityCondition = "visible" if source_visible else "blind"
        actual_source = f"{prompt.creator_provider}/{prompt.creator_model}"
        
        return await self.critique_prompt_v2(
            db=db,
            prompt=prompt,
            critic_model=critic_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            visibility=visibility,
            actual_source=actual_source,
            claimed_source=None,
            replication_id=0,
        )

    async def critique_prompt_v2(
        self,
        *,
        db: AsyncSession,
        prompt: Prompt,
        critic_model: ModelInfo,
        temperature: float,
        max_tokens: int,
        seed: int,
        visibility: VisibilityCondition,
        actual_source: str | None = None,
        claimed_source: str | None = None,
        replication_id: int = 0,
    ) -> Critique:
        """
        V2.0 critique method with full condition support.
        
        Args:
            visibility: visible, blind, or misattributed
            actual_source: Real source for tracking (not shown if blind/misattributed)
            claimed_source: What to tell the critic (for misattribution)
            replication_id: Which replication this is (for statistical analysis)
        """
        source_info = self._build_source_info(
            actual_source=actual_source,
            visibility=visibility,
            claimed_source=claimed_source,
        )
        
        system_prompt = CRITIC_SYSTEM_PROMPT_V2 if self.use_v2_prompt else CRITIC_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{source_info}\n\nPROMPT:\n{prompt.content}",
            },
        ]

        response = await self.llm.generate(
            messages=messages,
            model=critic_model.full_id(),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        parsed = self._parse_critique_response(response["content"])

        critique = Critique(
            prompt_id=str(prompt.id),
            critic_provider=critic_model.provider,
            critic_model=critic_model.model_name,
            critic_version=critic_model.version,
            source_visible=(visibility == "visible"),
            score=float(parsed["score"]),
            strengths=parsed["strengths"],
            weaknesses=parsed["weaknesses"],
            suggestions=parsed["suggestions"],
            tone=parsed.get("tone"),
            raw_text=response["content"],
            # V2 fields (stored in raw_text JSON if model doesn't support yet)
            # visibility_condition=visibility,
            # claimed_source=claimed_source,
            # replication_id=replication_id,
            # seed_used=seed,
        )

        db.add(critique)
        await db.commit()
        await db.refresh(critique)

        return critique
    
    def _parse_critique_response(self, raw_content: str) -> dict:
        """Parse structured JSON response from critique."""
        raw_content = raw_content.strip()
        
        # Default for empty responses
        if not raw_content:
            return {
                "score": 0.0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "tone": None,
            }
        
        try:
            # Extract JSON from code fences
            json_str = self._extract_json(raw_content)
            parsed = json.loads(json_str)
            
            # Ensure required fields
            parsed.setdefault("score", 0.0)
            parsed.setdefault("strengths", [])
            parsed.setdefault("weaknesses", [])
            parsed.setdefault("suggestions", [])
            
            return parsed
            
        except (json.JSONDecodeError, ValueError):
            return {
                "score": 0.0,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "tone": None,
            }
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from markdown code fences if present."""
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        return content
