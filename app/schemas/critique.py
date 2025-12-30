from app.schemas.base import BaseSchema
from app.schemas.model import ModelInfo
from uuid import UUID
from typing import Literal

class CritiqueSchema(BaseSchema):
    prompt_id: UUID
    critic_model: ModelInfo

    source_visible: Literal[True, False]
    
    score: float
    strength: list[str]
    weakness: list[str]
    suggestions: list[str]

    tone: Literal["neutral", "positive", "negative"] | None = None

    raw_text: str