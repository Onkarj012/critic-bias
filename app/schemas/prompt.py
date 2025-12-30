from app.schemas.base import BaseSchema
from app.schemas.model import ModelInfo
from uuid import UUID

class PromptSchema(BaseSchema):
    task_id: UUID
    creator_model: ModelInfo

    content: str

    token_count: int | None = None