from app.schemas.base import BaseSchema
from uuid import UUID

class MetricSchema(BaseSchema):
    run_id: UUID
    
    name: str
    target_model: str
    value: float

    metadata: dict = {}