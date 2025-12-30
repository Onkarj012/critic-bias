from app.schemas.base import BaseSchema
from typing import Literal

class RunSchema(BaseSchema):
    name: str
    description: str

    condition: Literal["source_visible", "source_hidden"]

    seed: int
    temperature_creator: float
    temperature_critic: float

    status: Literal["created", "running", "completed", "failed"] = "created"