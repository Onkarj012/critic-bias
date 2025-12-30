from pydantic import BaseModel
from datetime import datetime
from uuid import UUID, uuid4

class BaseSchema(BaseModel):
    id: UUID = uuid4()
    created_at: datetime = datetime.utcnow()

    class Config:
        from_attributes = True
        