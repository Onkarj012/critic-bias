from app.schemas.base import BaseSchema


class TaskSchema(BaseSchema):
    name: str
    description: str

    system_prompt: str
    user_prompt: str

    constraints: dict = {}
