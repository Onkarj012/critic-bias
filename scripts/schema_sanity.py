from app.schemas.model import ModelInfo
from app.schemas.task import TaskSchema

model = ModelInfo(provider="openai", model_name="gpt-4o")

task = TaskSchema(
    name="calculus tutor",
    description="Tutor for calculus",
    system_prompt="You are a calculus tutor.",
    user_prompt="teach me about integrals"
)

print(model.full_id())
print(task)