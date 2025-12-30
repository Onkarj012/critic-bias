from pydantic import BaseModel


class ModelInfo(BaseModel):
    provider: str            # openai, anthropic, meta, mistral
    model_name: str          # gpt-4o, claude-3.5-sonnet
    version: str | None = None

    def full_id(self) -> str:
        return f"{self.provider}/{self.model_name}"
