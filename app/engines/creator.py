from http.client import responses
from sqlalchemy.ext.asyncio import AsyncSession

from app.llms.base import BaseLLMClient
from app.schemas.task import TaskSchema
from app.schemas.model import ModelInfo
from app.db.models import Prompt

class CreatorEngine:

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    async def generate_prompt(
        self,
        *,
        db: AsyncSession,
        run_id: str,
        task: TaskSchema,
        creator_model: ModelInfo,
        temperature: float,
        max_tokens: int,
        seed: int | None = None
    ) -> Prompt:

        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_prompt}
        ]

        response = await self.llm.generate(
            messages=messages,
            model=creator_model.full_id(),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,  
        )

        prompt = Prompt(
            run_id=run_id,
            task_id=str(task.id),
            creator_provider=creator_model.provider,
            creator_model=creator_model.model_name,
            creator_version=creator_model.version,
            content=response["content"],
            token_count=response.get("output_tokens"),
        )

        db.add(prompt)
        await db.commit()
        await db.refresh(prompt)

        return prompt