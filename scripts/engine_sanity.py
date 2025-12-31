import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.engines.creator import CreatorEngine
from app.engines.critic import CriticEngine
from app.llms.mock import MockLLMClient
from app.schemas.model import ModelInfo
from app.schemas.task import TaskSchema
from app.db.models import Run

async def main():
    async with AsyncSessionLocal() as db:
        # Create a run
        run = Run(
            name="sanity-run",
            description="engine test",
            condition="source_visible",
            seed=42,
            temperature_creator=0.7,
            temperature_critic=0.2,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        # Create a task in the database
        from app.db.models import Task
        task_db = Task(
            name="test-task",
            description="sanity",
            system_prompt="You are a tutor.",
            user_prompt="Explain calculus.",
        )
        db.add(task_db)
        await db.commit()
        await db.refresh(task_db)

        # Create task schema from DB model
        task = TaskSchema(
            id=task_db.id,
            name=task_db.name,
            description=task_db.description,
            system_prompt=task_db.system_prompt,
            user_prompt=task_db.user_prompt,
            created_at=task_db.created_at,
        )

        # Generate prompt
        creator = CreatorEngine(MockLLMClient())
        prompt = await creator.generate_prompt(
            db=db,
            run_id=str(run.id),
            task=task,
            creator_model=ModelInfo(provider="mock", model_name="creator"),
            temperature=0.7,
            max_tokens=200,
            seed=42,
        )

        print(f"✅ Created prompt: {prompt.id}")

        # Critique prompt
        critic = CriticEngine(MockLLMClient())
        critique = await critic.critique_prompt(
            db=db,
            prompt=prompt,
            critic_model=ModelInfo(provider="mock", model_name="critic"),
            temperature=0.2,
            max_tokens=300,
            seed=42,
            source_visible=True,
        )

        print(f"✅ Created critique: {critique.id}")
        print(f"   Score: {critique.score}")
        print(f"   Tone: {critique.tone}")

if __name__ == "__main__":
    asyncio.run(main())
