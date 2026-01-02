import yaml
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Task
from app.schemas.model import ModelInfo
from app.schemas.task import TaskSchema
from app.engines.creator import CreatorEngine
from app.engines.critic import CriticEngine
from app.llms.base import BaseLLMClient

class BenchmarkRunner:
    """
    Coordinates full CRITIQ-BIAS experiments.
    """

    def __init__(
        self,
        *,
        creator_llm: BaseLLMClient,
        critic_llm: BaseLLMClient,
    ):
        self.creator_engine = CreatorEngine(creator_llm)
        self.critic_engine = CriticEngine(critic_llm)

    async def run_experiment(
        self,
        *,
        db: AsyncSession,
        config_path: str,
    ) -> Run:

        config = self._load_config(config_path)

        # 1️⃣ Create Run
        run = Run(
            name=config["name"],
            description=config["description"],
            condition="mixed",  # multiple conditions in one run
            seed=config["parameters"]["seed"],
            temperature_creator=config["parameters"]["temperature_creator"],
            temperature_critic=config["parameters"]["temperature_critic"],
            status="running",
        )

        db.add(run)
        await db.commit()
        await db.refresh(run)

        try:
            # 2️⃣ Create Task
            task_cfg = config["task"]
            task = Task(
                name=task_cfg["name"],
                description=task_cfg["description"],
                system_prompt=task_cfg["system_prompt"],
                user_prompt=task_cfg["user_prompt"],
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)

            # 3️⃣ Generate Prompts
            prompts = []
            for creator_cfg in config["models"]["creators"]:
                creator_model = ModelInfo(**creator_cfg)

                prompt = await self.creator_engine.generate_prompt(
                    db=db,
                    run_id=str(run.id),
                    task=TaskSchema.model_validate(task),
                    creator_model=creator_model,
                    temperature=config["parameters"]["temperature_creator"],
                    max_tokens=config["parameters"]["max_tokens_creator"],
                    seed=config["parameters"]["seed"],
                )
                prompts.append(prompt)

            # 4️⃣ Critique Prompts
            for condition in config["conditions"]:
                source_visible = condition == "source_visible"

                for prompt in prompts:
                    for critic_cfg in config["models"]["critics"]:
                        critic_model = ModelInfo(**critic_cfg)

                        await self.critic_engine.critique_prompt(
                            db=db,
                            prompt=prompt,
                            critic_model=critic_model,
                            temperature=config["parameters"]["temperature_critic"],
                            max_tokens=config["parameters"]["max_tokens_critic"],
                            seed=config["parameters"]["seed"],
                            source_visible=source_visible,
                        )

            # 5️⃣ Finalize
            run.status = "completed"
            await db.commit()

        except Exception:
            run.status = "failed"
            await db.commit()
            raise

        return run

    def _load_config(self, path: str) -> dict:
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)
