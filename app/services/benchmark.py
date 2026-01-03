"""
BenchmarkRunner - Coordinates full CRITIQ-BIAS experiments.

Features:
- Cache-aware (skips duplicate API calls)
- Progress logging
- Per-step error handling
"""

import yaml
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Task
from app.schemas.model import ModelInfo
from app.schemas.task import TaskSchema
from app.engines.creator import CreatorEngine
from app.engines.critic import CriticEngine
from app.llms.base import BaseLLMClient
from app.core.cache import ResponseCache, CacheStats
from app.core.exceptions import ExperimentError

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Coordinates full CRITIQ-BIAS experiments with caching support.
    """

    def __init__(
        self,
        *,
        creator_llm: BaseLLMClient,
        critic_llm: BaseLLMClient,
    ):
        self.creator_engine = CreatorEngine(creator_llm)
        self.critic_engine = CriticEngine(critic_llm)
        self.cache_stats = CacheStats()

    async def run_experiment(
        self,
        *,
        db: AsyncSession,
        config_path: str,
        use_cache: bool = True,
    ) -> Run:
        """
        Run a full experiment from a YAML config file.
        
        Args:
            db: Database session
            config_path: Path to experiment YAML config
            use_cache: If True, skip API calls for existing results
            
        Returns:
            Completed Run object
        """
        config = self._load_config(config_path)
        experiment_name = config["name"]
        
        logger.info(f"🚀 Starting experiment: {experiment_name}")
        
        # Initialize cache
        cache = ResponseCache(db) if use_cache else None

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
        
        run_id = str(run.id)
        logger.info(f"📝 Run created: {run_id}")

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
            
            task_id = str(task.id)
            logger.info(f"📋 Task created: {task.name}")

            # 3️⃣ Generate Prompts (with caching)
            prompts = []
            creators = config["models"]["creators"]
            
            logger.info(f"🎨 Generating prompts from {len(creators)} creators...")
            
            for i, creator_cfg in enumerate(creators, 1):
                creator_model = ModelInfo(**creator_cfg)
                model_id = f"{creator_model.provider}/{creator_model.model_name}"
                
                # Check cache first
                if cache:
                    cached_prompt = await cache.get_creator_response(
                        run_id=run_id,
                        task_id=task_id,
                        creator_model=creator_model.model_name,
                        creator_provider=creator_model.provider,
                    )
                    if cached_prompt:
                        self.cache_stats.record_hit()
                        prompts.append(cached_prompt)
                        logger.info(f"  [{i}/{len(creators)}] {model_id} (cached)")
                        continue
                
                # Generate new prompt
                self.cache_stats.record_miss()
                logger.info(f"  [{i}/{len(creators)}] {model_id} (calling API...)")
                
                prompt = await self.creator_engine.generate_prompt(
                    db=db,
                    run_id=run_id,
                    task=TaskSchema.model_validate(task),
                    creator_model=creator_model,
                    temperature=config["parameters"]["temperature_creator"],
                    max_tokens=config["parameters"]["max_tokens_creator"],
                    seed=config["parameters"]["seed"],
                )
                prompts.append(prompt)

            logger.info(f"✅ Generated {len(prompts)} prompts")

            # 4️⃣ Critique Prompts (with caching)
            conditions = config["conditions"]
            critics = config["models"]["critics"]
            total_critiques = len(prompts) * len(critics) * len(conditions)
            critique_count = 0
            
            logger.info(f"🔍 Generating {total_critiques} critiques...")
            
            for condition in conditions:
                source_visible = condition == "source_visible"
                condition_label = "visible" if source_visible else "blind"

                for prompt in prompts:
                    prompt_id = str(prompt.id)
                    creator_id = f"{prompt.creator_provider}/{prompt.creator_model}"
                    
                    for critic_cfg in critics:
                        critic_model = ModelInfo(**critic_cfg)
                        critic_id = f"{critic_model.provider}/{critic_model.model_name}"
                        critique_count += 1
                        
                        # Check cache first
                        if cache:
                            cached_critique = await cache.get_critique_response(
                                prompt_id=prompt_id,
                                critic_model=critic_model.model_name,
                                critic_provider=critic_model.provider,
                                source_visible=source_visible,
                            )
                            if cached_critique:
                                self.cache_stats.record_hit()
                                logger.info(
                                    f"  [{critique_count}/{total_critiques}] "
                                    f"{critic_id} → {creator_id} ({condition_label}) (cached)"
                                )
                                continue
                        
                        # Generate new critique
                        self.cache_stats.record_miss()
                        logger.info(
                            f"  [{critique_count}/{total_critiques}] "
                            f"{critic_id} → {creator_id} ({condition_label}) (calling API...)"
                        )
                        
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
            
            logger.info(f"✅ Experiment completed: {experiment_name}")
            logger.info(f"📊 {self.cache_stats}")

        except Exception as e:
            run.status = "failed"
            await db.commit()
            logger.error(f"❌ Experiment failed: {e}")
            raise ExperimentError(
                f"Experiment '{experiment_name}' failed: {e}",
                run_id=run_id,
            )

        return run

    def _load_config(self, path: str) -> dict:
        """Load experiment configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise ExperimentError(f"Config file not found: {path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ["name", "description", "task", "models", "parameters", "conditions"]
        missing = [f for f in required if f not in config]
        if missing:
            raise ExperimentError(f"Missing required config fields: {missing}")
        
        return config
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return self.cache_stats.summary()
