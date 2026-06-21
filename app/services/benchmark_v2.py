"""
BenchmarkRunner V2.0 - Enhanced with factorial design and prompt datasets.

New features:
- Load prompts from curated datasets (PromptLoader)
- Factorial experimental design with full condition crossing
- Misattribution condition for brand bias isolation
- Multiple replications for statistical power
- Backward compatible with v1 experiment configs
"""

import yaml
import logging
from itertools import product
from pathlib import Path
from typing import Literal
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Task, Prompt
from app.schemas.model import ModelInfo
from app.schemas.task import TaskSchema
from app.engines.creator import CreatorEngine
from app.engines.critic import CriticEngine, VisibilityCondition
from app.llms.base import BaseLLMClient
from app.loaders.prompt_loader import PromptLoader, PromptItem
from app.core.cache import ResponseCache, CacheStats
from app.core.exceptions import ExperimentError

logger = logging.getLogger(__name__)


class BenchmarkRunnerV2:
    """
    V2.0 Benchmark Runner with factorial design support.
    
    Supports:
    - v1 configs: Task-based prompt generation (backward compatible)
    - v2 configs: Dataset-based prompts with factorial design
    """

    def __init__(
        self,
        *,
        creator_llm: BaseLLMClient,
        critic_llm: BaseLLMClient,
        prompt_loader: PromptLoader | None = None,
    ):
        self.creator_engine = CreatorEngine(creator_llm)
        self.critic_engine = CriticEngine(critic_llm, use_v2_prompt=True)
        self.prompt_loader = prompt_loader or PromptLoader()
        self.cache_stats = CacheStats()

    async def run_experiment(
        self,
        *,
        db: AsyncSession,
        config_path: str,
        use_cache: bool = True,
    ) -> Run:
        """
        Run experiment, auto-detecting v1 or v2 config format.
        """
        config = self._load_config(config_path)
        
        version = config.get("version", "1.0")
        if version.startswith("2"):
            return await self._run_v2_experiment(db, config, use_cache)
        else:
            return await self._run_v1_experiment(db, config_path, use_cache)

    async def _run_v2_experiment(
        self,
        db: AsyncSession,
        config: dict,
        use_cache: bool,
    ) -> Run:
        """
        V2.0 factorial experiment with dataset prompts.
        """
        experiment_name = config["name"]
        logger.info(f"🚀 Starting V2.0 factorial experiment: {experiment_name}")
        
        cache = ResponseCache(db) if use_cache else None
        
        # Load prompts from dataset
        prompts_cfg = config["prompts"]
        prompt_items = self._load_prompts(prompts_cfg, config["statistics"]["seeds"][0])
        logger.info(f"📚 Loaded {len(prompt_items)} prompts from {prompts_cfg.get('dataset_name', 'custom')}")
        
        # Generate factorial conditions
        design = config["design"]
        conditions = self._generate_factorial_conditions(design)
        logger.info(f"🔬 Generated {len(conditions)} experimental conditions")
        
        # Create Run
        stats_cfg = config["statistics"]
        run = Run(
            name=config["name"],
            description=config["description"],
            condition="factorial",
            seed=stats_cfg["seeds"][0],
            temperature_creator=0.0,  # Not used in v2
            temperature_critic=config["parameters"]["temperature_critic"],
            status="running",
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        run_id = str(run.id)
        logger.info(f"📝 Run created: {run_id}")
        
        try:
            # Create Prompt records from dataset items
            db_prompts = []
            prompt_source = prompts_cfg.get("source", "dataset")
            dataset_name = prompts_cfg.get("dataset_name", "custom")
            for item in prompt_items:
                prompt = Prompt(
                    run_id=run_id,
                    task_id=None,
                    creator_provider="dataset",
                    creator_model=item.source,
                    content=item.content,
                    token_count=len(item.content.split()),
                    source_type=prompt_source,
                    dataset_name=dataset_name,
                    prompt_category=item.category,
                    ground_truth_score=item.ground_truth_score,
                )
                db.add(prompt)
                db_prompts.append((prompt, item))
            
            await db.commit()
            for p, _ in db_prompts:
                await db.refresh(p)
            
            # Run all conditions across all replications
            critics = [ModelInfo(**c) for c in config["models"]["critics"]]
            replications = stats_cfg["replications"]
            seeds = stats_cfg["seeds"]
            
            total_evals = len(db_prompts) * len(conditions) * len(critics) * replications
            eval_count = 0
            
            logger.info(f"🔍 Running {total_evals} evaluations ({replications} replications)...")
            
            for rep_idx in range(replications):
                seed = seeds[rep_idx % len(seeds)]
                logger.info(f"  📊 Replication {rep_idx + 1}/{replications} (seed={seed})")
                
                for prompt, item in db_prompts:
                    for condition in conditions:
                        visibility = condition["source_visibility"]
                        claimed_source = condition.get("claimed_source")
                        
                        for critic_model in critics:
                            eval_count += 1
                            critic_id = f"{critic_model.provider}/{critic_model.model_name}"
                            
                            # Skip misattributed if claimed_source not applicable
                            if visibility != "misattributed" and claimed_source:
                                continue
                            if visibility == "misattributed" and not claimed_source:
                                continue
                            
                            # Log progress
                            if eval_count % 10 == 0 or eval_count == total_evals:
                                logger.info(f"    [{eval_count}/{total_evals}] {critic_id} ({visibility})")
                            
                            # Run critique
                            await self.critic_engine.critique_prompt_v2(
                                db=db,
                                prompt=prompt,
                                critic_model=critic_model,
                                temperature=config["parameters"]["temperature_critic"],
                                max_tokens=config["parameters"]["max_tokens_critic"],
                                seed=seed,
                                visibility=visibility,
                                actual_source=item.source,
                                claimed_source=claimed_source,
                                replication_id=rep_idx,
                            )
            
            run.status = "completed"
            await db.commit()
            logger.info(f"✅ V2.0 experiment completed: {experiment_name}")
            
        except Exception as e:
            run.status = "failed"
            await db.commit()
            logger.error(f"❌ Experiment failed: {e}")
            raise ExperimentError(f"Experiment '{experiment_name}' failed: {e}", run_id=run_id)
        
        return run

    async def _run_v1_experiment(
        self,
        db: AsyncSession,
        config_path: str,
        use_cache: bool,
    ) -> Run:
        """
        Legacy v1 experiment (task-based prompt generation).
        Delegates to original BenchmarkRunner logic.
        """
        from app.services.benchmark import BenchmarkRunner
        
        # Create a legacy runner and delegate
        legacy_runner = BenchmarkRunner(
            creator_llm=self.creator_engine.llm,
            critic_llm=self.critic_engine.llm,
        )
        
        return await legacy_runner.run_experiment(
            db=db,
            config_path=config_path,
            use_cache=use_cache,
        )

    def _load_prompts(
        self,
        config: dict,
        seed: int,
    ) -> list[PromptItem]:
        """Load prompts based on config."""
        source = config.get("source", "dataset")
        
        if source == "dataset":
            dataset_name = config["dataset_name"]
            sample_size = config.get("sample_size", 10)
            categories = config.get("categories")
            
            return self.prompt_loader.sample(
                dataset_name=dataset_name,
                n=sample_size,
                seed=seed,
                categories=categories,
            )
        
        elif source == "file":
            return self.prompt_loader.load_from_file(config["file_path"])
        
        elif source == "inline":
            return self.prompt_loader.load_inline(config["prompts"])
        
        else:
            raise ExperimentError(f"Unknown prompt source: {source}")

    def _generate_factorial_conditions(self, design: dict) -> list[dict]:
        """
        Generate all conditions from factorial design.
        
        For factors like:
          source_visibility: [visible, blind, misattributed]
          claimed_source: [openai, anthropic] (applies_when: misattributed)
        
        Generates:
          - visible (no claimed_source)
          - blind (no claimed_source)
          - misattributed + openai
          - misattributed + anthropic
        """
        factors = design.get("factors", {})
        
        if not factors:
            return [{"source_visibility": "blind"}]
        
        # Handle special case: claimed_source only applies to misattributed
        visibility_levels = factors.get("source_visibility", {}).get("levels", ["blind"])
        claimed_sources = factors.get("claimed_source", {}).get("levels", [])
        
        conditions = []
        
        for vis in visibility_levels:
            if vis == "misattributed" and claimed_sources:
                # Cross with claimed_source
                for claimed in claimed_sources:
                    conditions.append({
                        "source_visibility": vis,
                        "claimed_source": claimed,
                    })
            else:
                # No claimed_source needed
                conditions.append({
                    "source_visibility": vis,
                    "claimed_source": None,
                })
        
        return conditions

    def _load_config(self, path: str) -> dict:
        """Load experiment configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise ExperimentError(f"Config file not found: {path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return self.cache_stats.summary()
