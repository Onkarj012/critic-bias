"""
Model Favoritism Index (MFI) v2.0 - Enhanced with statistical rigor.

Features:
- Bootstrap confidence intervals
- Cohen's d effect sizes
- Significance testing
- Backward compatible with v1 output format
"""

from app.metrics.base import BaseMetric
from app.metrics.statistical import StatisticalTests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt


class ModelFavoritismIndex(BaseMetric):
    """
    Measures if critics favor certain creators.
    
    MFI > 1.0: Critic favors this creator
    MFI < 1.0: Critic disfavors this creator
    MFI = 1.0: No bias
    
    v2.0 adds:
    - ci_lower, ci_upper: 95% bootstrap confidence interval
    - effect_size: Cohen's d
    - significant: True if CI excludes 1.0
    - interpretation: Human-readable bias assessment
    """
    name = "MFI"

    async def compute(
        self, 
        *, 
        db: AsyncSession, 
        run_id: str,
        include_statistics: bool = True,
    ) -> list[dict]:
        """
        Compute MFI for all critic-creator pairs.
        """
        results = []

        # Fetch all scores with condition info
        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Prompt.creator_provider,
                Prompt.creator_model,
                Critique.score,
                Critique.source_visible,
                Critique.visibility_condition,
                Critique.claimed_source,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
        )

        rows = (await db.execute(stmt)).all()
        
        if not rows:
            return results

        # Group data by condition -> critic -> creator -> scores
        # Condition key: (source_visible, visibility_condition, claimed_source)
        grouped_data = {}

        for r in rows:
            # Normalize condition info
            vis_cond = r.visibility_condition
            claimed = r.claimed_source
            
            # Legacy fallback
            if vis_cond is None:
                vis_cond = "visible" if r.source_visible else "blind"
            
            # Construct condition label for metadata
            if vis_cond == "misattributed" and claimed:
                condition_label = f"misattributed_as_{claimed.replace('/', '_')}"
            else:
                condition_label = vis_cond
            
            critic_key = f"{r.critic_provider}/{r.critic_model}"
            creator_key = f"{r.creator_provider}/{r.creator_model}"
            
            if condition_label not in grouped_data:
                grouped_data[condition_label] = {}
            if critic_key not in grouped_data[condition_label]:
                grouped_data[condition_label][critic_key] = {}
            if creator_key not in grouped_data[condition_label][critic_key]:
                grouped_data[condition_label][critic_key][creator_key] = []
            
            grouped_data[condition_label][critic_key][creator_key].append(float(r.score))

        # Compute MFI per condition
        for condition, critics in grouped_data.items():
            for critic, creators in critics.items():
                for creator, target_scores in creators.items():
                    # Gather all other creators' scores in this condition
                    other_scores = []
                    for other_creator, scores in creators.items():
                        if other_creator != creator:
                            other_scores.extend(scores)
                    
                    if not other_scores:
                        continue
                    
                    # Basic MFI calculation
                    target_mean = sum(target_scores) / len(target_scores)
                    other_mean = sum(other_scores) / len(other_scores)
                    
                    if other_mean == 0:
                        continue
                    
                    mfi = target_mean / other_mean
                    
                    result = {
                        "name": self.name,
                        "target_model": f"{critic} -> {creator}",
                        "value": float(mfi),
                        "metadata": {"condition": condition},
                    }
                    
                    # v2.0: Add statistical measures
                    if include_statistics:
                        mfi_est, ci_lower, ci_upper = StatisticalTests.bootstrap_mfi(
                            target_scores, other_scores
                        )
                        effect_size = StatisticalTests.cohens_d(target_scores, other_scores)
                        
                        result.update({
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "effect_size": effect_size,
                            "effect_interpretation": StatisticalTests.interpret_effect_size(effect_size),
                            "significant": ci_lower > 1.0 or ci_upper < 1.0,
                            "interpretation": StatisticalTests.mfi_significant(ci_lower, ci_upper),
                            "n_target": len(target_scores),
                            "n_others": len(other_scores),
                        })
                    
                    results.append(result)
        
        return results
    
    async def compute_legacy(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        """Backward compatible v1 computation (no statistics)."""
        return await self.compute(db=db, run_id=run_id, include_statistics=False)