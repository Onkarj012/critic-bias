"""
ANOVA metric — tests whether visibility condition significantly affects scores.

Implements one-way ANOVA across visibility conditions (visible, blind,
misattributed) per critic, as specified in v2 factorial experiment configs.
"""

from app.metrics.base import BaseMetric
from app.metrics.statistical import StatisticalTests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt


class ConditionEffectAnalysis(BaseMetric):
    """
    One-way ANOVA testing whether visibility condition affects critic scores.

    Reports F-statistic, p-value, and eta-squared effect size per critic.
    Significant result (p < 0.05) indicates condition-dependent scoring bias.
    """
    name = "ANOVA"

    async def compute(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Critique.score,
                Critique.visibility_condition,
                Critique.source_visible,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
        )

        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        # Group scores by critic -> condition
        grouped: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            critic = f"{r.critic_provider}/{r.critic_model}"
            vis = r.visibility_condition or ("visible" if r.source_visible else "blind")
            grouped.setdefault(critic, {}).setdefault(vis, []).append(float(r.score))

        results = []
        for critic, conditions in grouped.items():
            tested_conditions = {
                name: scores for name, scores in conditions.items() if len(scores) >= 2
            }
            groups = list(tested_conditions.values())
            if len(groups) < 2:
                continue

            anova_result = StatisticalTests.one_way_anova(groups)

            results.append({
                "name": self.name,
                "target_model": critic,
                "value": float(anova_result.statistic),
                "metadata": {
                    "p_value": anova_result.p_value,
                    "effect_size": anova_result.effect_size,
                    "significant": anova_result.significant,
                    "method": anova_result.method,
                    "conditions": list(tested_conditions.keys()),
                    "group_sizes": {k: len(v) for k, v in tested_conditions.items()},
                    "interpretation": (
                        "condition_significant" if anova_result.significant
                        else "no_condition_effect"
                    ),
                },
            })

        return results
