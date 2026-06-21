"""
Source Bias Index (SBI) — measures score shifts under misattribution.

Inspired by the CALM framework's perturbation-based bias quantification.
SBI isolates brand/source attribution effects by comparing scores when
a critic is told the wrong source vs. when they evaluate blindly.
"""

from app.metrics.base import BaseMetric
from app.metrics.ground_truth import _visibility_condition
from app.metrics.statistical import StatisticalTests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt


class SourceBiasIndex(BaseMetric):
    """
    Source Bias Index: mean per-prompt score difference under misattribution vs. blind.

    SBI > 0: critic scores higher when misattributed to this source
    SBI < 0: critic scores lower when misattributed to this source
    SBI ≈ 0: no source attribution effect
    """
    name = "SBI"

    async def compute(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Critique.score,
                Critique.visibility_condition,
                Critique.source_visible,
                Critique.claimed_source,
                Prompt.id,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
        )

        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        data: dict[str, dict[str, dict[str, list[float]]]] = {}
        for r in rows:
            critic = f"{r.critic_provider}/{r.critic_model}"
            prompt_id = str(r.id)
            vis = _visibility_condition(r.visibility_condition, r.source_visible)

            if vis == "misattributed" and r.claimed_source:
                cond_key = f"misattributed_{r.claimed_source.replace('/', '_')}"
            else:
                cond_key = vis

            data.setdefault(critic, {}).setdefault(prompt_id, {}).setdefault(cond_key, []).append(
                float(r.score)
            )

        results = []
        for critic, prompts in data.items():
            deltas_by_source: dict[str, list[float]] = {}

            for conditions in prompts.values():
                if "blind" not in conditions:
                    continue

                blind_mean = sum(conditions["blind"]) / len(conditions["blind"])

                for cond_key, scores in conditions.items():
                    if not cond_key.startswith("misattributed_"):
                        continue
                    source = cond_key.replace("misattributed_", "")
                    mis_mean = sum(scores) / len(scores)
                    deltas_by_source.setdefault(source, []).append(mis_mean - blind_mean)

            for source, deltas in deltas_by_source.items():
                if not deltas:
                    continue

                sbi = sum(deltas) / len(deltas)

                # Per-prompt paired means for effect size
                mis_means: list[float] = []
                blind_means: list[float] = []
                for conditions in prompts.values():
                    if "blind" not in conditions:
                        continue
                    cond_key = f"misattributed_{source}"
                    if cond_key not in conditions:
                        continue
                    blind_means.append(sum(conditions["blind"]) / len(conditions["blind"]))
                    mis_means.append(sum(conditions[cond_key]) / len(conditions[cond_key]))

                metadata = {
                    "claimed_source": source,
                    "n_prompts": len(deltas),
                    "interpretation": self._interpret(sbi),
                }

                if len(mis_means) >= 3:
                    metadata["effect_size"] = StatisticalTests.cohens_d(mis_means, blind_means)
                    metadata["effect_interpretation"] = StatisticalTests.interpret_effect_size(
                        metadata["effect_size"]
                    )

                results.append({
                    "name": self.name,
                    "target_model": f"{critic} -> {source}",
                    "value": float(sbi),
                    "metadata": metadata,
                })

        return results

    @staticmethod
    def _interpret(sbi: float) -> str:
        if sbi > 0.5:
            return "favors_claimed_source"
        elif sbi < -0.5:
            return "disfavors_claimed_source"
        return "no_source_effect"
