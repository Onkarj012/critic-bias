"""
Source Bias Index (SBI) — measures score shifts under misattribution.

Inspired by the CALM framework's perturbation-based bias quantification.
SBI isolates brand/source attribution effects by comparing scores when
a critic is told the wrong source vs. when they evaluate blindly.
"""

from app.metrics.base import BaseMetric
from app.metrics.statistical import StatisticalTests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt


class SourceBiasIndex(BaseMetric):
    """
    Source Bias Index: mean score difference under misattribution vs. blind.

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
                Critique.claimed_source,
                Prompt.id,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
        )

        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        # Group: critic -> prompt_id -> condition -> scores
        data: dict[str, dict[str, dict[str, list[float]]]] = {}
        for r in rows:
            critic = f"{r.critic_provider}/{r.critic_model}"
            prompt_id = str(r.id)
            vis = r.visibility_condition or "blind"

            if vis == "misattributed" and r.claimed_source:
                cond_key = f"misattributed_{r.claimed_source.replace('/', '_')}"
            else:
                cond_key = vis

            data.setdefault(critic, {}).setdefault(prompt_id, {}).setdefault(cond_key, []).append(
                float(r.score)
            )

        results = []
        for critic, prompts in data.items():
            # Collect blind and misattributed scores per claimed source
            blind_scores = []
            misattributed: dict[str, list[float]] = {}

            for prompt_id, conditions in prompts.items():
                if "blind" in conditions:
                    blind_scores.extend(conditions["blind"])

                for cond_key, scores in conditions.items():
                    if cond_key.startswith("misattributed_"):
                        source = cond_key.replace("misattributed_", "")
                        misattributed.setdefault(source, []).extend(scores)

            if not blind_scores:
                continue

            blind_mean = sum(blind_scores) / len(blind_scores)

            for source, mis_scores in misattributed.items():
                mis_mean = sum(mis_scores) / len(mis_scores)
                sbi = mis_mean - blind_mean

                result = {
                    "name": self.name,
                    "target_model": f"{critic} -> {source}",
                    "value": float(sbi),
                    "metadata": {
                        "claimed_source": source,
                        "blind_mean": blind_mean,
                        "misattributed_mean": mis_mean,
                        "n_blind": len(blind_scores),
                        "n_misattributed": len(mis_scores),
                        "interpretation": self._interpret(sbi),
                    },
                }

                if len(mis_scores) >= 3 and len(blind_scores) >= 3:
                    effect_size = StatisticalTests.cohens_d(mis_scores, blind_scores)
                    result["effect_size"] = effect_size
                    result["effect_interpretation"] = StatisticalTests.interpret_effect_size(effect_size)

                results.append(result)

        return results

    @staticmethod
    def _interpret(sbi: float) -> str:
        if sbi > 0.5:
            return "favors_claimed_source"
        elif sbi < -0.5:
            return "disfavors_claimed_source"
        return "no_source_effect"
