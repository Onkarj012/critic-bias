"""
Ground Truth Correlation (GTC) and Scoring Calibration Error (SCE).

Validates that critics accurately assess prompt quality against human
ground-truth ratings — inspired by HLE confidence calibration and
scoring bias research (2025).
"""

from app.metrics.base import BaseMetric
from app.metrics.statistical import StatisticalTests
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt
import numpy as np


class GroundTruthCorrelation(BaseMetric):
    """
    Pearson correlation between critic scores and human ground-truth ratings.

    GTC near 1.0: critic aligns with human judgment
    GTC near 0.0: critic scores are uncorrelated with quality
    GTC negative: critic inversely correlates (systematic miscalibration)
    """
    name = "GTC"

    async def compute(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Critique.score,
                Critique.visibility_condition,
                Critique.source_visible,
                Prompt.ground_truth_score,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
            .where(Prompt.ground_truth_score.isnot(None))
        )

        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        # Group by critic and condition
        grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for r in rows:
            vis = r.visibility_condition or ("visible" if r.source_visible else "blind")
            critic_key = f"{r.critic_provider}/{r.critic_model}"
            grouped.setdefault(critic_key, {}).setdefault(vis, []).append(
                (float(r.score), float(r.ground_truth_score))
            )

        results = []
        for critic, conditions in grouped.items():
            for condition, pairs in conditions.items():
                if len(pairs) < 3:
                    continue

                scores = [p[0] for p in pairs]
                truths = [p[1] for p in pairs]
                pearson_r = float(np.corrcoef(scores, truths)[0, 1])

                result = {
                    "name": self.name,
                    "target_model": critic,
                    "value": pearson_r,
                    "metadata": {
                        "condition": condition,
                        "n": len(pairs),
                        "interpretation": self._interpret(pearson_r),
                    },
                }

                # Bootstrap CI on correlation
                _, ci_lower, ci_upper = StatisticalTests.bootstrap_ci(
                    [pearson_r],  # point estimate already computed
                    lambda x: x[0],
                    n_bootstrap=100,
                )
                # Re-bootstrap properly
                rng = np.random.RandomState(42)
                boot_rs = []
                scores_arr = np.array(scores)
                truths_arr = np.array(truths)
                for _ in range(500):
                    idx = rng.choice(len(pairs), size=len(pairs), replace=True)
                    if len(set(idx)) < 2:
                        continue
                    boot_rs.append(float(np.corrcoef(scores_arr[idx], truths_arr[idx])[0, 1]))

                if boot_rs:
                    result["ci_lower"] = float(np.percentile(boot_rs, 2.5))
                    result["ci_upper"] = float(np.percentile(boot_rs, 97.5))

                results.append(result)

        return results

    @staticmethod
    def _interpret(r: float) -> str:
        r_abs = abs(r)
        if r_abs >= 0.7:
            return "strong_correlation"
        elif r_abs >= 0.4:
            return "moderate_correlation"
        elif r_abs >= 0.2:
            return "weak_correlation"
        return "no_correlation"


class ScoringCalibrationError(BaseMetric):
    """
    Mean Absolute Error between critic scores and human ground truth.

    SCE = 0: perfect calibration
    SCE > 2: significant miscalibration (on 0-10 scale)
    """
    name = "SCE"

    async def compute(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Critique.score,
                Critique.visibility_condition,
                Critique.source_visible,
                Prompt.ground_truth_score,
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
            .where(Prompt.ground_truth_score.isnot(None))
        )

        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        grouped: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            vis = r.visibility_condition or ("visible" if r.source_visible else "blind")
            critic_key = f"{r.critic_provider}/{r.critic_model}"
            error = abs(float(r.score) - float(r.ground_truth_score))
            grouped.setdefault(critic_key, {}).setdefault(vis, []).append(error)

        results = []
        for critic, conditions in grouped.items():
            for condition, errors in conditions.items():
                if not errors:
                    continue

                mae = sum(errors) / len(errors)
                results.append({
                    "name": self.name,
                    "target_model": critic,
                    "value": float(mae),
                    "metadata": {
                        "condition": condition,
                        "n": len(errors),
                        "max_error": float(max(errors)),
                        "interpretation": (
                            "well_calibrated" if mae < 1.5
                            else "moderate_miscalibration" if mae < 2.5
                            else "poorly_calibrated"
                        ),
                    },
                })

        return results
