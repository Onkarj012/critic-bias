"""
Ground Truth Correlation (GTC) and Scoring Calibration Error (SCE).

Validates that critics accurately assess prompt quality against human
ground-truth ratings — inspired by HLE confidence calibration and
scoring bias research (2025).
"""

from app.metrics.base import BaseMetric
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Critique, Prompt
import numpy as np


def _visibility_condition(visibility_condition: str | None, source_visible: bool) -> str:
    if visibility_condition:
        return visibility_condition
    return "visible" if source_visible else "blind"


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

        grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for r in rows:
            vis = _visibility_condition(r.visibility_condition, r.source_visible)
            critic_key = f"{r.critic_provider}/{r.critic_model}"
            grouped.setdefault(critic_key, {}).setdefault(vis, []).append(
                (float(r.score), float(r.ground_truth_score))
            )

        results = []
        for critic, conditions in grouped.items():
            for condition, pairs in conditions.items():
                if len(pairs) < 3:
                    continue

                scores = np.array([p[0] for p in pairs])
                truths = np.array([p[1] for p in pairs])
                pearson_r = float(np.corrcoef(scores, truths)[0, 1])

                metadata = {
                    "condition": condition,
                    "n": len(pairs),
                    "interpretation": self._interpret(pearson_r),
                }

                boot_rs = self._bootstrap_correlation(scores, truths, seed=hash((critic, condition)))
                if boot_rs:
                    metadata["ci_lower"] = float(np.percentile(boot_rs, 2.5))
                    metadata["ci_upper"] = float(np.percentile(boot_rs, 97.5))

                results.append({
                    "name": self.name,
                    "target_model": critic,
                    "value": pearson_r,
                    "metadata": metadata,
                })

        return results

    @staticmethod
    def _bootstrap_correlation(
        scores: np.ndarray,
        truths: np.ndarray,
        *,
        n_bootstrap: int = 500,
        seed: int,
    ) -> list[float]:
        rng = np.random.RandomState(seed % (2**31))
        boot_rs = []
        n = len(scores)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            if len(set(idx)) < 2:
                continue
            boot_rs.append(float(np.corrcoef(scores[idx], truths[idx])[0, 1]))

        return boot_rs

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
            vis = _visibility_condition(r.visibility_condition, r.source_visible)
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
