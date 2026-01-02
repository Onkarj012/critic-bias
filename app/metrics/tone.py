from app.metrics.base import BaseMetric
from app.db.models import Critique
from sqlalchemy import select


TONE_MAP = {
    "polite": 1.0,
    "neutral": 0.0,
    "brutal": -1.0,
}


class TonePolarityScore(BaseMetric):
    name = "TPS"

    async def compute(self, *, db, run_id: str) -> list[dict]:
        stmt = select(Critique).join(Critique.prompt).where(
            Critique.prompt.has(run_id=run_id)
        )
        critiques = (await db.execute(stmt)).scalars().all()

        results = []

        for c in critiques:
            if c.tone not in TONE_MAP:
                continue

            results.append(
                {
                    "name": self.name,
                    "target_model": f"{c.critic_provider}/{c.critic_model}",
                    "value": TONE_MAP[c.tone],
                    "metadata": {"prompt_id": str(c.prompt_id)},
                }
            )

        return results
