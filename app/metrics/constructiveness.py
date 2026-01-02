from app.metrics.base import BaseMetric
from app.db.models import Critique
from sqlalchemy import select


class ConstructivenessRatio(BaseMetric):
    name = "CR"

    async def compute(self, *, db, run_id: str) -> list[dict]:
        stmt = select(Critique).join(Critique.prompt).where(
            Critique.prompt.has(run_id=run_id)
        )
        critiques = (await db.execute(stmt)).scalars().all()

        results = []

        for c in critiques:
            total = len(c.weaknesses) + len(c.suggestions)
            if total == 0:
                continue

            cr = len(c.suggestions) / total

            results.append(
                {
                    "name": self.name,
                    "target_model": f"{c.critic_provider}/{c.critic_model}",
                    "value": float(cr),
                    "metadata": {"prompt_id": str(c.prompt_id)},
                }
            )

        return results
