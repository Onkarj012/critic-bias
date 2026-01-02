from app.metrics.base import BaseMetric
from app.db.models import Critique
from sqlalchemy import select


NEGATIVE_MARKERS = {
    "fails",
    "flawed",
    "weak",
    "poor",
    "unacceptable",
    "incorrect",
    "bad",
}


class BrutalityIndex(BaseMetric):
    name = "BI"

    async def compute(self, *, db, run_id: str) -> list[dict]:
        stmt = select(Critique).join(Critique.prompt).where(
            Critique.prompt.has(run_id=run_id)
        )
        critiques = (await db.execute(stmt)).scalars().all()

        results = []

        for c in critiques:
            tokens = c.raw_text.lower().split()
            if not tokens:
                continue

            neg = sum(1 for t in tokens if t in NEGATIVE_MARKERS)
            bi = neg / len(tokens)

            results.append(
                {
                    "name": self.name,
                    "target_model": f"{c.critic_provider}/{c.critic_model}",
                    "value": float(bi),
                    "metadata": {"prompt_id": str(c.prompt_id)},
                }
            )

        return results
