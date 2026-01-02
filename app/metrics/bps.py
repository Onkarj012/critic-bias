from app.metrics.base import BaseMetric
from app.db.models import Metric
from sqlalchemy import select
from collections import defaultdict


class BiasPersistenceScore(BaseMetric):
    name = "BPS"

    async def compute(self, *, db, run_id: str) -> list[dict]:
        stmt = select(Metric).where(
            Metric.run_id == run_id,
            Metric.name == "MFI",
        )

        rows = (await db.execute(stmt)).scalars().all()

        buckets = defaultdict(dict)
        for r in rows:
            key = r.target_model
            buckets[key][r.metadata.get("condition")] = r.value

        results = []
        for key, vals in buckets.items():
            if "source_visible" in vals and "source_blind" in vals:
                bps = abs(vals["source_visible"] - vals["source_blind"])
                results.append(
                    {
                        "name": self.name,
                        "target_model": key,
                        "value": float(bps),
                        "metadata": {},
                    }
                )

        return results
