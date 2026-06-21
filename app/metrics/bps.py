from app.metrics.base import BaseMetric
from app.db.models import Metric
from sqlalchemy import select
from collections import defaultdict


def _resolve_condition_value(vals: dict, primary: str, fallback: str) -> float | None:
    """Resolve a condition value without treating 0.0 as missing."""
    if primary in vals:
        return vals[primary]
    if fallback in vals:
        return vals[fallback]
    return None


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
            condition = r.meta_data.get("condition") if r.meta_data else None
            if condition:
                buckets[key][condition] = r.value

        results = []
        for key, vals in buckets.items():
            visible_val = _resolve_condition_value(vals, "visible", "source_visible")
            blind_val = _resolve_condition_value(vals, "blind", "source_blind")

            if visible_val is not None and blind_val is not None:
                bps = abs(visible_val - blind_val)
                results.append(
                    {
                        "name": self.name,
                        "target_model": key,
                        "value": float(bps),
                        "metadata": {
                            "visible_mfi": visible_val,
                            "blind_mfi": blind_val,
                        },
                    }
                )

        return results
