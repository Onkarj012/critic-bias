from app.db.models import Metric
from app.metrics.mfi import ModelFavoritismIndex
from app.metrics.brutality import BrutalityIndex
from app.metrics.constructiveness import ConstructivenessRatio
from app.metrics.tone import TonePolarityScore
from app.metrics.bps import BiasPersistenceScore


METRICS = [
    ModelFavoritismIndex(),
    BrutalityIndex(),
    ConstructivenessRatio(),
    TonePolarityScore(),
    BiasPersistenceScore(),
]


async def run_metrics(*, db, run_id: str):
    for metric in METRICS:
        rows = await metric.compute(db=db, run_id=run_id)
        for r in rows:
            db.add(
                Metric(
                    run_id=run_id,
                    name=r["name"],
                    target_model=r["target_model"],
                    value=r["value"],
                    meta_data=r.get("metadata", {}),
                )
            )
    await db.commit()
