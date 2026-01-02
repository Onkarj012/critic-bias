import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Metric
from sqlalchemy import select

async def load_metrics(
    *,
    db: AsyncSession,
    run_id: str,
    metric_name: str | None = None,
) -> pd.DataFrame:
    
    stmt = select(Metric).where(Metric.run_id == run_id)

    if metric_name:
        stmt = stmt.where(Metric.name == metric_name)

    rows = (await db.execute(stmt)).scalars().all()
    
    return pd.DataFrame(
        [
            {
                "name": r.name,
                "target_model": r.target_model,
                "value": r.value,
                "metadata": r.meta_data,
            }
            for r in rows
        ]
    )