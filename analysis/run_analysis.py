import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import AsyncSessionLocal
from analysis.queries.load_metrics import load_metrics
from analysis.plots.mfi_heatmap import plot_mfi_heatmap, parse_mfi_targets


async def main(run_id: str):
    async with AsyncSessionLocal() as db:
        df = await load_metrics(db=db, run_id=run_id, metric_name="MFI")
        df = parse_mfi_targets(df)
        plot_mfi_heatmap(df, title="CRITIQ-BIAS — Model Favoritism Index")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <run_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
