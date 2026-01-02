import asyncio
from app.db.session import AsyncSessionLocal
from analysis.queries.load_metrics import load_metrics
from analysis.plots.mfi_heatmap import plot_mfi_heatmap, parse_mfi_targets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

async def main(run_id: str):
    async with AsyncSessionLocal() as db:
        df = await load_metrics(db=db, run_id=run_id, metric_name="MFI")
        df = parse_mfi_targets(df)
        plot_mfi_heatmap(df, title="CRITIQ-BIAS — Model Favoritism Index")


if __name__ == "__main__":
    import sys
    asyncio.run(main(sys.argv[1]))
