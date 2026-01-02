import asyncio
import sys
from app.db.session import AsyncSessionLocal
from app.metrics.runner import run_metrics


async def main():
    run_id = sys.argv[1]

    async with AsyncSessionLocal() as db:
        await run_metrics(db=db, run_id=run_id)
        print("Metrics computed.")

if __name__ == "__main__":
    asyncio.run(main())
