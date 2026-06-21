import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import AsyncSessionLocal
from app.metrics.runner import MetricRunner


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_metrics.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]

    async with AsyncSessionLocal() as db:
        runner = MetricRunner(db)
        result = await runner.compute_all(run_id)
        print(f"Metrics computed: {result['summary']}")


if __name__ == "__main__":
    asyncio.run(main())
