import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import AsyncSessionLocal
from app.services.benchmark import BenchmarkRunner
from app.llms.openrouter import OpenRouterClient
from app.llms.mock import MockLLMClient


async def main():
    config_path = sys.argv[1]

    async with AsyncSessionLocal() as db:
        runner = BenchmarkRunner(
            creator_llm=MockLLMClient(),
            critic_llm=MockLLMClient(),
            # use MockLLMClient() for dry runs  
        )

        run = await runner.run_experiment(
            db=db,
            config_path=config_path,
        )

        print(f"Run completed: {run.id}")


if __name__ == "__main__":
    asyncio.run(main())
