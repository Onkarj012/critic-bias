import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.llms.mock import MockLLMClient

async def main():
    llm = MockLLMClient()
    res = await llm.generate(
        messages=[
            {"role": "system", "content": "You are a tester."},
            {"role": "user", "content": "Hello world"},
        ],
        model="test-model",
        temperature=0.0,
        max_tokens=50,
        seed=42,
    )
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
