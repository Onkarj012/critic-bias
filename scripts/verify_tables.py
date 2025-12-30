import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.db.session import engine


async def verify_tables():
    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        ))
        tables = [row[0] for row in result]
        
        print("✅ Tables created successfully:")
        for table in tables:
            print(f"  - {table}")
        
        print(f"\nTotal: {len(tables)} tables")
    
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(verify_tables())
