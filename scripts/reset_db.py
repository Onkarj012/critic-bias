import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.db.session import engine
from app.db.base import Base
from app.db import models  # CRITICAL: registers all tables


async def drop_and_recreate():
    print("⚠️  Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    print("✅ Creating all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database schema updated!")


if __name__ == "__main__":
    asyncio.run(drop_and_recreate())
