from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import AsyncSession

class BaseMetric(ABC):

    name: str

    @abstractmethod
    async def compute(self, *, db: AsyncSession, run_id: str) -> list[dict]:
        raise NotImplementedError