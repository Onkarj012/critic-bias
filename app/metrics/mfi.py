from app.metrics.base import BaseMetric
from sqlalchemy import select, func
from app.db.models import Critique, Prompt

class ModelFavoritismIndex(BaseMetric):
    name = "MFI"

    async def compute(self, *, db, run_id: str) -> list[dict]:
        results = []

        stmt = (
            select(
                Critique.critic_provider,
                Critique.critic_model,
                Prompt.creator_provider,
                Prompt.creator_model,
                func.avg(Critique.score).label("avg_score"),
            )
            .join(Prompt, Prompt.id == Critique.prompt_id)
            .where(Prompt.run_id == run_id)
            .group_by(
                Critique.critic_provider,
                Critique.critic_model,
                Prompt.creator_provider,
                Prompt.creator_model,
            )
        )

        rows = (await db.execute(stmt)).all()

        critic_map = {}

        for r in rows:
            critic_key = f"{r.critic_provider}/{r.critic_model}"
            creator_key = f"{r.creator_provider}/{r.creator_model}"
            critic_map.setdefault(critic_key, {})[creator_key] = r.avg_score

        for critic, creators in critic_map.items():
            for creator, score in creators.items():
                others = [v for k, v in creators.items() if k != creator]
                if not others:
                    continue
                mfi = score / (sum(others) / len(others))
                results.append(
                    {
                        "name" : self.name,
                        "target_model": f"{critic} -> {creator}",
                        "value": float(mfi),
                        "metadata" : {}.
                    }
                )
        
        return results
        