"""
MetricRunner - Orchestrates computation of all metrics for a run.
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Metric
from app.metrics.mfi import ModelFavoritismIndex
from app.metrics.tone import TonePolarityScore
from app.metrics.bps import BiasPersistenceScore
from app.metrics.brutality import BrutalityIndex
from app.metrics.constructiveness import ConstructivenessRatio
from app.core.exceptions import MetricError

logger = logging.getLogger(__name__)

# Registry of all available metrics
METRIC_CLASSES = [
    ModelFavoritismIndex,
    TonePolarityScore,
    BiasPersistenceScore,
    BrutalityIndex,
    ConstructivenessRatio,
]


class MetricRunner:
    """
    Runs all registered metrics for a given experiment run.
    
    Features:
    - Per-metric error handling (one failure doesn't stop others)
    - Saves results to database
    - Returns aggregated results
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.metrics = [cls() for cls in METRIC_CLASSES]
    
    async def compute_all(self, run_id: str) -> dict:
        """
        Compute all metrics for a run.
        
        Args:
            run_id: ID of the experiment run
            
        Returns:
            Dict with results per metric and summary stats
        """
        all_results = {}
        errors = []
        total_entries = 0
        
        logger.info(f"📊 Computing {len(self.metrics)} metrics for run {run_id}...")
        
        for metric in self.metrics:
            metric_name = metric.name
            
            try:
                logger.info(f"  Computing {metric_name}...")
                results = await metric.compute(db=self.db, run_id=run_id)
                
                # Save to database
                for result in results:
                    db_metric = Metric(
                        run_id=run_id,
                        name=result["name"],
                        target_model=result["target_model"],
                        value=result["value"],
                        meta_data=result.get("metadata", {}),
                    )
                    self.db.add(db_metric)
                
                all_results[metric_name] = results
                total_entries += len(results)
                logger.info(f"  ✓ {metric_name}: {len(results)} entries")
                
            except Exception as e:
                error_msg = f"Failed to compute {metric_name}: {e}"
                logger.error(f"  ✗ {error_msg}")
                errors.append(error_msg)
                all_results[metric_name] = {"error": str(e)}
        
        # Commit all metrics
        await self.db.commit()
        
        summary = {
            "run_id": run_id,
            "metrics_computed": len(self.metrics) - len(errors),
            "metrics_failed": len(errors),
            "total_entries": total_entries,
            "errors": errors,
        }
        
        logger.info(
            f"✅ Metrics complete: {summary['metrics_computed']}/{len(self.metrics)} succeeded, "
            f"{total_entries} entries saved"
        )
        
        return {
            "results": all_results,
            "summary": summary,
        }
    
    async def compute_single(self, run_id: str, metric_name: str) -> list[dict]:
        """
        Compute a single metric by name.
        
        Args:
            run_id: ID of the experiment run
            metric_name: Name of the metric (e.g., "MFI")
            
        Returns:
            List of metric results
        """
        metric = next((m for m in self.metrics if m.name == metric_name), None)
        if not metric:
            raise MetricError(f"Unknown metric: {metric_name}", metric_name=metric_name)
        
        try:
            results = await metric.compute(db=self.db, run_id=run_id)
            
            # Save to database
            for result in results:
                db_metric = Metric(
                    run_id=run_id,
                    name=result["name"],
                    target_model=result["target_model"],
                    value=result["value"],
                    meta_data=result.get("metadata", {}),
                )
                self.db.add(db_metric)
            
            await self.db.commit()
            return results
            
        except Exception as e:
            raise MetricError(f"Failed to compute {metric_name}: {e}", metric_name=metric_name)
    
    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return list of available metric names."""
        return [m.name for m in METRIC_CLASSES]
