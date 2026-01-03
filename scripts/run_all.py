#!/usr/bin/env python3
"""
run_all.py - Single command to run all CRITIQ-BIAS experiments.

Features:
- Runs all experiment YAML files in experiments/ directory
- Uses caching to skip duplicate API calls (saves cost)
- Computes all metrics
- Generates exports (JSON, Markdown, plots)
- Saves everything to results/ directory

Usage:
    python scripts/run_all.py                    # Run all experiments
    python scripts/run_all.py --mock             # Use mock LLM (for testing)
    python scripts/run_all.py --experiment v1    # Run specific experiment
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import AsyncSessionLocal
from app.services.benchmark_v2 import BenchmarkRunnerV2 as BenchmarkRunner
from app.metrics.runner import MetricRunner
from app.llms.openrouter import OpenRouterClient
from app.llms.mock import MockLLMClient
from analysis.exporter import ResultExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_experiment_files(experiments_dir: Path, filter_name: str | None = None) -> list[Path]:
    """Get list of experiment YAML files."""
    files = sorted(experiments_dir.glob("*.yaml"))
    
    if filter_name:
        files = [f for f in files if filter_name in f.stem]
    
    return files


async def run_single_experiment(
    config_path: Path,
    use_mock: bool = False,
    use_cache: bool = True,
) -> dict:
    """Run a single experiment and return results."""
    
    experiment_name = config_path.stem
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 EXPERIMENT: {experiment_name}")
    logger.info(f"{'='*60}")
    
    # Select LLM client
    if use_mock:
        logger.info("Using MockLLMClient (no API calls)")
        llm_client = MockLLMClient()
    else:
        logger.info("Using OpenRouterClient (live API)")
        llm_client = OpenRouterClient()
    
    async with AsyncSessionLocal() as db:
        # Run experiment
        runner = BenchmarkRunner(
            creator_llm=llm_client,
            critic_llm=llm_client,
        )
        
        run = await runner.run_experiment(
            db=db,
            config_path=str(config_path),
            use_cache=use_cache,
        )
        
        run_id = str(run.id)
        cache_stats = runner.get_cache_stats()
        
        # Compute metrics
        logger.info("\n📊 Computing metrics...")
        metric_runner = MetricRunner(db)
        metrics_result = await metric_runner.compute_all(run_id)
        
        # Export results
        logger.info("\n📁 Exporting results...")
        exporter = ResultExporter(db)
        export_paths = await exporter.export_all(
            run_id=run_id,
            run_name=experiment_name,
        )
        
        return {
            "run_id": run_id,
            "experiment": experiment_name,
            "status": run.status,
            "cache_stats": cache_stats,
            "metrics_summary": metrics_result["summary"],
            "export_paths": export_paths,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Run CRITIQ-BIAS experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_all.py                    # Run all experiments
  python scripts/run_all.py --mock             # Test mode (no API calls)
  python scripts/run_all.py --experiment v1    # Run only v1_* experiments
  python scripts/run_all.py --no-cache         # Disable caching
        """
    )
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API costs)")
    parser.add_argument("--experiment", "-e", help="Filter experiments by name")
    parser.add_argument("--no-cache", action="store_true", help="Disable result caching")
    args = parser.parse_args()
    
    # Find experiments
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    
    experiment_files = get_experiment_files(experiments_dir, args.experiment)
    
    if not experiment_files:
        logger.error("No experiment files found!")
        sys.exit(1)
    
    logger.info(f"\n🚀 CRITIQ-BIAS Experiment Runner")
    logger.info(f"   Found {len(experiment_files)} experiment(s)")
    logger.info(f"   Mode: {'Mock' if args.mock else 'Live API'}")
    logger.info(f"   Cache: {'Disabled' if args.no_cache else 'Enabled'}")
    
    # Run experiments
    results = []
    start_time = datetime.now()
    
    for config_path in experiment_files:
        try:
            result = await run_single_experiment(
                config_path=config_path,
                use_mock=args.mock,
                use_cache=not args.no_cache,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Experiment failed: {config_path.name}")
            logger.error(f"   Error: {e}")
            results.append({
                "experiment": config_path.stem,
                "status": "failed",
                "error": str(e),
            })
    
    # Summary
    elapsed = datetime.now() - start_time
    successful = sum(1 for r in results if r.get("status") == "completed")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   Total experiments: {len(results)}")
    logger.info(f"   Successful: {successful}")
    logger.info(f"   Failed: {len(results) - successful}")
    logger.info(f"   Duration: {elapsed}")
    
    # Print result locations
    if successful > 0:
        logger.info(f"\n📂 Results saved to: {project_root / 'results'}/")
        for r in results:
            if r.get("status") == "completed":
                logger.info(f"   - {r['experiment']}/")
                if "cache_stats" in r:
                    stats = r["cache_stats"]
                    logger.info(f"     Cache: {stats}")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
