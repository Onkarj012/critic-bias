#!/usr/bin/env python3
"""
sanity_check.py - Single file to verify the entire system works.

Run this once after setup to ensure everything is configured correctly.
If all checks pass, you're ready to run experiments.

Usage:
    python scripts/sanity_check.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def print_result(name: str, passed: bool, message: str = ""):
    icon = "✅" if passed else "❌"
    msg = f" - {message}" if message else ""
    print(f"  {icon} {name}{msg}")
    return passed


async def check_database():
    """Test database connection and tables."""
    try:
        from app.db.session import AsyncSessionLocal
        from app.db.models import Run, Task, Prompt, Critique, Metric
        from sqlalchemy import text
        
        async with AsyncSessionLocal() as db:
            # Test connection
            result = await db.execute(text("SELECT 1"))
            result.scalar()
            
            # Test tables exist
            tables = ["runs", "tasks", "prompts", "critiques", "metrics"]
            for table in tables:
                await db.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
            
            return True, "Connected and all tables exist"
    except Exception as e:
        return False, str(e)


async def check_settings():
    """Test settings and environment variables."""
    try:
        from app.core.settings import settings
        
        errors = []
        if not settings.DATABASE_URL:
            errors.append("DATABASE_URL not set")
        if not settings.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY not set")
        
        if errors:
            return False, ", ".join(errors)
        
        return True, "All required settings configured"
    except Exception as e:
        return False, str(e)


async def check_mock_llm():
    """Test mock LLM client."""
    try:
        from app.llms.mock import MockLLMClient
        
        client = MockLLMClient()
        response = await client.generate(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            temperature=0.7,
            max_tokens=100,
        )
        
        if "content" in response:
            return True, "Mock LLM working"
        return False, "Invalid response format"
    except Exception as e:
        return False, str(e)


async def check_metrics():
    """Test metric classes instantiate correctly."""
    try:
        from app.metrics.runner import MetricRunner
        
        metrics = MetricRunner.available_metrics()
        if len(metrics) >= 5:
            return True, f"Found {len(metrics)} metrics: {', '.join(metrics)}"
        return False, f"Expected 5+ metrics, found {len(metrics)}"
    except Exception as e:
        return False, str(e)


async def check_experiments():
    """Check experiment YAML files exist."""
    try:
        from pathlib import Path
        
        experiments_dir = Path(__file__).parent.parent / "experiments"
        yaml_files = list(experiments_dir.glob("*.yaml"))
        
        if len(yaml_files) >= 1:
            names = [f.stem for f in yaml_files]
            return True, f"Found {len(yaml_files)} experiments: {', '.join(names)}"
        return False, "No experiment YAML files found"
    except Exception as e:
        return False, str(e)


async def check_cache():
    """Test cache functionality."""
    try:
        from app.core.cache import generate_cache_key, CacheStats
        
        # Test key generation
        key = generate_cache_key(
            model_id="test",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
            seed=42,
        )
        
        if len(key) == 64:  # SHA256 hex length
            return True, "Cache key generation working"
        return False, "Invalid cache key format"
    except Exception as e:
        return False, str(e)


async def check_results_dir():
    """Ensure results directory can be created."""
    try:
        from pathlib import Path
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Test write access
        test_file = results_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        return True, "Results directory writable"
    except Exception as e:
        return False, str(e)


async def main():
    print("\n🔍 CRITIQ-BIAS Sanity Check")
    print("=" * 50)
    
    checks = [
        ("Settings", check_settings),
        ("Database", check_database),
        ("Mock LLM", check_mock_llm),
        ("Metrics", check_metrics),
        ("Experiments", check_experiments),
        ("Cache", check_cache),
        ("Results Dir", check_results_dir),
    ]
    
    results = []
    
    for name, check_fn in checks:
        try:
            passed, message = await check_fn()
        except Exception as e:
            passed, message = False, f"Unexpected error: {e}"
        
        results.append(print_result(name, passed, message))
    
    print("=" * 50)
    
    passed_count = sum(results)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"\n✅ All {total_count} checks passed!")
        print("   You're ready to run experiments.")
        print("\n   Next steps:")
        print("   1. Run with mock: python scripts/run_all.py --mock")
        print("   2. Run live:      python scripts/run_all.py")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} of {total_count} checks failed!")
        print("   Please fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
