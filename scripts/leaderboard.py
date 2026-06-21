#!/usr/bin/env python3
"""
leaderboard.py - Aggregate CRITIQ-BIAS metrics across all completed runs.

Produces a ranked leaderboard CSV and JSON from results/ directory,
enabling cross-experiment comparison of critic models.

Usage:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --output results/leaderboard.csv
    python scripts/leaderboard.py --metric MFI
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Metrics where lower is better
LOWER_IS_BETTER = {"SCE", "BI", "BPS", "SBI"}

# Primary ranking metrics (higher weight in composite score)
PRIMARY_METRICS = ["GTC", "MFI", "SCE", "BPS", "SBI"]


def load_all_results(results_dir: Path) -> list[dict]:
    """Load metric data from all experiment result directories."""
    entries = []

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return entries

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        data_path = exp_dir / "data.json"
        if not data_path.exists():
            continue

        try:
            with open(data_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"  Skipping {exp_dir.name}: {e}")
            continue

        run_info = data.get("run", {})
        metrics = data.get("metrics", [])

        for metric in metrics:
            entries.append({
                "experiment": exp_dir.name,
                "run_name": run_info.get("name", exp_dir.name),
                "run_status": run_info.get("status", "unknown"),
                "metric_name": metric.get("name"),
                "target_model": metric.get("target_model"),
                "value": metric.get("value"),
                "metadata": metric.get("metadata", {}),
            })

    return entries


def build_leaderboard(entries: list[dict], metric_filter: str | None = None) -> pd.DataFrame:
    """Build a ranked leaderboard DataFrame from metric entries."""
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)

    if metric_filter:
        df = df[df["metric_name"] == metric_filter]

    if df.empty:
        return df

    # Extract critic model from target_model (format: "provider/model -> creator" or just "provider/model")
    df["critic"] = df["target_model"].apply(
        lambda x: x.split(" -> ")[0] if " -> " in str(x) else str(x)
    )

    # Pivot: one row per critic per experiment, columns per metric
    pivot = df.pivot_table(
        index=["experiment", "critic"],
        columns="metric_name",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Compute composite fairness score (lower bias = better)
    # MFI deviation from 1.0, BPS, SCE (inverted), SBI (abs)
    composite_parts = []
    if "MFI" in pivot.columns:
        composite_parts.append((pivot["MFI"] - 1.0).abs())
    if "BPS" in pivot.columns:
        composite_parts.append(pivot["BPS"])
    if "SCE" in pivot.columns:
        composite_parts.append(pivot["SCE"])
    if "SBI" in pivot.columns:
        composite_parts.append(pivot["SBI"].abs())

    if composite_parts:
        pivot["fairness_score"] = sum(composite_parts) / len(composite_parts)
        pivot = pivot.sort_values("fairness_score")

    # GTC ranking (higher = better calibration)
    if "GTC" in pivot.columns:
        pivot["calibration_rank"] = pivot["GTC"].rank(ascending=False)

    pivot["rank"] = range(1, len(pivot) + 1)
    return pivot


def main():
    parser = argparse.ArgumentParser(description="CRITIQ-BIAS Leaderboard Aggregator")
    parser.add_argument("--output", "-o", help="Output CSV path (default: results/leaderboard.csv)")
    parser.add_argument("--metric", "-m", help="Filter by metric name (e.g., MFI, GTC)")
    parser.add_argument("--json", action="store_true", help="Also output JSON format")
    args = parser.parse_args()

    logger.info("CRITIQ-BIAS Leaderboard Aggregator")
    logger.info(f"  Scanning: {RESULTS_DIR}")

    entries = load_all_results(RESULTS_DIR)
    logger.info(f"  Found {len(entries)} metric entries across experiments")

    if not entries:
        logger.info("  No results found. Run experiments first: python scripts/run_all.py --mock")
        return 0

    leaderboard = build_leaderboard(entries, args.metric)

    if leaderboard.empty:
        logger.info("  No leaderboard data to display.")
        return 0

    # Display summary
    logger.info(f"\n{'='*70}")
    logger.info("LEADERBOARD (ranked by fairness_score — lower is better)")
    logger.info(f"{'='*70}")

    display_cols = ["rank", "experiment", "critic"]
    for col in ["fairness_score", "GTC", "MFI", "SCE", "BPS", "SBI", "ANOVA"]:
        if col in leaderboard.columns:
            display_cols.append(col)

    print(leaderboard[display_cols].to_string(index=False, float_format="%.3f"))

    # Save outputs
    output_csv = Path(args.output) if args.output else RESULTS_DIR / "leaderboard.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_csv, index=False)
    logger.info(f"\n  Saved CSV: {output_csv}")

    if args.json:
        output_json = output_csv.with_suffix(".json")
        leaderboard.to_json(output_json, orient="records", indent=2)
        logger.info(f"  Saved JSON: {output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
