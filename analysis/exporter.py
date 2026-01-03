"""
ResultExporter - Export experiment results to JSON, Markdown, and plots.

All results are saved to results/{run_name}/ directory.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import Run, Task, Prompt, Critique, Metric

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


class ResultExporter:
    """
    Exports experiment results to various formats.
    
    Output structure:
        results/{run_name}/
        ├── data.json           # Full structured data
        ├── report.md           # Human-readable prompts & responses
        └── plots/
            ├── mfi_heatmap.png
            ├── tone_distribution.png
            ├── score_comparison.png
            └── visible_vs_blind.png
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def export_all(
        self,
        run_id: str,
        run_name: str,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        """
        Export all formats for a run.
        
        Returns dict of format -> file path
        """
        # Create output directory
        if output_dir is None:
            output_dir = RESULTS_DIR / run_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        # Load all data
        data = await self._load_run_data(run_id)
        
        # Export JSON
        json_path = output_dir / "data.json"
        self._export_json(data, json_path)
        paths["json"] = json_path
        logger.info(f"  ✓ JSON: {json_path}")
        
        # Export Markdown
        md_path = output_dir / "report.md"
        self._export_markdown(data, md_path)
        paths["markdown"] = md_path
        logger.info(f"  ✓ Markdown: {md_path}")
        
        # Export Plots
        plot_paths = self._export_plots(data, plots_dir)
        paths["plots"] = plot_paths
        for name, path in plot_paths.items():
            logger.info(f"  ✓ Plot: {path}")
        
        return paths
    
    async def _load_run_data(self, run_id: str) -> dict[str, Any]:
        """Load all data for a run from database."""
        
        # Load run
        run = await self.db.get(Run, run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        
        # Load prompts with critiques (eager load to avoid lazy loading issues)
        from sqlalchemy.orm import selectinload
        stmt = select(Prompt).where(Prompt.run_id == run_id).options(selectinload(Prompt.critiques))
        prompts = (await self.db.execute(stmt)).scalars().all()
        
        # Load task
        task = None
        if prompts:
            task = await self.db.get(Task, prompts[0].task_id)
        
        # Load metrics
        stmt = select(Metric).where(Metric.run_id == run_id)
        metrics = (await self.db.execute(stmt)).scalars().all()
        
        # Structure data
        data = {
            "run": {
                "id": str(run.id),
                "name": run.name,
                "description": run.description,
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "parameters": {
                    "seed": run.seed,
                    "temperature_creator": run.temperature_creator,
                    "temperature_critic": run.temperature_critic,
                },
            },
            "task": {
                "name": task.name if task else None,
                "description": task.description if task else None,
                "system_prompt": task.system_prompt if task else None,
                "user_prompt": task.user_prompt if task else None,
            } if task else None,
            "prompts": [],
            "metrics": {},
        }
        
        # Add prompts and critiques
        for prompt in prompts:
            prompt_data = {
                "id": str(prompt.id),
                "creator": f"{prompt.creator_provider}/{prompt.creator_model}",
                "content": prompt.content,
                "token_count": prompt.token_count,
                "critiques": [],
            }
            
            for critique in prompt.critiques:
                critique_data = {
                    "id": str(critique.id),
                    "critic": f"{critique.critic_provider}/{critique.critic_model}",
                    "source_visible": critique.source_visible,
                    "score": critique.score,
                    "tone": critique.tone,
                    "strengths": critique.strengths,
                    "weaknesses": critique.weaknesses,
                    "suggestions": critique.suggestions,
                    "raw_text": critique.raw_text,
                }
                prompt_data["critiques"].append(critique_data)
            
            data["prompts"].append(prompt_data)
        
        # Add metrics grouped by type
        for metric in metrics:
            if metric.name not in data["metrics"]:
                data["metrics"][metric.name] = []
            
            data["metrics"][metric.name].append({
                "target_model": metric.target_model,
                "value": metric.value,
                "metadata": metric.meta_data,
            })
        
        return data
    
    def _export_json(self, data: dict, path: Path) -> None:
        """Export data to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_markdown(self, data: dict, path: Path) -> None:
        """Export data to Markdown report."""
        run = data["run"]
        task = data["task"]
        prompts = data["prompts"]
        metrics = data["metrics"]
        
        lines = []
        
        # Header
        lines.append(f"# {run['name']}")
        lines.append("")
        lines.append(f"> {run['description']}")
        lines.append("")
        lines.append(f"**Run ID:** `{run['id']}`")
        lines.append(f"**Status:** {run['status']}")
        lines.append(f"**Created:** {run['created_at']}")
        lines.append("")
        
        # Task
        if task:
            lines.append("## Task")
            lines.append("")
            lines.append(f"**Name:** {task['name']}")
            lines.append("")
            lines.append("**System Prompt:**")
            lines.append("```")
            lines.append(task['system_prompt'])
            lines.append("```")
            lines.append("")
            lines.append("**User Prompt:**")
            lines.append("```")
            lines.append(task['user_prompt'])
            lines.append("```")
            lines.append("")
        
        # Parameters
        lines.append("## Parameters")
        lines.append("")
        params = run['parameters']
        lines.append(f"- **Seed:** {params['seed']}")
        lines.append(f"- **Creator Temperature:** {params['temperature_creator']}")
        lines.append(f"- **Critic Temperature:** {params['temperature_critic']}")
        lines.append("")
        
        # Prompts and Critiques
        lines.append("## Prompts & Critiques")
        lines.append("")
        
        for i, prompt in enumerate(prompts, 1):
            lines.append(f"### Prompt {i}: {prompt['creator']}")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>View Content</summary>")
            lines.append("")
            lines.append("```")
            lines.append(prompt['content'][:2000])  # Truncate long content
            if len(prompt['content']) > 2000:
                lines.append("... [truncated]")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
            
            # Critiques
            for critique in prompt['critiques']:
                condition = "Visible" if critique['source_visible'] else "Blind"
                lines.append(f"#### Critique by {critique['critic']} ({condition})")
                lines.append("")
                lines.append(f"**Score:** {critique['score']}/10 | **Tone:** {critique['tone']}")
                lines.append("")
                
                if critique['strengths']:
                    lines.append("**Strengths:**")
                    for s in critique['strengths'][:3]:  # Limit to 3
                        lines.append(f"- {s}")
                    lines.append("")
                
                if critique['weaknesses']:
                    lines.append("**Weaknesses:**")
                    for w in critique['weaknesses'][:3]:
                        lines.append(f"- {w}")
                    lines.append("")
                
                if critique['suggestions']:
                    lines.append("**Suggestions:**")
                    for s in critique['suggestions'][:3]:
                        lines.append(f"- {s}")
                    lines.append("")
        
        # Metrics Summary
        if metrics:
            lines.append("## Metrics Summary")
            lines.append("")
            
            for metric_name, values in metrics.items():
                lines.append(f"### {metric_name}")
                lines.append("")
                lines.append("| Target Model | Value |")
                lines.append("|--------------|-------|")
                for v in values[:10]:  # Limit to 10
                    lines.append(f"| {v['target_model']} | {v['value']:.3f} |")
                lines.append("")
        
        # Write file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _export_plots(self, data: dict, plots_dir: Path) -> dict[str, Path]:
        """Generate and save all plots."""
        paths = {}
        metrics = data["metrics"]
        prompts = data["prompts"]
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. MFI Heatmap
        if "MFI" in metrics and metrics["MFI"]:
            path = plots_dir / "mfi_heatmap.png"
            self._plot_mfi_heatmap(metrics["MFI"], path)
            paths["mfi_heatmap"] = path
        
        # 2. Score Comparison (Visible vs Blind)
        if prompts:
            path = plots_dir / "score_comparison.png"
            self._plot_score_comparison(prompts, path)
            paths["score_comparison"] = path
        
        # 3. Tone Distribution
        if prompts:
            path = plots_dir / "tone_distribution.png"
            self._plot_tone_distribution(prompts, path)
            paths["tone_distribution"] = path
        
        # 4. Critic Scores by Creator
        if prompts:
            path = plots_dir / "critic_vs_creator.png"
            self._plot_critic_vs_creator(prompts, path)
            paths["critic_vs_creator"] = path
        
        return paths
    
    def _plot_mfi_heatmap(self, mfi_data: list, path: Path) -> None:
        """Plot MFI heatmap."""
        # Parse target_model into critic and creator
        rows = []
        for item in mfi_data:
            if "->" in item["target_model"]:
                critic, creator = item["target_model"].split("->")
                rows.append({
                    "critic": critic.strip(),
                    "creator": creator.strip(),
                    "value": item["value"],
                })
        
        if not rows:
            return
        
        df = pd.DataFrame(rows)
        # Use pivot_table with mean aggregation to handle duplicates (e.g. visible vs blind)
        pivot = df.pivot_table(index="critic", columns="creator", values="value", aggfunc="mean")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.0,
            cbar_kws={"label": "Favoritism Index (Avg)"},
        )
        plt.title("Model Favoritism Index (MFI)\n>1 = Favors creator, <1 = Disfavors creator")
        plt.xlabel("Creator Model")
        plt.ylabel("Critic Model")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def _plot_score_comparison(self, prompts: list, path: Path) -> None:
        """Plot score comparison: visible vs blind."""
        rows = []
        for prompt in prompts:
            for critique in prompt["critiques"]:
                rows.append({
                    "creator": prompt["creator"],
                    "critic": critique["critic"],
                    "condition": "Visible" if critique["source_visible"] else "Blind",
                    "score": critique["score"],
                })
        
        if not rows:
            return
        
        df = pd.DataFrame(rows)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="critic", y="score", hue="condition", palette="Set2")
        plt.title("Critic Scores: Visible vs Blind Condition")
        plt.xlabel("Critic Model")
        plt.ylabel("Score (0-10)")
        plt.legend(title="Condition")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def _plot_tone_distribution(self, prompts: list, path: Path) -> None:
        """Plot tone distribution by critic."""
        rows = []
        for prompt in prompts:
            for critique in prompt["critiques"]:
                if critique["tone"]:
                    rows.append({
                        "critic": critique["critic"],
                        "tone": critique["tone"],
                        "condition": "Visible" if critique["source_visible"] else "Blind",
                    })
        
        if not rows:
            return
        
        df = pd.DataFrame(rows)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # By critic
        tone_by_critic = df.groupby(["critic", "tone"]).size().unstack(fill_value=0)
        tone_by_critic.plot(kind="bar", ax=axes[0], colormap="Set2")
        axes[0].set_title("Tone by Critic Model")
        axes[0].set_xlabel("Critic")
        axes[0].set_ylabel("Count")
        axes[0].legend(title="Tone")
        axes[0].tick_params(axis="x", rotation=45)
        
        # By condition
        tone_by_condition = df.groupby(["condition", "tone"]).size().unstack(fill_value=0)
        tone_by_condition.plot(kind="bar", ax=axes[1], colormap="Set2")
        axes[1].set_title("Tone by Condition")
        axes[1].set_xlabel("Condition")
        axes[1].set_ylabel("Count")
        axes[1].legend(title="Tone")
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def _plot_critic_vs_creator(self, prompts: list, path: Path) -> None:
        """Plot average critic scores by creator (heatmap style)."""
        rows = []
        for prompt in prompts:
            for critique in prompt["critiques"]:
                rows.append({
                    "creator": prompt["creator"],
                    "critic": critique["critic"],
                    "score": critique["score"],
                })
        
        if not rows:
            return
        
        df = pd.DataFrame(rows)
        pivot = df.groupby(["critic", "creator"])["score"].mean().unstack()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Average Score"},
        )
        plt.title("Average Score by Critic → Creator")
        plt.xlabel("Creator Model")
        plt.ylabel("Critic Model")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


def parse_mfi_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Parse MFI target strings into critic/creator columns."""
    critics, creators = [], []
    for t in df["target_model"]:
        if "->" in t:
            critic, creator = t.split("->")
            critics.append(critic.strip())
            creators.append(creator.strip())
        else:
            critics.append(t)
            creators.append("")
    
    df["critic"] = critics
    df["creator"] = creators
    return df
