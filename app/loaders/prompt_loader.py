"""
PromptLoader - Load real prompts from curated datasets.

Supports loading from:
- Built-in datasets (awesome_prompts, coding_prompts)
- Custom JSON files
- Inline prompts in experiment YAML
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "datasets"


@dataclass
class PromptItem:
    """A single prompt from a dataset."""
    id: str
    content: str
    category: str
    source: str  # Dataset name or "custom"
    ground_truth_score: float | None = None  # Human rating if available
    metadata: dict = field(default_factory=dict)


class PromptLoader:
    """
    Load real prompts from curated sources instead of generating them.
    
    This addresses the core issue: we should be testing critique of real
    prompts, not model-generated answers to tasks.
    """
    
    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, list[PromptItem]] = {}
    
    def list_datasets(self) -> list[str]:
        """List available prompt datasets."""
        if not self.prompts_dir.exists():
            return []
        return [f.stem for f in self.prompts_dir.glob("*.json")]
    
    def load_dataset(self, name: str) -> list[PromptItem]:
        """Load all prompts from a dataset."""
        if name in self._cache:
            return self._cache[name]
        
        path = self.prompts_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        prompts = []
        for item in data.get("prompts", []):
            prompts.append(PromptItem(
                id=item.get("id", f"{name}_{len(prompts)}"),
                content=item["content"],
                category=item.get("category", "general"),
                source=name,
                ground_truth_score=item.get("ground_truth_score"),
                metadata=item.get("metadata", {}),
            ))
        
        self._cache[name] = prompts
        return prompts
    
    def sample(
        self, 
        dataset_name: str, 
        n: int, 
        seed: int,
        categories: list[str] | None = None,
    ) -> list[PromptItem]:
        """
        Sample n prompts from a dataset.
        
        Args:
            dataset_name: Name of the dataset to sample from
            n: Number of prompts to sample
            seed: Random seed for reproducibility
            categories: Optional filter by category
        """
        prompts = self.load_dataset(dataset_name)
        
        if categories:
            prompts = [p for p in prompts if p.category in categories]
        
        if n > len(prompts):
            raise ValueError(
                f"Requested {n} prompts but dataset only has {len(prompts)}"
            )
        
        rng = random.Random(seed)
        return rng.sample(prompts, n)
    
    def load_from_file(self, path: str | Path) -> list[PromptItem]:
        """Load prompts from a custom JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        return [
            PromptItem(
                id=item.get("id", f"custom_{i}"),
                content=item["content"],
                category=item.get("category", "custom"),
                source=str(path),
                ground_truth_score=item.get("ground_truth_score"),
                metadata=item.get("metadata", {}),
            )
            for i, item in enumerate(data.get("prompts", data))
        ]
    
    def load_inline(self, prompts: list[dict]) -> list[PromptItem]:
        """Load prompts defined inline in experiment YAML."""
        return [
            PromptItem(
                id=item.get("id", f"inline_{i}"),
                content=item["content"],
                category=item.get("category", "inline"),
                source="inline",
                ground_truth_score=item.get("ground_truth_score"),
                metadata=item.get("metadata", {}),
            )
            for i, item in enumerate(prompts)
        ]
    
    def with_quality_labels(self, dataset_name: str) -> list[PromptItem]:
        """Get only prompts that have ground truth scores."""
        prompts = self.load_dataset(dataset_name)
        return [p for p in prompts if p.ground_truth_score is not None]
