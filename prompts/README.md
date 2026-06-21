# Prompt Datasets

This directory contains curated prompt datasets for the CRITIQ-BIAS v2.0 benchmark.

## Available Datasets

| Dataset           | Prompts | Categories                                               | Has Ground Truth |
| ----------------- | ------- | -------------------------------------------------------- | ---------------- |
| `awesome_prompts` | 20      | coding, creative, analysis, reasoning, instruction, meta | ✅ Yes           |
| `sota_prompts`    | 40      | coding, reasoning, agentic, creative, analysis, instruction, safety, meta | ✅ Yes |

## Format

Each dataset is a JSON file with the following structure:

```json
{
  "name": "dataset_name",
  "description": "Dataset description",
  "prompts": [
    {
      "id": "unique_id",
      "category": "coding",
      "content": "The actual prompt text...",
      "ground_truth_score": 8.5,
      "metadata": { "key": "value" }
    }
  ]
}
```

## Adding Custom Datasets

1. Create a new JSON file in this directory
2. Follow the format above
3. Use `PromptLoader().load_dataset("your_dataset")` to load it

## Ground Truth Scores

Prompts can optionally include a `ground_truth_score` (0-10) from human evaluation.
This enables validation that critics are accurately assessing prompt quality,
not just showing bias patterns.
