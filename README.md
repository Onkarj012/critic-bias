# CRITIQ-BIAS 🔬

**Cross-Model Critique Bias Benchmarking System**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

CRITIQ-BIAS is the leading benchmark for measuring **LLM-as-Judge critique bias** — whether AI models rate prompts differently when they know (or are misled about) which model created them. As every major benchmark (HLE, SWE-Bench, MT-Bench) increasingly relies on automated judges, CRITIQ-BIAS answers the meta-question: **can we trust the judges?**

> See [docs/BENCHMARK_ROADMAP.md](docs/BENCHMARK_ROADMAP.md) for the full SOTA positioning and upgrade plan.

## 🎯 Key Features

- **Factorial experimental design**: Visible, blind, and misattribution conditions isolate source bias
- **Ground-truth calibration**: GTC and SCE metrics validate critics against human ratings
- **Statistical rigor**: Bootstrap CIs, Cohen's d, ANOVA, and replications
- **SOTA-aligned datasets**: 40 curated prompts across 8 categories (coding, reasoning, agentic, safety, etc.)
- **Multi-model experiments**: Test GPT-4o, Claude, Gemini, and more
- **Leaderboard**: Aggregate and rank critic fairness across experiments
- **Caching**: Skip duplicate API calls to save costs
- **Rich exports**: JSON, Markdown reports, and visualizations
- **Dashboard**: Interactive Plotly dashboard with REST API

## 🚀 Quick Start

```bash
# 1. Setup
git clone <repo-url>
cd critic-bias
uv sync                    # Install dependencies

# 2. Configure
cp .env.example .env
# Edit .env with your DATABASE_URL and OPENROUTER_API_KEY

# 3. Initialize database
python scripts/init_db.py

# 4. Verify setup
python scripts/sanity_check.py

# 5. Run experiments (mock mode for testing)
python scripts/run_all.py --mock

# 6. Run experiments (live API)
python scripts/run_all.py
```

## 📁 Project Structure

```
critic-bias/
├── app/                       # Core application
│   ├── core/                  # Settings, cache, exceptions
│   ├── db/                    # Database models & session
│   ├── engines/               # Creator & Critic engines
│   ├── llms/                  # LLM clients (OpenRouter, Mock)
│   ├── metrics/               # Bias metrics (MFI, TPS, etc.)
│   └── services/              # Benchmark runner
├── analysis/                  # Analysis tools
│   ├── exporter.py            # JSON/Markdown/plots export
│   └── plots/                 # Visualization functions
├── dashboard/                 # Interactive dashboard
│   └── app.py                 # Plotly Dash + REST API
├── experiments/               # Experiment YAML configs
│   ├── v1_calculus.yaml
│   ├── v2_coding.yaml
│   ├── v3_creative.yaml
│   └── v4_reasoning.yaml
├── scripts/                   # CLI scripts
│   ├── run_all.py             # Main entry point
│   ├── sanity_check.py        # System verification
│   └── init_db.py             # Database setup
└── results/                   # Generated results (git-ignored)
```

## 🔧 Commands

| Command                                     | Description                       |
| ------------------------------------------- | --------------------------------- |
| `python scripts/run_all.py`                 | Run all experiments with live API |
| `python scripts/run_all.py --mock`          | Run with mock LLM (no API costs)  |
| `python scripts/run_all.py --experiment v1` | Run specific experiment           |
| `python scripts/run_all.py --no-cache`      | Disable caching                   |
| `python scripts/run_all.py --experiment v8` | Run SOTA factorial experiment |
| `python scripts/leaderboard.py`             | Generate cross-experiment leaderboard |
| `python dashboard/app.py`                   | Start dashboard at localhost:8050 |

## 📊 Metrics

| Metric | Name | Description |
| ------ | ---- | ----------- |
| **MFI** | Model Favoritism Index | Ratio of mean score for creator vs. others. MFI > 1 = favoritism |
| **BPS** | Bias Persistence Score | \|MFI_visible − MFI_blind\| — how much bias changes with visibility |
| **SBI** | Source Bias Index | Score shift under misattribution vs. blind (CALM-inspired) |
| **GTC** | Ground Truth Correlation | Pearson r between critic scores and human ground-truth ratings |
| **SCE** | Scoring Calibration Error | Mean absolute error vs. human ground truth (HLE-inspired) |
| **ANOVA** | Condition Effect Analysis | F-test: does visibility condition significantly affect scores? |
| **TPS** | Tone Polarity Score | polite (+1), neutral (0), brutal (−1) |
| **BI** | Brutality Index | Density of harsh/critical language in critiques |
| **CR** | Constructiveness Ratio | Ratio of suggestions to weaknesses cited |

## 💾 Caching

The system caches LLM responses to avoid duplicate API calls:

- **First run**: All API calls are made and cached
- **Subsequent runs**: Cached responses are reused (instant, free)
- **Cache key**: Based on model + prompt + temperature + seed

To force fresh API calls: `python scripts/run_all.py --no-cache`

## 📈 Results

After running experiments, results are saved to `results/{experiment_name}/`:

```
results/critiq-bias-v1-calculus/
├── data.json           # Full structured data
├── report.md           # Human-readable report
└── plots/
    ├── mfi_heatmap.png
    ├── score_comparison.png
    ├── tone_distribution.png
    └── critic_vs_creator.png
```

## 🌐 Dashboard

Start the interactive dashboard:

```bash
python dashboard/app.py
```

Opens at http://localhost:8050 with:

- Experiment overview with summary stats
- MFI heatmaps
- Score comparisons (visible vs blind)
- Tone analysis

### REST API (for NextJS)

The dashboard exposes these endpoints:

| Endpoint                              | Description              |
| ------------------------------------- | ------------------------ |
| `GET /api/experiments`                | List all experiments     |
| `GET /api/experiments/{name}`         | Get full experiment data |
| `GET /api/experiments/{name}/metrics` | Get metrics only         |
| `GET /api/health`                     | Health check             |

## 🔬 How It Works

```mermaid
flowchart LR
    A[Experiment YAML] --> B[Creator Models]
    B --> C[Generated Prompts]
    C --> D[Critic Models]
    D --> E{Visible vs Blind}
    E -->|Visible| F[Critique with source info]
    E -->|Blind| G[Critique without source info]
    F --> H[Compute Metrics]
    G --> H
    H --> I[Export Results]
```

1. **Creator models** generate prompts for a given task
2. **Critic models** evaluate each prompt under two conditions:
   - **Visible**: Critic knows which model created the prompt
   - **Blind**: Critic doesn't know the source
3. **Metrics** quantify bias patterns
4. **Results** exported as JSON, Markdown, and plots

## 📋 Example Experiment Config

```yaml
name: "critiq-bias-v1-calculus"
description: "Cross-model critique bias on calculus prompts"

task:
  name: "calculus_tutor"
  system_prompt: "You are an expert AI tutor."
  user_prompt: "Teach calculus derivatives to beginners."

models:
  creators:
    - provider: openai
      model_name: gpt-4o
    - provider: anthropic
      model_name: claude-3.5-sonnet

  critics:
    - provider: openai
      model_name: gpt-4o
    - provider: anthropic
      model_name: claude-3.5-sonnet

parameters:
  seed: 42
  temperature_creator: 0.7
  temperature_critic: 0.2

conditions:
  - source_visible
  - source_blind
```

## 🔐 Environment Variables

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LOG_LEVEL=INFO
```

## 📄 License

MIT
