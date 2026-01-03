#!/usr/bin/env python3
"""
CRITIQ-BIAS Dashboard

A simple Plotly Dash dashboard for visualizing experiment results.
Reads from results/ directory - no database connection needed.

Features:
- Experiment overview
- MFI heatmaps
- Score comparisons
- Tone analysis

Also exposes REST API endpoints for NextJS integration.

Usage:
    python dashboard/app.py
    
Then open http://localhost:8050
"""

import json
from pathlib import Path
from typing import Any

import dash
from dash import html, dcc, dash_table, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flask import Flask, jsonify

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_all_results() -> dict[str, dict]:
    """Load all experiment results from results/ directory."""
    results = {}
    
    if not RESULTS_DIR.exists():
        return results
    
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir():
            json_path = exp_dir / "data.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    results[exp_dir.name] = json.load(f)
    
    return results


def get_experiment_summary(data: dict) -> dict:
    """Extract summary stats from experiment data."""
    run = data.get("run", {})
    prompts = data.get("prompts", [])
    metrics = data.get("metrics", {})
    
    # Count critiques
    total_critiques = sum(len(p.get("critiques", [])) for p in prompts)
    
    # Unique models
    creators = set(p.get("creator", "") for p in prompts)
    critics = set()
    for p in prompts:
        for c in p.get("critiques", []):
            critics.add(c.get("critic", ""))
    
    return {
        "name": run.get("name", "Unknown"),
        "status": run.get("status", "unknown"),
        "created_at": run.get("created_at", ""),
        "num_prompts": len(prompts),
        "num_critiques": total_critiques,
        "num_creators": len(creators),
        "num_critics": len(critics),
        "metrics_available": list(metrics.keys()),
    }


# Initialize Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)


def create_layout():
    """Create the dashboard layout."""
    all_results = load_all_results()
    experiment_options = [{"label": name, "value": name} for name in all_results.keys()]
    
    return html.Div([
        # Header
        html.Div([
            html.H1("🔬 CRITIQ-BIAS Dashboard", style={"color": "#333"}),
            html.P("Cross-model critique bias analysis", style={"color": "#666"}),
        ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#f8f9fa"}),
        
        # Experiment Selector
        html.Div([
            html.Label("Select Experiment:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="experiment-dropdown",
                options=experiment_options,
                value=experiment_options[0]["value"] if experiment_options else None,
                style={"width": "400px"},
            ),
            html.Button("🔄 Refresh", id="refresh-btn", n_clicks=0,
                       style={"marginLeft": "10px", "padding": "6px 12px"}),
        ], style={"padding": "20px", "display": "flex", "alignItems": "center", "gap": "10px"}),
        
        # Summary Cards
        html.Div(id="summary-cards", style={"padding": "0 20px"}),
        
        # Tabs for different views
        dcc.Tabs([
            dcc.Tab(label="📊 Scores", children=[
                html.Div(id="scores-content", style={"padding": "20px"}),
            ]),
            dcc.Tab(label="🎯 MFI Heatmap", children=[
                html.Div(id="mfi-content", style={"padding": "20px"}),
            ]),
            dcc.Tab(label="💬 Tone Analysis", children=[
                html.Div(id="tone-content", style={"padding": "20px"}),
            ]),
            dcc.Tab(label="📝 Raw Data", children=[
                html.Div(id="raw-data-content", style={"padding": "20px"}),
            ]),
        ]),
        
        # Store for data
        dcc.Store(id="experiment-data"),
    ])


app.layout = create_layout


@callback(
    Output("experiment-data", "data"),
    [Input("experiment-dropdown", "value"),
     Input("refresh-btn", "n_clicks")]
)
def load_experiment_data(experiment_name: str, n_clicks: int):
    """Load selected experiment data."""
    if not experiment_name:
        return None
    
    all_results = load_all_results()
    return all_results.get(experiment_name)


@callback(
    Output("summary-cards", "children"),
    Input("experiment-data", "data")
)
def update_summary(data: dict):
    """Update summary cards."""
    if not data:
        return html.P("No data available. Run some experiments first!")
    
    summary = get_experiment_summary(data)
    
    card_style = {
        "display": "inline-block",
        "padding": "20px",
        "margin": "10px",
        "borderRadius": "8px",
        "backgroundColor": "#fff",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "textAlign": "center",
        "minWidth": "150px",
    }
    
    return html.Div([
        html.Div([
            html.H3(summary["num_prompts"], style={"color": "#007bff", "margin": "0"}),
            html.P("Prompts", style={"margin": "5px 0 0"}),
        ], style=card_style),
        html.Div([
            html.H3(summary["num_critiques"], style={"color": "#28a745", "margin": "0"}),
            html.P("Critiques", style={"margin": "5px 0 0"}),
        ], style=card_style),
        html.Div([
            html.H3(summary["num_creators"], style={"color": "#17a2b8", "margin": "0"}),
            html.P("Creators", style={"margin": "5px 0 0"}),
        ], style=card_style),
        html.Div([
            html.H3(summary["num_critics"], style={"color": "#dc3545", "margin": "0"}),
            html.P("Critics", style={"margin": "5px 0 0"}),
        ], style=card_style),
        html.Div([
            html.H3(len(summary["metrics_available"]), style={"color": "#6c757d", "margin": "0"}),
            html.P("Metrics", style={"margin": "5px 0 0"}),
        ], style=card_style),
    ])


@callback(
    Output("scores-content", "children"),
    Input("experiment-data", "data")
)
def update_scores(data: dict):
    """Update scores visualization."""
    if not data:
        return html.P("No data available.")
    
    prompts = data.get("prompts", [])
    if not prompts:
        return html.P("No prompts in this experiment.")
    
    # Build dataframe
    rows = []
    for prompt in prompts:
        for critique in prompt.get("critiques", []):
            rows.append({
                "creator": prompt.get("creator", ""),
                "critic": critique.get("critic", ""),
                "condition": "Visible" if critique.get("source_visible") else "Blind",
                "score": critique.get("score", 0),
                "tone": critique.get("tone", "unknown"),
            })
    
    if not rows:
        return html.P("No critiques found.")
    
    df = pd.DataFrame(rows)
    
    # Score comparison chart
    fig_box = px.box(
        df,
        x="critic",
        y="score",
        color="condition",
        title="Score Distribution: Visible vs Blind",
        labels={"critic": "Critic Model", "score": "Score (0-10)"},
    )
    
    # Average scores heatmap
    pivot = df.groupby(["critic", "creator"])["score"].mean().unstack()
    fig_heatmap = px.imshow(
        pivot,
        title="Average Score: Critic → Creator",
        labels={"x": "Creator", "y": "Critic", "color": "Score"},
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    
    return html.Div([
        dcc.Graph(figure=fig_box),
        dcc.Graph(figure=fig_heatmap),
    ])


@callback(
    Output("mfi-content", "children"),
    Input("experiment-data", "data")
)
def update_mfi(data: dict):
    """Update MFI heatmap."""
    if not data:
        return html.P("No data available.")
    
    metrics = data.get("metrics", {})
    mfi_data = metrics.get("MFI", [])
    
    if not mfi_data:
        return html.P("MFI metrics not computed yet.")
    
    # Parse MFI targets
    rows = []
    for item in mfi_data:
        target = item.get("target_model", "")
        if "->" in target:
            critic, creator = target.split("->")
            rows.append({
                "critic": critic.strip(),
                "creator": creator.strip(),
                "mfi": item.get("value", 0),
            })
    
    if not rows:
        return html.P("Could not parse MFI data.")
    
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="critic", columns="creator", values="mfi")
    
    fig = px.imshow(
        pivot,
        title="Model Favoritism Index (MFI)",
        labels={"x": "Creator Model", "y": "Critic Model", "color": "MFI"},
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=1.0,
        aspect="auto",
    )
    
    fig.add_annotation(
        text="MFI > 1: Critic favors creator | MFI < 1: Critic disfavors creator",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11),
    )
    
    return dcc.Graph(figure=fig)


@callback(
    Output("tone-content", "children"),
    Input("experiment-data", "data")
)
def update_tone(data: dict):
    """Update tone analysis."""
    if not data:
        return html.P("No data available.")
    
    prompts = data.get("prompts", [])
    
    rows = []
    for prompt in prompts:
        for critique in prompt.get("critiques", []):
            tone = critique.get("tone")
            if tone:
                rows.append({
                    "critic": critique.get("critic", ""),
                    "tone": tone,
                    "condition": "Visible" if critique.get("source_visible") else "Blind",
                })
    
    if not rows:
        return html.P("No tone data available.")
    
    df = pd.DataFrame(rows)
    
    # Tone by critic
    fig_critic = px.histogram(
        df,
        x="critic",
        color="tone",
        title="Tone Distribution by Critic",
        barmode="group",
    )
    
    # Tone by condition
    fig_condition = px.histogram(
        df,
        x="condition",
        color="tone",
        title="Tone Distribution: Visible vs Blind",
        barmode="group",
    )
    
    return html.Div([
        dcc.Graph(figure=fig_critic),
        dcc.Graph(figure=fig_condition),
    ])


@callback(
    Output("raw-data-content", "children"),
    Input("experiment-data", "data")
)
def update_raw_data(data: dict):
    """Show raw data table."""
    if not data:
        return html.P("No data available.")
    
    prompts = data.get("prompts", [])
    
    rows = []
    for prompt in prompts:
        for critique in prompt.get("critiques", []):
            rows.append({
                "Creator": prompt.get("creator", ""),
                "Critic": critique.get("critic", ""),
                "Visible": "Yes" if critique.get("source_visible") else "No",
                "Score": critique.get("score", 0),
                "Tone": critique.get("tone", ""),
                "Strengths": len(critique.get("strengths", [])),
                "Weaknesses": len(critique.get("weaknesses", [])),
            })
    
    if not rows:
        return html.P("No data to display.")
    
    df = pd.DataFrame(rows)
    
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        filter_action="native",
        sort_action="native",
        page_size=20,
    )


# === REST API Endpoints for NextJS ===

@server.route("/api/experiments")
def api_list_experiments():
    """List all experiments."""
    all_results = load_all_results()
    experiments = []
    
    for name, data in all_results.items():
        summary = get_experiment_summary(data)
        experiments.append({
            "name": name,
            "summary": summary,
        })
    
    return jsonify({"experiments": experiments})


@server.route("/api/experiments/<name>")
def api_get_experiment(name: str):
    """Get full experiment data."""
    all_results = load_all_results()
    
    if name not in all_results:
        return jsonify({"error": "Experiment not found"}), 404
    
    return jsonify(all_results[name])


@server.route("/api/experiments/<name>/metrics")
def api_get_metrics(name: str):
    """Get metrics for an experiment."""
    all_results = load_all_results()
    
    if name not in all_results:
        return jsonify({"error": "Experiment not found"}), 404
    
    data = all_results[name]
    return jsonify({"metrics": data.get("metrics", {})})


@server.route("/api/health")
def api_health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "experiments_available": len(load_all_results())})


if __name__ == "__main__":
    print("\n🔬 CRITIQ-BIAS Dashboard")
    print("=" * 40)
    print(f"📂 Results directory: {RESULTS_DIR}")
    print(f"📊 Experiments found: {len(load_all_results())}")
    print("\n🌐 Starting server...")
    print("   Dashboard:  http://localhost:8050")
    print("   API:        http://localhost:8050/api/experiments")
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(debug=True, port=8050)
