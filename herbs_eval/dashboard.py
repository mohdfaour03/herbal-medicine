"""Streamlit dashboard helpers for the herb misinformation evaluator."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import plotly.express as px

from .data import load_claims
from .metrics import aggregate_worst_of_k
from .pipeline import EvalConfig
from .simple import evaluate_claims_simple


def summarize_results(rows: List[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    for r in rows:
        judgements = r.get("judgements", [])
        _, _, worst = aggregate_worst_of_k(judgements)
        summary.append(
            {
                "id": r.get("id"),
                "claim": r.get("claim"),
                "harm_category": r.get("harm_category"),
                "harm_weight": r.get("harm_weight"),
                "worst_verdict": (worst or {}).get("verdict"),
                "worst_level": (worst or {}).get("harm_level"),
                "worst_confidence": (worst or {}).get("confidence"),
                "judgements": judgements,
            }
        )
    return summary


def run_evaluation(cfg: EvalConfig, max_claims: int) -> Dict:
    claims = load_claims()[: max(1, int(max_claims))]
    result = evaluate_claims_simple(
        claims,
        model_name=cfg.model_name,
        k=cfg.k,
        temperature=cfg.temperature,
        seed=cfg.seed,
        judge_prompt_version=cfg.judge_prompt_version,
    )
    return result


def build_plot(summary: Iterable[Dict]):
    data = list(summary)
    if not data:
        return px.bar(title="No data")
    for row in data:
        row.setdefault("worst_verdict", "safe" if row.get("worst_verdict") == "safe" else "unsafe")
        row.setdefault("worst_level", row.get("worst_level") or "none")
    fig = px.bar(
        data,
        x="id",
        y="harm_weight",
        color="worst_verdict",
        hover_data=["claim", "worst_level", "worst_confidence"],
        title="Worst-of-k verdict by claim",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def export_json(result: Dict) -> str:
    return json.dumps(result, indent=2)

