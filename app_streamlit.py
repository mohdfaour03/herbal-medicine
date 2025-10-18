import json
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from herbs_eval.dashboard import build_plot, export_json, run_evaluation, summarize_results
from herbs_eval.pipeline import EvalConfig


DEFAULT_MODELS = [
    "mock/any",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "ollama/llama3.1",
]


def _list_runs(limit: int = 20) -> List[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]


def _load_run(path: Path) -> Dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    st.set_page_config(page_title="LLM Herbs Eval Dashboard", layout="wide")
    st.title("LLM Herbs Eval — Streamlit Dashboard")
    st.caption("Deterministic evaluation of medicinal herb claims with risk-aware metrics.")

    with st.sidebar:
        st.subheader("Configuration")
        model = st.selectbox("Model", options=DEFAULT_MODELS, index=0)
        k = st.select_slider("k (responses per claim)", options=[1, 3, 5], value=1)
        max_claims = st.slider("Max claims", min_value=1, max_value=40, value=5, step=1)
        judge_prompt = st.radio("Judge prompt", options=["A", "B"], index=0, horizontal=True)
        temperature = st.slider("Temperature", min_value=0.0, max_value=0.3, value=0.1, step=0.05)
        seed = st.number_input("Seed", value=42, step=1)
        run_clicked = st.button("Run evaluation", use_container_width=True)

        st.markdown("---")
        st.subheader("Previous runs")
        runs = _list_runs()
        run_label_map = {"—": None}
        for p in runs:
            run_label_map[p.name] = p
        selected_run_label = st.selectbox("Inspect run", options=list(run_label_map.keys()), index=0)
        selected_run_path = run_label_map[selected_run_label]

    result: Dict | None = None

    if run_clicked:
        cfg = EvalConfig(
            model_name=model,
            judge_model_name=None,
            judge_prompt_version=str(judge_prompt),
            k=int(k),
            temperature=float(temperature),
            seed=int(seed),
        )
        with st.spinner("Running evaluation…"):
            result = run_evaluation(cfg, max_claims)
        st.success("Run complete")
        st.session_state["last_result"] = result
    elif selected_run_path:
        try:
            result = _load_run(selected_run_path)
            st.info(f"Loaded run from {selected_run_path}")
            st.session_state["last_result"] = result
        except Exception as exc:
            st.error(f"Failed to load run: {exc}")
    elif "last_result" in st.session_state:
        result = st.session_state["last_result"]

    if not result:
        st.warning("Run an evaluation or select a previous run to see results.")
        return

    metrics = result.get("metrics", {})
    summary = summarize_results(result.get("results", []))

    col_metrics, col_chart = st.columns([1, 1])
    with col_metrics:
        st.subheader("Metrics")
        st.json(metrics)
        st.subheader("Run directory")
        st.code(result.get("run_dir", ""))
        st.download_button(
            "Download JSON",
            export_json(result),
            file_name=f"herb_eval_{result.get('run_dir','run')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_chart:
        st.subheader("Severity overview")
        fig = build_plot(summary)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-claim worst-of-k summary")
    st.dataframe(summary, use_container_width=True)

    st.caption(
        "Results are stored under runs/. You can reopen them here or via the Gradio UI using the same files."
    )


if __name__ == "__main__":
    main()

