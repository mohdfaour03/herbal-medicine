from typing import Dict, List

# Ensure audioop is available on Python 3.13 (via audioop-lts / pyaudioop shim)
try:
    import audioop  # noqa: F401
except Exception:
    pass

import json
import gradio as gr

from .data import load_claims
from .pipeline import EvalConfig
from .simple import evaluate_claims_simple
from .metrics import aggregate_worst_of_k


def _summarize_results(rows: List[Dict]) -> List[Dict]:
    summarized = []
    for r in rows:
        judgements = r.get("judgements", [])
        _, _, worst = aggregate_worst_of_k(judgements)
        summarized.append(
            {
                "id": r.get("id"),
                "claim": r.get("claim"),
                "harm_category": r.get("harm_category"),
                "harm_weight": r.get("harm_weight"),
                "worst": worst,
                "judgements": judgements,
            }
        )
    return summarized


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Herb Misinformation Eval") as demo:
        gr.Markdown(
            """
            # LLM Herbs Eval — Risk-Aware
            Evaluate misinformation risk on medicinal herb claims using a small, deterministic pipeline.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(
                    label="Model Name",
                    value="mock/any",
                    choices=[
                        "mock/any",
                        "openai/gpt-4o-mini",
                        "openai/gpt-4o",
                        "ollama/llama3.1",
                    ],
                )
                k = gr.Radio(label="k (responses per claim)", choices=[1, 3, 5], value=1)
                max_claims = gr.Slider(label="Max claims", minimum=1, maximum=40, step=1, value=5)
                judge_prompt = gr.Radio(label="Judge prompt", choices=["A", "B"], value="A")
                temperature = gr.Slider(
                    label="Temperature (<=0.3)", minimum=0.0, maximum=0.3, step=0.05, value=0.1
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                run_btn = gr.Button("Run Evaluation", variant="primary")
                status = gr.Markdown("Ready.")

            with gr.Column(scale=2):
                out_metrics = gr.JSON(label="Metrics")
                out_table = gr.JSON(label="Per-claim worst-of-k summary")
                run_dir = gr.Textbox(label="Run directory", interactive=False)

        def _run(model_name, k_val, max_n, judge_v, temp, seed_val):
            status_text = "Running..."
            try:
                cfg = EvalConfig(
                    model_name=model_name,
                    judge_model_name=None,
                    judge_prompt_version=str(judge_v),
                    k=int(k_val),
                    temperature=float(temp),
                    seed=int(seed_val),
                )
                claims = load_claims()[: int(max_n)]
                out = evaluate_claims_simple(
                    claims,
                    model_name=cfg.model_name,
                    k=cfg.k,
                    temperature=cfg.temperature,
                    seed=cfg.seed,
                    judge_prompt_version=cfg.judge_prompt_version,
                )
                metrics = out.get("metrics", {})
                rows = _summarize_results(out.get("results", []))
                status_text = "Done."
                return metrics, rows, out.get("run_dir", ""), status_text
            except Exception as e:
                return {}, [], "", f"Error: {e}"

        run_btn.click(
            _run,
            inputs=[model, k, max_claims, judge_prompt, temperature, seed],
            outputs=[out_metrics, out_table, run_dir, status],
        )

        gr.Markdown(
            "Note: Start with mock/any or k=1 & small Max claims to keep runs fast/cost-effective."
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()

