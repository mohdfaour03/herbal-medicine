# llm-herbs-eval

Minimal, reproducible evaluation-first pipeline for LLM misinformation in medicinal herbs. Focus: assess risk on claims (no model training) using evidence-aligned prompts and harm-aware metrics.

- Evidence domain: NCCIH / MedlinePlus / NIH ODS / WHO references
- Components: harm taxonomy + HWM metric, LLM-as-judge with prompt A/B, LangGraph pipeline, Gradio UI, Streamlit dashboard
- Determinism: fixed seed, temperature ≤ 0.3, full JSON logging of prompts/responses
- Strong defaults with exposed dials: k ∈ {1,3,5}, model name, judge prompt A/B

## Quickstart

1) Install

```bash
pip install -r requirements.txt
```

Set credentials for your provider (example: OpenAI):

```bash
# macOS/Linux
export OPENAI_API_KEY=...

# Windows (new shell afterward)
setx OPENAI_API_KEY ...
```

2) Launch an interface

```bash
# Gradio (one-screen controls)
python app.py

# Streamlit + Plotly dashboard
streamlit run app_streamlit.py
```

Pick a model (e.g., `openai/gpt-4o-mini`), choose k and judge prompt (A/B), then run. Outputs include risk-aware metrics and per-claim judgements. Every run is logged under `runs/<timestamp>/` and can be reopened from either interface.

## Components

- Dataset: `data/claims.jsonl` — ≤40 claims with labels, harm category/weight, references
- Taxonomy: `herbs_eval/taxonomy.py` — harm categories and default weights
- LLM-as-judge: `herbs_eval/judge.py`; prompts in `herbs_eval/prompts.py` (A/B variants)
- Pipeline: `herbs_eval/pipeline.py` — LangGraph flow (generate → judge → aggregate)
- Metrics: `herbs_eval/metrics.py` — misinfo_rate@K and harm-weighted misinformation (HWM@K)
- UI: `herbs_eval/ui.py` — Gradio controls for quick runs
- Dashboard: `app_streamlit.py` + `herbs_eval/dashboard.py` — Streamlit + Plotly exploration, downloads & run history

## Metrics

- `misinfo_rate@K`: fraction of claims where any of the K responses is judged unsafe
- `HWM@K`: Σ(weight × severity × unsafe) / Σ(weight), severity ∈ {0, 0.4, 0.7, 1.0}

## Determinism & Logging

- Global seed for Python/NumPy; temperature clamped ≤0.3
- OpenAI client uses `seed` where supported; Ollama is best-effort deterministic
- JSON logging for each generate/judge event under `runs/`; summary written to `metrics.json`

## Notes

- `mock/any` model supports offline smoke tests (no API key)
- References embedded in the dataset span NCCIH, MedlinePlus, NIH ODS, FDA, WHO
- Streamlit dashboard exposes download buttons so results can be reused in Gradio (and vice versa)

