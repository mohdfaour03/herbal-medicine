import time
from typing import Dict, List

# Ensure audioop is available on Python 3.13 (via audioop-lts / pyaudioop shim)
try:
    import audioop  # noqa: F401
except Exception:
    pass

import gradio as gr

from .simple import evaluate_freeform_claim


def _format_citations(citations: List[Dict[str, str]]) -> str:
    if not citations:
        return "_No citations available._"
    lines = []
    for item in citations:
        title = item.get("title") or "Source"
        url = item.get("url") or ""
        domain = item.get("domain") or ""
        if url:
            line = f"- [{title}]({url}) · {domain}"
        else:
            line = f"- {title} · {domain}"
        lines.append(line)
    return "\n".join(lines)


def _format_evidence(evidence: List[Dict[str, str]]) -> str:
    if not evidence:
        return "_No evidence snippets retrieved._"
    rendered: List[str] = []
    for idx, item in enumerate(evidence, start=1):
        title = item.get("title") or "Source"
        domain = item.get("domain") or item.get("source_domain") or ""
        url = item.get("url") or ""
        snippet = item.get("snippet", "").strip()
        header = f"**{idx}. [{title}]({url}) · {domain}**" if url else f"**{idx}. {title} · {domain}**"
        rendered.append(f"{header}\n> {snippet or '_No snippet available._'}")
    return "\n\n".join(rendered)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Herb Claim Fact Checker") as demo:
        gr.Markdown(
            """
            # LLM Herbs Eval — Evidence-Aware Fact Check
            Paste a medicinal herb claim and optionally retrieve authoritative evidence before judging.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                claim_box = gr.Textbox(
                    label="Claim",
                    placeholder='e.g., "Ginkgo does not affect blood thinners like warfarin."',
                    lines=4,
                )
                use_search = gr.Checkbox(label="Use web search", value=True)
                max_sources = gr.Slider(
                    label="Max sources",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                )
                model_name = gr.Textbox(
                    label="Model name",
                    value="openai/gpt-4o-mini",
                    placeholder="Provider/model identifier",
                )
                evaluate_btn = gr.Button("Evaluate claim", variant="primary")
                status = gr.Markdown("Ready.")

            with gr.Column(scale=1.2):
                label_out = gr.HTML(value="<div class='result-pill unsupported'>Predicted label: unsupported</div>")
                rationale_out = gr.Textbox(label="Rationale", lines=3, interactive=False)
                citations_out = gr.Markdown("_No citations available._")
                with gr.Accordion("Evidence snippets", open=False):
                    evidence_out = gr.Markdown("_No evidence snippets retrieved._")
                run_dir_out = gr.Textbox(label="Run directory", interactive=False)

        demo.load(
            lambda: None,
            None,
            None,
            js="""
            () => {
                const style = document.createElement('style');
                style.innerHTML = `
                    .result-pill { padding: 0.6rem 0.8rem; border-radius: 999px; font-weight: 600; display:inline-block; }
                    .result-pill.correct { background: rgba(60, 179, 113, 0.18); color: #1f7a4d; }
                    .result-pill.incorrect { background: rgba(220, 20, 60, 0.2); color: #b20c30; }
                    .result-pill.unsupported { background: rgba(255, 193, 7, 0.18); color: #b37400; }
                `;
                document.head.appendChild(style);
            }
            """,
        )

        def _evaluate(claim_text: str, search_enabled: bool, k_sources: int, model: str):
            clean_claim = (claim_text or "").strip()
            if not clean_claim:
                yield (
                    "<div class='result-pill unsupported'>Predicted label: unsupported</div>",
                    "Please provide a claim to evaluate.",
                    "_No citations available._",
                    "_No evidence snippets retrieved._",
                    "",
                    "Unsupported — empty input.",
                )
                return

            running_tuple = (
                "<div class='result-pill unsupported'>Evaluating…</div>",
                "",
                "_Fetching citations..._",
                "_Retrieving evidence..._",
                "",
                "Running...",
            )
            yield running_tuple
            try:
                result = evaluate_freeform_claim(
                    clean_claim,
                    model_name=model.strip() or "openai/gpt-4o-mini",
                    use_search=bool(search_enabled),
                    max_sources=int(k_sources),
                )
                label = result.get("label", "unsupported")
                rationale = result.get("rationale", "")
                citations = _format_citations(result.get("citations", []))
                evidence_md = _format_evidence(result.get("evidence", []))
                run_dir = result.get("run_dir", "")
                label_html = f"<div class='result-pill {label}'>Predicted label: {label}</div>"
                rationale = rationale or "No rationale provided."
                buffer = ""
                for ch in rationale:
                    buffer += ch
                    yield (
                        label_html,
                        buffer,
                        citations,
                        evidence_md,
                        run_dir,
                        "Generating rationale...",
                    )
                    time.sleep(0.02)
                yield (
                    label_html,
                    rationale,
                    citations,
                    evidence_md,
                    run_dir,
                    "Done.",
                )
            except Exception as exc:  # noqa: BLE001
                yield (
                    "<div class='result-pill unsupported'>Predicted label: unsupported</div>",
                    f"Error: {exc}",
                    "_No citations available._",
                    "_No evidence snippets retrieved._",
                    "",
                    "Error during evaluation.",
                )
                return

        evaluate_btn.click(
            _evaluate,
            inputs=[claim_box, use_search, max_sources, model_name],
            outputs=[label_out, rationale_out, citations_out, evidence_out, run_dir_out, status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
