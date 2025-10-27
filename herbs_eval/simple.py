import json
import os
from typing import Dict, List

from .data import Claim, ensure_run_dir
from .llm import LLMClient, LLMConfig, set_global_seed
from .metrics import ClaimResult, compute_metrics
from .pipeline import EvidenceEvalConfig, evaluate_freeform_claim as pipeline_evaluate_freeform_claim
from .prompts import RESPONSE_PROMPT
from .judge import judge_answer
from .tools import SearchTool


def _log_event(run_dir: str, event: Dict) -> None:
    path = os.path.join(run_dir, "events.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def evaluate_claims_simple(claims: List[Claim], *, model_name: str = "mock/any", k: int = 3, seed: int = 42, temperature: float = 0.1, judge_prompt_version: str = "A") -> Dict:
    set_global_seed(seed)
    run_dir = ensure_run_dir()

    # Persist snapshot
    cfg_obj = {
        "model_name": model_name,
        "k": int(k),
        "seed": int(seed),
        "temperature": float(temperature),
        "judge_prompt_version": judge_prompt_version,
        "driver": "simple",
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_obj, f, indent=2)
    with open(os.path.join(run_dir, "claims.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    gen_llm = LLMClient(LLMConfig(model=model_name, temperature=temperature, max_tokens=400, seed=seed))
    judge_llm = LLMClient(LLMConfig(model=model_name, temperature=min(temperature, 0.2), max_tokens=300, seed=seed))

    rows: List[Dict] = []
    per_claim_results: List[ClaimResult] = []

    for claim in claims:
        messages = [
            {"role": "system", "content": RESPONSE_PROMPT},
            {"role": "user", "content": f"Claim: {claim.claim}\nRespond with a concise evaluation."},
        ]
        responses: List[str] = []
        kk = max(1, min(int(k), 5))
        for i in range(kk):
            gen_llm.config.seed = int(seed) + i
            text = gen_llm.chat(messages)
            responses.append(text)
            _log_event(run_dir, {"type": "generate", "claim_id": claim.id, "k_index": i, "messages": messages, "response": text})

        judgements = []
        for i, answer in enumerate(responses):
            j = judge_answer(
                claim=claim.claim,
                ground_truth=claim.ground_truth,
                source_urls=claim.source_urls,
                answer=answer,
                judge_prompt_version=judge_prompt_version,
                llm=judge_llm,
            )
            judgements.append(j)
            _log_event(
                run_dir,
                {
                    "type": "judge",
                    "claim_id": claim.id,
                    "k_index": i,
                    "judgement": {
                        "verdict": j.verdict,
                        "harm_level": j.harm_level,
                        "confidence": j.confidence,
                        "rationale": j.rationale,
                        "flags": j.flags,
                    },
                    "raw": j.raw_text,
                },
            )

        judg_norm = [
            {
                "verdict": j.verdict,
                "harm_level": j.harm_level,
                "confidence": j.confidence,
                "rationale": j.rationale,
                "flags": j.flags,
            }
            for j in judgements
        ]
        per_claim_results.append(ClaimResult(claim_id=claim.id, harm_weight=claim.harm_weight, judgements=judg_norm))
        rows.append({
            "id": claim.id,
            "claim": claim.claim,
            "harm_category": claim.harm_category,
            "harm_weight": claim.harm_weight,
            "judgements": judg_norm,
        })

    metrics = compute_metrics(per_claim_results)
    summary = {"metrics": metrics, "results": rows, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_freeform_claim(
    claim_text: str,
    *,
    model_name: str = "openai/gpt-4o-mini",
    judge_model_name: str | None = None,
    seed: int = 42,
    max_tokens: int = 400,
    use_search: bool = True,
    max_sources: int = 3,
    search_timeout: float = 15.0,
) -> Dict:
    """Evaluate a free-form claim by retrieving evidence and judging."""
    config = EvidenceEvalConfig(
        model_name=model_name,
        judge_model_name=judge_model_name,
        seed=seed,
        max_tokens=max_tokens,
        use_search=use_search,
        max_sources=max(1, min(int(max_sources or 1), 5)),
        search_timeout=search_timeout,
    )
    search_tool = SearchTool()
    return pipeline_evaluate_freeform_claim(
        claim_text,
        config,
        search_tool=search_tool,
    )
