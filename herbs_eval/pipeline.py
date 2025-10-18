import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, START, END

from .data import Claim, ensure_run_dir
from .judge import Judgement, judge_answer
from .llm import LLMClient, LLMConfig, set_global_seed
from .metrics import ClaimResult, compute_metrics
from .prompts import RESPONSE_PROMPT


@dataclass
class EvalConfig:
    model_name: str = "openai/gpt-4o-mini"
    judge_model_name: Optional[str] = None  # if None, reuse model_name
    judge_prompt_version: str = "A"  # "A" or "B"
    k: int = 3
    temperature: float = 0.1
    seed: int = 42
    max_tokens: int = 400


class EvalState(dict):  # LangGraph state as dict
    pass


def _log_event(run_dir: str, event: Dict) -> None:
    path = os.path.join(run_dir, "events.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_graph(config: EvalConfig, run_dir: str):
    set_global_seed(config.seed)
    graph = StateGraph(EvalState)

    gen_llm = LLMClient(
        LLMConfig(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )
    judge_llm = LLMClient(
        LLMConfig(
            model=config.judge_model_name or config.model_name,
            temperature=min(config.temperature, 0.2),
            max_tokens=300,
            seed=config.seed,
        )
    )

    def node_generate(state: EvalState) -> EvalState:
        claim: Claim = state["claim"]
        k = max(1, min(int(config.k), 5))
        messages = [
            {"role": "system", "content": RESPONSE_PROMPT},
            {
                "role": "user",
                "content": f"Claim: {claim.claim}\nRespond with a concise evaluation.",
            },
        ]
        responses: List[str] = []
        for i in range(k):
            # Seed hop for deterministic diversity
            gen_llm.config.seed = int(config.seed) + i
            text = gen_llm.chat(messages)
            responses.append(text)
            _log_event(
                run_dir,
                {
                    "type": "generate",
                    "claim_id": claim.id,
                    "k_index": i,
                    "messages": messages,
                    "response": text,
                },
            )
        state["responses"] = responses
        return state

    def node_judge(state: EvalState) -> EvalState:
        claim: Claim = state["claim"]
        judgements: List[Judgement] = []
        for i, answer in enumerate(state["responses"]):
            j = judge_answer(
                claim=claim.claim,
                ground_truth=claim.ground_truth,
                source_urls=claim.source_urls,
                answer=answer,
                judge_prompt_version=config.judge_prompt_version,
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
        state["judgements"] = judgements
        return state

    def node_aggregate(state: EvalState) -> EvalState:
        # No-op here; aggregation in driver
        return state

    graph.add_node("generate", node_generate)
    graph.add_node("judge", node_judge)
    graph.add_node("aggregate", node_aggregate)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "judge")
    graph.add_edge("judge", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def evaluate_claims(claims: List[Claim], config: EvalConfig) -> Dict:
    run_dir = ensure_run_dir()
    # Persist config and claims snapshot for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
    with open(os.path.join(run_dir, "claims.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    app = build_graph(config, run_dir)
    per_claim_results: List[ClaimResult] = []
    rows: List[Dict] = []

    for claim in claims:
        state = {"claim": claim}
        out = app.invoke(state)
        judgements: List[Judgement] = out["judgements"]
        # Normalize for metrics
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
        per_claim_results.append(
            ClaimResult(
                claim_id=claim.id, harm_weight=claim.harm_weight, judgements=judg_norm
            )
        )
        rows.append(
            {
                "id": claim.id,
                "claim": claim.claim,
                "harm_category": claim.harm_category,
                "harm_weight": claim.harm_weight,
                "judgements": judg_norm,
            }
        )

    metrics = compute_metrics(per_claim_results)
    summary = {"metrics": metrics, "results": rows, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

