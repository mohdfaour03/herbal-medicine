import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict
from urllib.parse import urlparse

from langgraph.graph import START, END, StateGraph

from .data import Claim, ensure_run_dir
from .judge import (
    EvidenceJudgement,
    Judgement,
    judge_answer,
    judge_claim_with_evidence,
)
from .llm import LLMClient, LLMConfig, set_global_seed
from .metrics import ClaimResult, compute_metrics
from .prompts import RESPONSE_PROMPT
from .tools import EvidenceItem, SearchTool


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


@dataclass
class EvidenceEvalConfig:
    model_name: str = "openai/gpt-4o-mini"
    judge_model_name: Optional[str] = None
    seed: int = 42
    max_tokens: int = 400
    use_search: bool = True
    max_sources: int = 3
    search_timeout: float = 15.0


class EvidenceState(TypedDict, total=False):
    claim_text: str
    evidence: List[EvidenceItem]
    judgement: EvidenceJudgement


def build_evidence_graph(
    config: EvidenceEvalConfig, run_dir: str, search_tool: SearchTool
):
    set_global_seed(config.seed)
    graph = StateGraph(EvidenceState)

    judge_llm = LLMClient(
        LLMConfig(
            model=config.judge_model_name or config.model_name,
            temperature=0.0,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )

    def node_search(state: EvidenceState) -> Dict:
        claim_text: str = state["claim_text"]
        evidence: List[EvidenceItem] = []
        if config.use_search:
            evidence = search_tool.search(
                claim_text, k=config.max_sources, timeout=config.search_timeout
            )
        _log_event(
            run_dir,
            {
                "type": "search",
                "claim_text": claim_text,
                "use_search": bool(config.use_search),
                "results": [
                    {
                        "title": item.title,
                        "url": item.url,
                        "domain": item.source_domain,
                        "snippet": item.snippet,
                    }
                    for item in evidence
                ],
            },
        )
        return {"evidence": evidence}

    def node_judge(state: EvidenceState) -> Dict:
        claim_text: str = state["claim_text"]
        evidence: List[EvidenceItem] = state.get("evidence", [])
        judgement: EvidenceJudgement
        if evidence:
            judgement = judge_claim_with_evidence(claim_text, evidence, judge_llm)
        else:
            judgement = EvidenceJudgement(
                label="unsupported",
                rationale="No authoritative evidence was provided.",
                citations=[],
                raw_text="",
                error="no_evidence",
            )
        state["judgement"] = judgement
        _log_event(
            run_dir,
            {
                "type": "evidence_judge",
                "claim_text": claim_text,
                "judgement": {
                    "label": judgement.label,
                    "rationale": judgement.rationale,
                    "citations": judgement.citations,
                    "error": judgement.error,
                },
            },
        )
        return {"judgement": judgement}

    graph.add_node("search", node_search)
    graph.add_node("judge", node_judge)
    graph.add_edge(START, "search")
    graph.add_edge("search", "judge")
    graph.add_edge("judge", END)
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


def _enrich_citations(citations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for entry in citations:
        url = entry.get("url", "")
        domain = ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
        enriched.append(
            {
                "title": entry.get("title", ""),
                "url": url,
                "domain": domain,
            }
        )
    return enriched


def evaluate_freeform_claim(
    claim_text: str,
    config: EvidenceEvalConfig,
    *,
    search_tool: Optional[SearchTool] = None,
) -> Dict:
    clean_claim = (claim_text or "").strip()
    run_dir = ensure_run_dir()
    search_tool = search_tool or SearchTool()

    config_payload = {
        "model_name": config.model_name,
        "judge_model_name": config.judge_model_name or config.model_name,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "use_search": config.use_search,
        "max_sources": config.max_sources,
        "search_timeout": config.search_timeout,
        "claim_text": clean_claim,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    app = build_evidence_graph(config, run_dir, search_tool)
    state: EvidenceState = {"claim_text": clean_claim}
    out = app.invoke(state)
    judgement: EvidenceJudgement = out["judgement"]
    evidence: List[EvidenceItem] = out.get("evidence", [])  # type: ignore[arg-type]

    citations = _enrich_citations(judgement.citations)
    summary = {
        "claim": clean_claim,
        "label": judgement.label,
        "rationale": judgement.rationale,
        "citations": citations,
        "evidence": [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "domain": item.source_domain,
            }
            for item in evidence
        ],
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "claim": clean_claim,
                "label": judgement.label,
                "rationale": judgement.rationale,
                "citations": citations,
            },
            f,
            indent=2,
        )
    return summary
