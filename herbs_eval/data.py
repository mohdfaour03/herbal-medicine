import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Claim:
    id: str
    claim: str
    ground_truth: str  # "true" | "false" | "uncertain"
    harm_category: str
    harm_weight: int
    source_urls: List[str]
    notes: Optional[str] = None


def load_claims(path: str = "data/claims.jsonl") -> List[Claim]:
    claims: List[Claim] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            claims.append(
                Claim(
                    id=obj["id"],
                    claim=obj["claim"],
                    ground_truth=obj["ground_truth"],
                    harm_category=obj["harm_category"],
                    harm_weight=int(obj.get("harm_weight", 2)),
                    source_urls=obj.get("source_urls", []),
                    notes=obj.get("notes"),
                )
            )
    return claims


def ensure_run_dir(base_dir: str = "runs") -> str:
    os.makedirs(base_dir, exist_ok=True)
    # timestamped directory but deterministic naming for repeated runs with same seed is not desired;
    # we want distinct runs to avoid overwrites.
    from datetime import datetime

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

