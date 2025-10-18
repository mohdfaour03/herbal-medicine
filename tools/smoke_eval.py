import json
import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from herbs_eval.data import load_claims
from herbs_eval.simple import evaluate_claims_simple


def main():
    claims = load_claims()[:3]
    out = evaluate_claims_simple(
        claims,
        model_name="mock/any",
        k=3,
        temperature=0.1,
        seed=123,
        judge_prompt_version="A",
    )
    print(json.dumps({
        "metrics": out.get("metrics", {}),
        "run_dir": out.get("run_dir"),
    }, indent=2))


if __name__ == "__main__":
    main()
