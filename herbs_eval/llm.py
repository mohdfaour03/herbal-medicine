import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# Load environment variables from .env if present (safe no-op if missing)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class LLMConfig:
    model: str  # e.g., "openai/gpt-4o-mini" or "ollama/llama3.1"
    temperature: float = 0.1
    max_tokens: int = 400
    seed: Optional[int] = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.seed is not None:
            set_global_seed(config.seed)

    def _provider_and_name(self) -> Tuple[str, str]:
        model = self.config.model.strip()
        if "/" in model:
            provider, name = model.split("/", 1)
        else:
            # default to openai
            provider, name = "openai", model
        return provider.lower(), name

    def chat(self, messages: List[Dict[str, str]]) -> str:
        provider, name = self._provider_and_name()
        temp = max(0.0, min(self.config.temperature, 0.3))
        if provider == "mock":
            # Deterministic canned output based on seed and content hash
            seed = self.config.seed or 0
            h = abs(hash(json.dumps(messages))) % 1000
            random.seed(seed + h)
            safe_reply = [
                "I cannot verify this fully; consult a clinician.",
                "Evidence is mixed; defer to medical guidance.",
                "Limited evidence; check reputable sources like NIH.",
            ]
            return safe_reply[h % len(safe_reply)]
        if provider == "openai":
            try:
                from openai import OpenAI

                client = OpenAI()
                kwargs: Dict[str, Any] = dict(
                    model=name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.config.max_tokens,
                )
                if self.config.seed is not None:
                    kwargs["seed"] = int(self.config.seed)
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                raise RuntimeError(f"OpenAI call failed: {e}")
        if provider == "ollama":
            # Minimal Ollama chat call
            try:
                url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
                payload = {
                    "model": name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        # Determinism is best-effort; seed not supported widely
                    },
                }
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "")
            except Exception as e:
                raise RuntimeError(f"Ollama call failed: {e}")
        raise ValueError(f"Unsupported provider: {provider}")
