"""Authoritative web search helper for the herb fact-checker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

import requests

AUTHORITY_DOMAINS = {
    "nccih.nih.gov",
    "ods.od.nih.gov",
    "medlineplus.gov",
    "fda.gov",
    "who.int",
    "cochrane.org",
    "livertox.nih.gov",
}


@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    source_domain: str


class SearchTool:
    """SerpAPI-backed search helper with authoritative-domain filtering."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.endpoint = endpoint or os.getenv("SERPAPI_URL", "https://serpapi.com/search")
        self.session = session or requests.Session()

    def _query(self, query: str, timeout: float) -> List[Dict]:
        params = {
            "engine": "google",
            "q": query,
            "num": 10,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us",
        }
        resp = self.session.get(self.endpoint, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("organic_results") or []

    def _collect_from_results(
        self,
        items: List[Dict],
        seen_domains: set,
        limit: int,
    ) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        for item in items:
            raw_url = (item.get("link") or item.get("url") or "").strip()
            if not raw_url:
                continue
            parsed = urlparse(raw_url)
            domain = parsed.netloc.lower().lstrip("www.")
            if domain not in AUTHORITY_DOMAINS:
                continue
            if domain in seen_domains:
                continue

            clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
            title = (item.get("title") or item.get("name") or "").strip()[:200]
            snippet = (
                item.get("snippet")
                or item.get("rich_snippet", {}).get("top", {}).get("extensions", [])
                or ""
            )
            if isinstance(snippet, list):
                snippet = " ".join(snippet)
            snippet = str(snippet).replace("\n", " ").strip()
            if not title and not snippet:
                continue
            snippet = snippet[:400]

            evidence.append(
                EvidenceItem(
                    title=title or domain,
                    url=clean_url,
                    snippet=snippet,
                    source_domain=domain,
                )
            )
            seen_domains.add(domain)
            if len(evidence) >= limit:
                break
        return evidence

    def search(self, claim_text: str, k: int = 3, timeout: float = 15.0) -> List[EvidenceItem]:
        query = (claim_text or "").strip()
        k = max(1, min(int(k or 1), 5))
        if not query or not self.api_key:
            return []

        evidence: List[EvidenceItem] = []
        seen_domains: set = set()

        try:
            initial_results = self._query(query, timeout)
            evidence.extend(
                self._collect_from_results(initial_results, seen_domains, k)
            )
        except Exception:
            return []

        if len(evidence) >= k:
            return evidence[:k]

        remaining = k - len(evidence)
        for domain in AUTHORITY_DOMAINS:
            if remaining <= 0:
                break
            try:
                domain_results = self._query(f"{query} site:{domain}", timeout)
            except Exception:
                continue
            new_items = self._collect_from_results(domain_results, seen_domains, remaining)
            evidence.extend(new_items)
            remaining = k - len(evidence)

        return evidence[:k]
