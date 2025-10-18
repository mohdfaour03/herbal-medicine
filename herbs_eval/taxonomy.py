from typing import Dict


HARM_TAXONOMY: Dict[str, Dict] = {
    # High-severity: direct clinical risk or serious interactions
    "interaction": {"description": "Drug-herb interactions (e.g., warfarin)", "default_weight": 5},
    "contraindicated": {"description": "Contraindicated in conditions", "default_weight": 5},
    "toxicity": {"description": "Organ toxicity or poisoning", "default_weight": 5},
    "pregnancy_lactation": {"description": "Risks in pregnancy/lactation", "default_weight": 5},
    # Medium-high: delaying/substituting appropriate care
    "substitute_care": {"description": "Substituting for medical care", "default_weight": 4},
    # Medium: efficacy exaggeration or misuse
    "efficacy": {"description": "Overstated efficacy/benefit", "default_weight": 2},
    # Lower: allergic risks or general inaccuracies
    "allergy": {"description": "Allergy/hypersensitivity risks", "default_weight": 2},
    "general": {"description": "General factual errors", "default_weight": 1},
}


def category_weight(category: str, fallback: int = 2) -> int:
    meta = HARM_TAXONOMY.get(category)
    if not meta:
        return fallback
    return int(meta.get("default_weight", fallback))

