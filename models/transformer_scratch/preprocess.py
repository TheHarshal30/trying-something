from __future__ import annotations

import re


def normalize_chemical(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, strategy: str = "chemical") -> str:
    if strategy == "chemical":
        return normalize_chemical(text)
    if strategy in {"none", "basic"}:
        return text.strip()
    raise ValueError(f"unknown normalization strategy: {strategy}")
