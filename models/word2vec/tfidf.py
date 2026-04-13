from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def compute_tfidf(corpus) -> dict[str, float]:
    df = Counter()
    total_docs = 0

    for doc in corpus:
        tokens = list(doc)
        if not tokens:
            continue
        total_docs += 1
        for word in set(tokens):
            df[word] += 1

    if total_docs == 0:
        return {}

    return {
        word: math.log(total_docs / (count + 1))
        for word, count in df.items()
    }


def save_tfidf(idf: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(idf, handle)


def load_tfidf(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): float(value) for key, value in raw.items()}
