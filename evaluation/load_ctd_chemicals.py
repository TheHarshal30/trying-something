from __future__ import annotations

import re
from pathlib import Path


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split()).strip()


def is_valid_name(name) -> bool:
    if name is None:
        return False

    name = str(name).strip().lower()
    if name == "" or name == "nan":
        return False
    if name.startswith("dtxsid"):
        return False
    if len(name) > 20 and name.isalnum():
        return False
    return True


def read_ctd_chemicals(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "ChemicalName": parts[0] if len(parts) > 0 else "",
                    "ChemicalID": parts[1] if len(parts) > 1 else "",
                    "Synonyms": parts[11] if len(parts) > 11 else "",
                }
            )
    return rows


def load_clean_kb(path: Path) -> tuple[list[str], list[str]]:
    rows = read_ctd_chemicals(path)

    print("\n[RAW KB SAMPLE]")
    for row in rows[:5]:
        print(row)

    if rows:
        print("[KB COLUMNS]", list(rows[0].keys()))

    kb_terms: list[str] = []
    kb_ids: list[str] = []

    for row in rows:
        name = row.get("ChemicalName")
        cid = row.get("ChemicalID")

        if not is_valid_name(name):
            continue
        if cid is None or str(cid).strip() == "" or str(cid).strip().lower() == "nan":
            continue

        name_norm = normalize(name)
        cid_norm = str(cid).strip()

        if not name_norm:
            continue
        if name_norm.startswith("dtxsid"):
            continue
        if len(name_norm) > 20 and name_norm.isalnum():
            continue

        kb_terms.append(name_norm)
        kb_ids.append(cid_norm)

        syns = row.get("Synonyms")
        if syns and str(syns).strip().lower() != "nan":
            for syn in str(syns).split("|"):
                if not is_valid_name(syn):
                    continue
                syn_norm = normalize(syn)
                if syn_norm:
                    kb_terms.append(syn_norm)
                    kb_ids.append(cid_norm)

    seen = set()
    dedup_terms: list[str] = []
    dedup_ids: list[str] = []
    for term, cid in zip(kb_terms, kb_ids):
        key = (term, cid)
        if key in seen:
            continue
        seen.add(key)
        dedup_terms.append(term)
        dedup_ids.append(cid)

    print("\n[KB SANITY CHECK]")
    for term in dedup_terms[:20]:
        print(term)

    assert len(dedup_terms) > 10000, f"KB too small -> wrong file ({len(dedup_terms)} terms)"
    assert any("aspirin" in term for term in dedup_terms), "KB missing basic chemicals"

    return dedup_terms, dedup_ids
