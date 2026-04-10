"""
evaluation/assets.py

Helpers for downloading missing datasets, KB files, and pretrained models.
"""

from __future__ import annotations

import ast
import gzip
import json
import random
import shutil
import zipfile
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import snapshot_download


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOOKUPS_DIR = DATA_DIR / "lookups" / "mesh"
DOWNLOADS_DIR = DATA_DIR / "_downloads"
MODELS_DIR = ROOT / "models"


MODEL_SPECS = {
    "pubmedbert": {
        "repo_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "local_dir": MODELS_DIR / "pubmedbert-local",
    },
    "sapbert": {
        "repo_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "local_dir": MODELS_DIR / "sapbert-local",
    },
    "biobert": {
        "repo_id": "dmis-lab/biobert-base-cased-v1.1",
        "local_dir": MODELS_DIR / "biobert-local",
    },
    "minilm": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": MODELS_DIR / "minilm-local",
    },
}


NCBI_URLS = {
    "train": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
    "dev": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
    "test": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
}

BC5CDR_URL = "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"

CTD_URLS = {
    "CTD_diseases.tsv": "https://ctdbase.org/reports/CTD_diseases.tsv.gz",
    "CTD_chemicals.tsv": "https://ctdbase.org/reports/CTD_chemicals.tsv.gz",
}


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)
    return dest


def _download_and_extract_zip(url: str, zip_path: Path, extract_to: Path) -> None:
    _download_file(url, zip_path)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_to)


def _write_jsonl(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _train_valid_test_split(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = max(1, int(total * 0.8))
    valid_end = max(train_end + 1, int(total * 0.9))

    train_rows = shuffled[:train_end]
    valid_rows = shuffled[train_end:valid_end]
    test_rows = shuffled[valid_end:]

    if not valid_rows:
        valid_rows = shuffled[-1:]
    if not test_rows:
        test_rows = shuffled[-1:]

    return {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }


def _format_trial_text(raw_trial: str) -> str:
    if not raw_trial:
        return ""

    try:
        trial = ast.literal_eval(raw_trial)
    except (ValueError, SyntaxError):
        return str(raw_trial).strip()

    parts = []
    trial_id = str(trial.get("Clinical Trial ID", "")).strip()
    if trial_id:
        parts.append(f"Clinical Trial ID: {trial_id}")

    for section_name in ("Intervention", "Eligibility", "Results", "Adverse Events"):
        value = trial.get(section_name, "")
        if not value:
            continue
        if isinstance(value, list):
            text = " ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
        if text:
            parts.append(f"{section_name}: {text}")

    return "\n".join(parts).strip()


def _convert_nli4ct_row(row: dict) -> dict:
    primary = _format_trial_text(row.get("Primary_ct", ""))
    secondary = _format_trial_text(row.get("Secondary_ct", ""))
    section = str(row.get("Section_id", "")).strip()
    comparison_type = str(row.get("Type", "")).strip()

    premise_parts = []
    if section:
        premise_parts.append(f"Section: {section}")
    if comparison_type:
        premise_parts.append(f"Task Type: {comparison_type}")
    if primary:
        premise_parts.append(f"Primary Trial:\n{primary}")
    if secondary:
        premise_parts.append(f"Secondary Trial:\n{secondary}")

    return {
        "sentence1": "\n\n".join(part for part in premise_parts if part).strip(),
        "sentence2": str(row.get("Statement", "")).strip(),
        "gold_label": str(row.get("Label", "")).strip(),
    }


def ensure_model(model_name: str) -> str:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"no download spec configured for model: {model_name}")

    local_dir = spec["local_dir"]
    if (local_dir / "config.json").exists():
        return str(local_dir)

    print(f"downloading model '{model_name}' from {spec['repo_id']} ...")
    snapshot_download(repo_id=spec["repo_id"], local_dir=str(local_dir))
    return str(local_dir)


def ensure_ncbi_dataset() -> None:
    out_dir = RAW_DIR / "ncbi_disease"
    sentinels = [
        out_dir / "NCBItrainset_corpus.txt",
        out_dir / "NCBIdevelopset_corpus.txt",
        out_dir / "NCBItestset_corpus.txt",
    ]
    if all(path.exists() for path in sentinels):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, url in NCBI_URLS.items():
        zip_path = DOWNLOADS_DIR / "ncbi_disease" / f"{split}.zip"
        _download_and_extract_zip(url, zip_path, out_dir)


def ensure_bc5cdr_dataset() -> None:
    out_dir = RAW_DIR / "bc5cdr"
    xml_root = out_dir / "CDR_Data" / "CDR.Corpus.v010516"
    sentinels = [
        xml_root / "CDR_TrainingSet.BioC.xml",
        xml_root / "CDR_DevelopmentSet.BioC.xml",
        xml_root / "CDR_TestSet.BioC.xml",
    ]
    if all(path.exists() for path in sentinels):
        return

    zip_path = DOWNLOADS_DIR / "bc5cdr" / "CDR_Data.zip"
    _download_and_extract_zip(BC5CDR_URL, zip_path, out_dir)


def ensure_biosses_dataset() -> None:
    out_dir = RAW_DIR / "biosses"
    sentinels = [out_dir / "train.jsonl", out_dir / "validation.jsonl", out_dir / "test.jsonl"]
    if all(path.exists() for path in sentinels):
        return

    dataset = load_dataset("biosses", split="train")
    rows = [
        {
            "text_1": row["sentence1"].strip(),
            "text_2": row["sentence2"].strip(),
            "label": float(row["score"]),
        }
        for row in dataset
    ]
    for split, split_rows in _train_valid_test_split(rows).items():
        _write_jsonl(split_rows, out_dir / f"{split}.jsonl")


def ensure_nli4ct_dataset() -> None:
    out_dir = RAW_DIR / "nli4ct"
    sentinels = [out_dir / "train.jsonl", out_dir / "validation.jsonl", out_dir / "test.jsonl"]
    if all(path.exists() for path in sentinels):
        return

    train_rows = [_convert_nli4ct_row(row) for row in load_dataset("tasksource/nli4ct", split="train")]
    valid_rows = [_convert_nli4ct_row(row) for row in load_dataset("tasksource/nli4ct", split="validation")]

    _write_jsonl(train_rows, out_dir / "train.jsonl")
    _write_jsonl(valid_rows, out_dir / "validation.jsonl")
    _write_jsonl(valid_rows, out_dir / "test.jsonl")


def ensure_mesh_lookups() -> None:
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in CTD_URLS.items():
        out_path = LOOKUPS_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        gz_path = DOWNLOADS_DIR / "mesh" / f"{filename}.gz"
        _download_file(url, gz_path)
        with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def ensure_entity_linking_assets(dataset: str | None = None) -> None:
    ensure_mesh_lookups()
    if dataset in (None, "ncbi"):
        ensure_ncbi_dataset()
    if dataset in (None, "bc5cdr_d", "bc5cdr_c"):
        ensure_bc5cdr_dataset()


def ensure_sts_assets(dataset: str | None = None) -> None:
    if dataset in (None, "biosses"):
        ensure_biosses_dataset()


def ensure_nli_assets(dataset: str | None = None) -> None:
    if dataset in (None, "nli4ct"):
        ensure_nli4ct_dataset()


def ensure_task_assets(task: str = "all", model_name: str | None = None) -> None:
    if model_name and model_name in MODEL_SPECS:
        ensure_model(model_name)

    if task in ("all", "entity_linking"):
        ensure_entity_linking_assets()
    if task in ("all", "sts"):
        ensure_sts_assets()
    if task in ("all", "nli"):
        ensure_nli_assets()
