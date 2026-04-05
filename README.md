# MedBench — Medical Embedding Evaluation Suite

A benchmarking framework for evaluating medical text embeddings across multiple downstream tasks. Built to compare custom-trained biomedical embedding models against pretrained baselines on standardized datasets.

---

## Overview

Medical embeddings are vector representations of clinical and biomedical text. Good embeddings should capture semantic meaning — diseases with similar symptoms should be close in vector space, and different conditions should be far apart.

This project evaluates embeddings on four downstream tasks:

| Task | Dataset | Metric |
|------|---------|--------|
| Entity Linking | NCBI Disease | Acc@1, Acc@5, MRR |
| Entity Linking | BC5CDR-d (diseases) | Acc@1, Acc@5, MRR |
| Entity Linking | BC5CDR-c (chemicals) | Acc@1, Acc@5, MRR |
| Semantic Similarity | BIOSSES | Pearson r |
| Natural Language Inference | NLI4CT | Accuracy, Macro F1 |

---

## Project structure

```
medical-entity-linking/
├── data/
│   ├── raw/              # datasets (not in git — see setup)
│   └── lookups/          # MeSH / UMLS KB files (not in git)
├── models/               # model architecture + weights (weights not in git)
│   ├── pubmedbert-local/ # pretrained baseline
│   ├── sapbert-local/    # pretrained baseline
│   ├── word2vec/         # team model
│   ├── word2vec_umls/    # team model
│   ├── transformer/      # team model
│   └── transformer_umls/ # team model
├── evaluation/           # benchmark scripts
│   ├── base_embedder.py          # model interface contract
│   ├── pubmedbert_embedder.py    # transformer embedder (reused for all pretrained)
│   ├── eval_entity_linking.py    # NCBI + BC5CDR evaluation
│   ├── eval_sts.py               # BIOSSES evaluation
│   ├── eval_nli.py               # NLI4CT evaluation
│   └── run_all.py                # master runner
├── notebooks/            # EDA notebooks per dataset
├── results/              # benchmark outputs (JSON + figures)
├── CONTRIBUTING_MODELS.md
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10+, M1 Mac or CUDA GPU recommended

**1. Clone the repo**

```bash
git clone https://github.com/your-org/medical-entity-linking.git
cd medical-entity-linking
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download team model weights**

Weights are not stored in git. Download from the shared Google Drive folder and place them in the correct `models/*/weights/` directories. See `CONTRIBUTING_MODELS.md` for the expected structure.

---

## Running benchmarks

**Run all tasks for a single model:**

```bash
cd evaluation
python run_all.py --model pubmedbert
```

**Run a specific task only:**

```bash
python run_all.py --model pubmedbert --task entity_linking
python run_all.py --model pubmedbert --task sts
python run_all.py --model pubmedbert --task nli
```

**Run on a specific dataset:**

```bash
python eval_entity_linking.py --model pubmedbert --dataset ncbi
python eval_entity_linking.py --model pubmedbert --dataset bc5cdr_d
python eval_entity_linking.py --model pubmedbert --dataset bc5cdr_c
```

**Results** are saved automatically to `results/<model_name>/` as JSON files and figures. A leaderboard CSV is generated at `results/leaderboard.csv` after each run.

---

## Adding a new model

See [`CONTRIBUTING_MODELS.md`](CONTRIBUTING_MODELS.md) for the full guide including folder structure, required files, a complete Word2Vec example, a complete Transformer example, and a PR checklist.

The short version — your model must implement two methods:

```python
class YourModel(BaseEmbedder):
    def load(self, model_path: str) -> None: ...
    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...
```

## Team

| Role | Responsibility |
|------|---------------|
| Dawood | Benchmark runner — datasets, evaluation scripts, results |
| Team | Model architecture, training, pushing weights to `models/` |

---

## Notes

- All entity linking evaluations search the full CTD KB (~13k disease terms, ~10k chemical terms). Published numbers typically use a corpus-specific candidate KB which gives higher Acc@1 — keep this in mind when comparing against papers.
