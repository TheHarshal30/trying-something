# Asset Download Wiring Notes

## Purpose

This document explains the work done to make the repository usable without manually placing datasets, MeSH lookup files, and baseline model folders into the project ahead of time.

Before these changes, the evaluation scripts assumed that:

- dataset files already existed under `data/raw/...`
- CTD lookup tables already existed under `data/lookups/mesh/...`
- local Hugging Face model snapshots already existed under `models/*-local`

That meant the repo cloned successfully, but could not run end to end on a fresh machine.

## What Was Added

### 1. Central asset manager

File:
- `evaluation/assets.py`

This module now handles first-run downloads for:

- pretrained baseline models
- NCBI Disease corpus
- BC5CDR corpus
- BIOSSES
- NLI4CT
- CTD disease and chemical lookup tables

It also standardizes where those assets are stored:

- `data/raw/...`
- `data/lookups/mesh/...`
- `models/*-local`
- `data/_downloads/...` for cached archives

### 2. Manual setup script

File:
- `evaluation/setup_assets.py`

This provides a simple CLI for bootstrapping assets before running benchmarks.

Examples:

```bash
python evaluation/setup_assets.py --task all --model pubmedbert
python evaluation/setup_assets.py --task sts --model minilm
python evaluation/setup_assets.py --task entity_linking
```

### 3. Automatic model fallback

File:
- `evaluation/pubmedbert_embedder.py`

If the requested local model path does not exist, the embedder now downloads the configured model snapshot automatically from Hugging Face and then loads it from the local repo cache directory.

Configured model mappings currently include:

- `pubmedbert`
- `sapbert`
- `biobert`
- `minilm`

### 4. Auto-download hooks in evaluation scripts

Files:

- `evaluation/run_all.py`
- `evaluation/eval_entity_linking.py`
- `evaluation/eval_sts.py`
- `evaluation/eval_nli.py`

Each task now ensures its required assets exist before evaluation starts.

`run_all.py` also supports:

```bash
python evaluation/run_all.py --model pubmedbert --task all --download_only
```

This downloads everything needed for that model/task combination and exits without running evaluation.

## Dataset Handling Details

### NCBI Disease

Downloaded from the official NCBI Disease corpus zip files and extracted into:

- `data/raw/ncbi_disease/NCBItrainset_corpus.txt`
- `data/raw/ncbi_disease/NCBIdevelopset_corpus.txt`
- `data/raw/ncbi_disease/NCBItestset_corpus.txt`

This matches the layout already expected by the original loader.

### BC5CDR

Downloaded from the BigBio-hosted `CDR_Data.zip` archive and extracted into:

- `data/raw/bc5cdr/CDR_Data/...`

The original repo expected preprocessed JSONL files, but none were present. Instead of introducing a separate conversion step, the entity-linking loader was patched to read directly from the downloaded BioC XML files when JSONL files are missing.

This keeps the existing code path working while reducing setup requirements.

### BIOSSES

Loaded from Hugging Face dataset `biosses`.

The repository expects local JSONL files, so the downloader writes:

- `data/raw/biosses/train.jsonl`
- `data/raw/biosses/validation.jsonl`
- `data/raw/biosses/test.jsonl`

Because the hosted version is small and does not ship with the same split layout as the repo expects, a deterministic split is created locally.

### NLI4CT

Loaded from Hugging Face dataset `tasksource/nli4ct`.

The repository expects:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

The hosted dataset exposed `train` and `validation` splits during validation. To preserve compatibility with the repo’s existing loader, the setup step writes:

- `train.jsonl` from hosted `train`
- `validation.jsonl` from hosted `validation`
- `test.jsonl` from hosted `validation`

This is a compatibility choice, not a benchmark-perfect reconstruction of the original competition split.

The downloader also reformats each row into the structure expected by the existing evaluator:

- `sentence1`
- `sentence2`
- `gold_label`

### CTD MeSH lookup files

Downloaded and decompressed into:

- `data/lookups/mesh/CTD_diseases.tsv`
- `data/lookups/mesh/CTD_chemicals.tsv`

These are the knowledge base files used by entity linking evaluation.

## Dependency Updates

File:
- `requirements.txt`

Added packages required by the new download and parsing flow:

- `datasets`
- `requests`
- `bioc`

The original checked-in `requiremnts.txt` file was empty and misspelled, so a working `requirements.txt` was added separately.

## Verification Performed

The following validation steps were completed:

### Syntax / import validation

```bash
python -m compileall evaluation
```

This passed successfully.

### Asset bootstrap validation

```bash
python evaluation/setup_assets.py --task sts --model minilm
python evaluation/setup_assets.py --task entity_linking
python evaluation/setup_assets.py --task nli
```

These completed successfully and materialized the expected local assets.

### Functional smoke test

```bash
python evaluation/run_all.py --model minilm --task sts --batch_size 8
```

This completed successfully, produced STS results, and wrote:

- `results/minilm/sts_biosses.json`
- `results/minilm/figures/biosses_results.png`

### Loader checks

The downloaded corpora were also validated by directly loading them through the existing evaluation loaders:

- NCBI test mentions loaded successfully
- BC5CDR disease mentions loaded successfully
- BC5CDR chemical mentions loaded successfully
- NLI4CT train/validation/test JSONL files loaded successfully

## Known Caveats

### 1. NLI4CT test split

The current compatibility layer reuses validation data as `test.jsonl` because the hosted source used during implementation did not expose a separate test split.

### 2. BIOSSES split reconstruction

The repo now creates local train/validation/test JSONL files from a small hosted dataset using a deterministic split. This is practical for repo usability, but it may not exactly match the split assumptions of the original project authors.

### 3. Hugging Face authentication

Downloads work without authentication, but the environment may print warnings about unauthenticated Hugging Face requests. Setting `HF_TOKEN` would improve rate limits and reliability.

### 4. Existing result files

Running validation updated generated outputs such as:

- `results/leaderboard.csv`
- `results/minilm/...`

These are expected side effects of the benchmark smoke test.

## Recommended Usage

Fresh setup:

```bash
cd "/home/harshal/nlp project /medical-entity-linking"
source .venv/bin/activate
python evaluation/setup_assets.py --task all --model pubmedbert
```

Download only through the main runner:

```bash
python evaluation/run_all.py --model pubmedbert --task all --download_only
```

Run a benchmark:

```bash
python evaluation/run_all.py --model pubmedbert --task all
python evaluation/run_all.py --model minilm --task sts
```

## Files Changed

- `evaluation/assets.py`
- `evaluation/setup_assets.py`
- `evaluation/pubmedbert_embedder.py`
- `evaluation/run_all.py`
- `evaluation/eval_entity_linking.py`
- `evaluation/eval_sts.py`
- `evaluation/eval_nli.py`
- `requirements.txt`

## Outcome

The repository now supports a first-run workflow on a clean machine:

1. clone the repo
2. create / activate the virtualenv
3. install dependencies
4. download assets automatically
5. run evaluations without manually arranging model and dataset files
