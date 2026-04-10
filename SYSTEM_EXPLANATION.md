# System Explanation

## What This System Is

This repository is a benchmark runner for **medical text embedding models**.

Its job is not to train a model from scratch during evaluation. Its job is to:

1. load an embedding model
2. load one or more benchmark datasets
3. convert text into vectors with that model
4. score the vectors on downstream tasks
5. save metrics and plots
6. aggregate results into a leaderboard

In short, it answers this question:

**"How useful are this model's embeddings for biomedical NLP tasks?"**

---

## What The System Evaluates

The system evaluates a model on three task families:

1. **Entity Linking**
   It checks whether an embedding for a mention like `"diabetes mellitus"` is closest to the correct disease or chemical entry in a knowledge base.

2. **Semantic Textual Similarity**
   It checks whether the cosine similarity between two sentence embeddings matches human similarity judgments.

3. **Natural Language Inference**
   It checks whether sentence-pair embeddings contain enough information for a classifier to tell entailment vs contradiction.

These are implemented in:

- [`eval_entity_linking.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_entity_linking.py)
- [`eval_sts.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_sts.py)
- [`eval_nli.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_nli.py)

---

## High-Level Flow

When you run:

```bash
python evaluation/run_all.py --model pubmedbert --task all
```

the system follows this pipeline:

1. Parse CLI arguments in [`run_all.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/run_all.py)
2. Ensure required assets exist through [`assets.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/assets.py)
3. Load the requested embedding model
4. Run the selected evaluation tasks
5. Save per-task JSON results and figures
6. Rebuild `results/leaderboard.csv`

So `run_all.py` is the orchestration layer, while the task files do the scoring work.

---

## Core Design Idea

The system is built around a very simple abstraction:

**every model only needs to provide `load()` and `encode()`**

That contract is defined in [`base_embedder.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/base_embedder.py).

The evaluation scripts do not care whether the model is:

- PubMedBERT
- SapBERT
- BioBERT
- MiniLM
- Word2Vec
- a custom transformer

They only assume:

- the model can be loaded
- the model can turn `list[str]` into a numeric matrix

That separation is what makes the benchmark reusable.

---

## How Model Loading Works

The model registry lives in [`run_all.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/run_all.py) inside `load_embedder()`.

For pretrained transformer models, the system currently reuses [`pubmedbert_embedder.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/pubmedbert_embedder.py).

Despite the filename, that class is now being used as a generic Hugging Face encoder for several baselines:

- `pubmedbert`
- `sapbert`
- `biobert`
- `minilm`

### What `PubMedBERTEmbedder` does

1. decides whether to run on `mps`, `cuda`, or `cpu`
2. loads tokenizer and model from a local directory
3. tokenizes text in batches
4. runs the transformer
5. takes the `CLS` token from `last_hidden_state`
6. returns a NumPy matrix of embeddings

That means all tasks ultimately consume a matrix shaped roughly like:

```python
(num_texts, embedding_dim)
```

---

## How Auto-Download Works

The download layer lives in [`assets.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/assets.py).

Its job is to make the repo runnable on a clean machine.

### What it downloads

It can fetch:

- baseline models from Hugging Face
- NCBI Disease files
- BC5CDR files
- BIOSSES
- NLI4CT
- CTD disease and chemical lookup tables

### Main functions

- `ensure_model(model_name)`
- `ensure_entity_linking_assets(dataset)`
- `ensure_sts_assets(dataset)`
- `ensure_nli_assets(dataset)`
- `ensure_task_assets(task, model_name)`

### Runtime behavior

Before a task runs, the system checks whether the needed files already exist.

If they do:
- it reuses them

If they do not:
- it downloads them
- extracts or reformats them
- writes them into the project layout expected by the older evaluation code

This means the benchmark code can stay mostly unchanged while the repo becomes self-bootstrapping.

---

## How Data Is Organized

### Raw task data

Stored under:

- [`data/raw`](/home/harshal/nlp%20project%20/medical-entity-linking/data/raw)

Examples:

- `data/raw/ncbi_disease/...`
- `data/raw/bc5cdr/...`
- `data/raw/biosses/...`
- `data/raw/nli4ct/...`

### Lookup / knowledge-base data

Stored under:

- [`data/lookups/mesh`](/home/harshal/nlp%20project%20/medical-entity-linking/data/lookups/mesh)

Examples:

- `CTD_diseases.tsv`
- `CTD_chemicals.tsv`

### Cached downloaded archives

Stored under:

- [`data/_downloads`](/home/harshal/nlp%20project%20/medical-entity-linking/data/_downloads)

This avoids redownloading large archives when the extracted version already exists.

### Models

Stored under:

- [`models`](/home/harshal/nlp%20project%20/medical-entity-linking/models)

Examples:

- `models/pubmedbert-local`
- `models/sapbert-local`
- `models/biobert-local`
- `models/minilm-local`

---

## How Each Task Works

## 1. Entity Linking

Implemented in [`eval_entity_linking.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_entity_linking.py).

### Goal

Map a mention to the correct disease or chemical concept.

Example:

- mention: `"lung carcinoma"`
- target concept: correct MeSH / CTD entry

### Inputs

- mention texts from NCBI or BC5CDR
- canonical knowledge-base names from CTD

### Process

1. Load mentions from the dataset
2. Load the CTD knowledge base
3. Encode all KB names into embeddings
4. Encode all mentions into embeddings
5. Compute cosine similarity between each mention embedding and all KB embeddings
6. Rank candidates
7. Check whether the gold concept appears in top 1, top 5, top 10

### Metrics

- `acc@1`
- `acc@5`
- `acc@10`
- `mrr`

### Important detail

The system uses the **full CTD KB**, not a small corpus-specific candidate set. That makes the task more realistic, but usually harder.

---

## 2. Semantic Textual Similarity

Implemented in [`eval_sts.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_sts.py).

### Goal

Check whether two sentences that humans judge as similar also have similar embeddings.

### Process

1. Load sentence pairs from BIOSSES
2. Encode sentence 1 for all pairs
3. Encode sentence 2 for all pairs
4. Compute cosine similarity for each pair
5. Compare model scores to human labels

### Metrics

- `pearson_r`
- `spearman_r`

This task answers:

**"Do the embedding distances line up with semantic similarity as judged by humans?"**

---

## 3. Natural Language Inference

Implemented in [`eval_nli.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_nli.py).

### Goal

Check whether embeddings contain enough signal to distinguish entailment from contradiction.

### Process

1. Load NLI4CT sentence pairs
2. Encode both sentences
3. Build pair features using:

```python
[A, B, A - B, A * B]
```

where:

- `A` is embedding of sentence 1
- `B` is embedding of sentence 2

4. Train a logistic regression classifier on top of those features
5. Evaluate on held-out examples

### Metrics

- `accuracy`
- `macro_f1`
- `majority_baseline`

This task asks:

**"Even if the embeddings are frozen, do they preserve enough information for a lightweight classifier to solve NLI?"**

---

## Why The NLI Task Uses A Classifier

Unlike STS or entity linking, NLI is not naturally solved by raw cosine similarity alone.

The system therefore does not compare embeddings directly. Instead, it uses embeddings as features for a shallow classifier.

That design choice means the benchmark is evaluating:

- the **quality of the embeddings**
- not a fully fine-tuned NLI model

So the classifier is intentionally simple. It acts as a probe.

---

## How Results Are Produced

Each task writes results to:

- [`results/<model_name>`](/home/harshal/nlp%20project%20/medical-entity-linking/results)

Typical outputs are:

- task JSON files
- task figures

Examples:

- `results/minilm/sts_biosses.json`
- `results/minilm/figures/biosses_results.png`

After task execution, [`run_all.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/run_all.py) calls `build_leaderboard()`, which scans result JSON files and rebuilds:

- [`results/leaderboard.csv`](/home/harshal/nlp%20project%20/medical-entity-linking/results/leaderboard.csv)

That file is the system’s cross-model summary table.

---

## What The Setup Script Does

[`setup_assets.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/setup_assets.py) is not an evaluator.

It is a convenience entry point for preparing the environment in advance.

For example:

```bash
python evaluation/setup_assets.py --task all --model pubmedbert
```

This downloads the assets needed for that task/model combination and exits.

That is useful when you want to:

- prewarm the machine
- avoid waiting during the first benchmark run
- verify that downloads succeed before heavy evaluation starts

---

## How The New Download Layer Fits Into The Older Code

The original project structure assumed data and models had already been placed manually.

The new asset layer does **not** replace the evaluation logic.

Instead, it sits in front of it:

1. old evaluator asks for dataset / model
2. asset layer ensures the files exist
3. evaluator proceeds exactly as before

That is why the system still feels like the original codebase, but now works on a fresh clone.

---

## Main Components And Their Roles

### [`run_all.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/run_all.py)

Top-level orchestrator.

Responsibilities:

- parse CLI args
- prepare assets
- load embedder
- run tasks
- build leaderboard

### [`base_embedder.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/base_embedder.py)

Interface contract for all embedding models.

### [`pubmedbert_embedder.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/pubmedbert_embedder.py)

Transformer-based encoder implementation used for current pretrained baselines.

### [`assets.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/assets.py)

Download and asset normalization layer.

### Task evaluators

- [`eval_entity_linking.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_entity_linking.py)
- [`eval_sts.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_sts.py)
- [`eval_nli.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_nli.py)

These implement the actual benchmarking logic.

---

## One Concrete Example

Suppose you run:

```bash
python evaluation/run_all.py --model minilm --task sts
```

What happens:

1. `run_all.py` reads `model=minilm`, `task=sts`
2. `ensure_task_assets(task='sts', model_name='minilm')` runs
3. `ensure_model('minilm')` downloads `sentence-transformers/all-MiniLM-L6-v2` if missing
4. `ensure_sts_assets()` downloads / writes BIOSSES JSONL files if missing
5. `load_embedder('minilm')` creates the transformer embedder
6. `eval_sts.evaluate(...)` loads BIOSSES test pairs
7. the model encodes both sentence columns
8. cosine similarity is computed pairwise
9. Pearson and Spearman correlations are calculated
10. result JSON and figure are saved
11. leaderboard is rebuilt

That is the system in miniature.

---

## What This System Is Good For

It is useful when you want to compare embedding models on:

- concept normalization behavior
- semantic similarity quality
- frozen-feature transfer to sentence-pair classification

It is especially useful for:

- comparing multiple biomedical encoders quickly
- evaluating a newly trained model against strong baselines
- generating reproducible benchmark artifacts

---

## Limits Of The Current System

A few design choices are important to understand:

- it evaluates embeddings, not full end-task fine-tuned models
- entity linking depends on nearest-neighbor similarity over KB names
- STS uses cosine similarity only
- NLI uses a shallow classifier probe, not end-to-end transformer fine-tuning
- some downloaded datasets are adapted into the repo’s expected local format

So the benchmark is best understood as an **embedding evaluation suite**, not a full training platform.

---

## Bottom Line

This system is a modular benchmark pipeline for biomedical embeddings.

It works by:

1. standardizing model access through an `encode()` interface
2. preparing benchmark assets automatically
3. running task-specific evaluations on top of those embeddings
4. saving comparable outputs across models

The new auto-download layer makes the system self-contained, but the core idea remains the same:

**load embeddings, test them on biomedical tasks, and compare models with shared metrics.**
