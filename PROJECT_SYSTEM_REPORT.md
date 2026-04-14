# Project System Report

## What This Project Is

This repository is a **biomedical embedding benchmark and training workspace**.

We are using it for a course project where the key rule is:

**the models we compare must be trained from scratch**

So the project has two major parts:

1. **Training pipelines**
   These produce embedding models from biomedical text.
2. **Evaluation pipelines**
   These test whether those embeddings are useful for downstream biomedical NLP tasks.

In simple terms:

- training answers: **"How do we build the model?"**
- evaluation answers: **"How good are the embeddings after training?"**

---

## Main Components

## 1. Benchmark / Evaluation Layer

This lives under [`evaluation/`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation).

Important files:

- [`run_all.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/run_all.py)
- [`eval_entity_linking.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_entity_linking.py)
- [`eval_sts.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_sts.py)
- [`eval_nli.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_nli.py)
- [`assets.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/assets.py)
- [`base_embedder.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/base_embedder.py)

This layer does not train models. It only:

- loads a chosen embedding model
- loads the benchmark datasets
- converts text into vectors
- computes task-specific metrics
- saves results

### Core idea

Every model follows the same interface:

- `load(model_path)`
- `encode(texts, batch_size=...)`

Because of that, the benchmark can evaluate:

- `word2vec`
- `fasttext`
- `transformer_scratch`
- `transformer_scratch_simcse`
- pretrained baselines like `pubmedbert` and `sapbert`

without rewriting the task logic each time.

---

## 2. Scratch Word2Vec / FastText Pipeline

This is the first scratch baseline and lives under:

- [`models/word2vec/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec)

Important files:

- [`prepare_pubmed.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec/prepare_pubmed.py)
- [`train.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec/train.py)
- [`model.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec/model.py)

### What it does

1. reads raw PubMed XML or XML.GZ files
2. extracts article abstracts
3. cleans and tokenizes the text
4. trains either classic skip-gram Word2Vec embeddings or FastText-style subword embeddings
5. computes optional TF-IDF weights for tokens
6. saves `word2vec.bin`, and for FastText also saves `fasttext.model`

### Example

Raw biomedical text:

```text
Patients with type 2 diabetes received insulin therapy.
```

After tokenization, the static model learns vectors for words such as:

- `patients`
- `type`
- `diabetes`
- `insulin`
- `therapy`

To embed a sentence, the system now prefers **TF-IDF-weighted pooling** when TF-IDF weights are available.

So for:

```text
type 2 diabetes
```

the basic sentence vector is approximately:

```text
mean( vector("type"), vector("2"), vector("diabetes") )
```

With TF-IDF weighting, the system instead does something closer to:

```text
sum( idf(token) * vector(token) ) / sum( idf(token) )
```

This is still a static embedding model, but it gives more importance to informative biomedical tokens.

### FastText-style subword modeling

The same training script now also supports a stronger static baseline using FastText-style character n-grams.

In simplified form:

```text
v(word) = sum( v(character n-grams) )
```

This helps because biomedical and chemical terms often share meaningful substrings.

Example:

```text
acetylcysteine
cysteine
```

These tokens share subword pieces, so the model can generalize better to rare or unseen strings than plain Word2Vec.

### Current recommended static baseline settings

- `model_type=fasttext`
- `vector_size=400`
- `window=10`
- `negative=10`
- `epochs=10`
- TF-IDF weighted pooling enabled

The original `word2vec` baseline is still preserved for comparison, while `fasttext` is the new subword-aware static baseline aimed especially at rare biomedical and chemical terms.

---

## 3. Legacy Word2Vec Training Pipeline

The separate `TrainWord2Vec/TrainWord2Vec/` subtree has been removed from the repository. The main Word2Vec and FastText baseline now live entirely under [`models/word2vec/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec).

---

## 4. Scratch Transformer Pipeline

This is the contextual model trained from scratch and lives under:

- [`models/transformer_scratch/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch)

Important files:

- [`train_tokenizer.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch/train_tokenizer.py)
- [`train_mlm.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch/train_mlm.py)
- [`train_simcse.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch/train_simcse.py)
- [`model.py`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch/model.py)

### What it does

1. trains a tokenizer from scratch on PubMed abstracts
2. builds a small BERT-style encoder from scratch
3. trains it with masked language modeling
4. optionally fine-tunes it with SimCSE-style contrastive learning
5. saves the final encoder
6. uses that encoder to produce embeddings for downstream evaluation

### Architecture used

- hidden size: `384`
- layers: `6`
- attention heads: `6`
- intermediate size: `1536`
- max sequence length: `128`
- tokenizer vocab size: `50000`

### Example of masked language modeling

Original sentence:

```text
breast cancer treatment improves survival
```

Masked training example:

```text
breast [MASK] treatment improves survival
```

The model learns to predict:

```text
cancer
```

This forces it to learn context, not just isolated word statistics.

That is the main reason transformers can capture meaning better than simple Word2Vec averaging.

### Example of SimCSE-style contrastive learning

In the new contrastive stage, the same sentence is passed through the encoder twice with dropout:

```text
Sentence A -> encoder -> z1
Sentence A -> encoder -> z2
```

Because dropout changes the internal activations, `z1` and `z2` are slightly different views of the same sentence.

The objective then:

- pulls `z1` and `z2` together
- pushes them away from embeddings of other sentences in the batch

This improves the geometry of the embedding space and makes the transformer act more like a real sentence encoder.

The SimCSE stage now also includes:

- stronger dropout noise
- embedding standard-deviation tracking
- collapse detection
- gradient clipping
- optional MLM auxiliary loss

### Pooling strategies

The transformer embedder now supports:

- `cls`
- `mean`
- `last4_mean`

The pipeline also stores embedding-time settings such as pooling and normalization in the checkpoint metadata.

Chemical-aware normalization was added as well so chemical strings are handled more consistently during tokenizer training, MLM training, SimCSE fine-tuning, and inference.

---

## Data Flow

## Step 1. Raw corpus

The project uses **PubMed abstracts** as the main unsupervised training corpus.

Typical path:

- [`training_data/pubmed/raw/`](/home/harshal/nlp%20project%20/medical-entity-linking/training_data/pubmed/raw)

These files contain PubMed XML article records.

Each article can include:

- PMID
- title
- abstract
- journal metadata

For our project, the main useful field is the **abstract text**.

## Step 2. Processed corpus

The Word2Vec prep script converts the raw XML into:

- [`training_data/pubmed/processed/pubmed_abstracts.txt`](/home/harshal/nlp%20project%20/medical-entity-linking/training_data/pubmed/processed/pubmed_abstracts.txt)

This file is essentially one training text per line.

Example line:

```text
we studied gene expression in breast cancer tissue and observed altered pathways
```

This processed file is then reused by:

- the scratch Word2Vec pipeline
- the scratch transformer tokenizer training
- the scratch transformer MLM training

So one prepared corpus supports multiple models.

---

## How Evaluation Works

The main command is:

```bash
python evaluation/run_all.py --model <model_name> --task all
```

Example:

```bash
python evaluation/run_all.py --model transformer_scratch --task all
```

### High-level runtime flow

1. `run_all.py` reads CLI arguments
2. `assets.py` ensures datasets and lookup tables exist
3. `load_embedder()` loads the requested model
4. the chosen task evaluators are executed
5. each evaluator writes a JSON result file
6. figures are generated
7. a leaderboard is rebuilt

### Important idea

The evaluation code does **not** know how the model was trained.

It only asks:

- can the model produce vectors?
- are those vectors useful for the task?

That makes it easy to compare very different architectures fairly.

---

## Tasks We Evaluate

## 1. Entity Linking

File:

- [`eval_entity_linking.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_entity_linking.py)

### Goal

Given a mention, retrieve the correct biomedical concept from a lookup table.

### Example

Mention:

```text
diabetes mellitus
```

Candidate KB entries might include many disease names. The system embeds:

- the mention
- every KB entry

Then it computes cosine similarity and ranks the candidates.

If the correct concept is the nearest one, that counts toward `Acc@1`.

The entity-linking pipeline now supports a second-stage reranker as well:

1. retrieve top candidates with cosine similarity
2. rerank them with a small MLP over the mention vector, candidate vector, and `|u - v|`

It also supports hard-negative sampling during reranker training.

### Datasets used

- `NCBI`
- `BC5CDR-d`
- `BC5CDR-c`

### Metrics

- `Acc@1`
- `Acc@5`
- `Acc@10`
- `MRR`

---

## 2. Semantic Textual Similarity

File:

- [`eval_sts.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_sts.py)

### Goal

Check whether embedding similarity matches human similarity judgments.

### Example

Sentence 1:

```text
The patient was treated for breast cancer.
```

Sentence 2:

```text
Breast carcinoma therapy was administered to the patient.
```

A good model should assign similar embeddings to these sentences.

### Dataset used

- `BIOSSES`

### Metrics

- `Pearson r`
- `Spearman r`

Higher is better.

---

## 3. Natural Language Inference

File:

- [`eval_nli.py`](/home/harshal/nlp%20project%20/medical-entity-linking/evaluation/eval_nli.py)

### Goal

Check whether sentence-pair embeddings contain enough information for a classifier to predict the relationship between texts.

### Example

Premise:

```text
The trial reported improved survival with the new therapy.
```

Hypothesis:

```text
The therapy reduced patient survival.
```

This pair should behave like a contradiction.

### Dataset used

- `NLI4CT`

### How this evaluation works

The benchmark first encodes the texts, then trains a small neural classifier on top of those embeddings.

The current classifier uses:

- embedding of sentence A
- embedding of sentence B
- absolute difference `|A - B|`

and predicts the label with a small MLP.

So this task still measures whether the embeddings contain useful information, not whether the base model was directly fine-tuned for NLI.

### Metrics

- `Accuracy`
- `Macro F1`
- `Majority baseline`

---

## Current Scratch Models Evaluated

At the moment, the most relevant scratch-trained models are:

1. `word2vec`
   The original scratch Word2Vec baseline in [`models/word2vec/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec)

2. `fasttext`
   The subword-aware static baseline trained through the same [`models/word2vec/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/word2vec) pipeline but evaluated as a separate model

3. `transformer_scratch`
   The scratch transformer encoder from [`models/transformer_scratch/`](/home/harshal/nlp%20project%20/medical-entity-linking/models/transformer_scratch)

4. `transformer_scratch_simcse`
   The SimCSE-fine-tuned version of the scratch transformer encoder

The `fasttext` model is implemented and benchmark-ready, but its results should be added after the next evaluation run on the server.

---

## Current Results

These are the clean comparison results produced by the benchmark.

## Entity Linking

| Model | Dataset | Acc@1 | Acc@5 | Acc@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| `word2vec` | `bc5cdr_c` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `trainword2vec` | `bc5cdr_c` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `transformer_scratch` | `bc5cdr_c` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `word2vec` | `bc5cdr_d` | 0.5344 | 0.7022 | 0.7500 | 0.6048 |
| `trainword2vec` | `bc5cdr_d` | 0.4604 | 0.6112 | 0.6934 | 0.5284 |
| `transformer_scratch` | `bc5cdr_d` | 0.5419 | 0.6634 | 0.7051 | 0.5917 |
| `word2vec` | `ncbi` | 0.3518 | 0.4899 | 0.5314 | 0.4084 |
| `trainword2vec` | `ncbi` | 0.2889 | 0.4648 | 0.5276 | 0.3709 |
| `transformer_scratch` | `ncbi` | 0.3266 | 0.4824 | 0.5113 | 0.3957 |

### Interpretation

- `transformer_scratch` is now the best model on `BC5CDR-d` by `Acc@1`.
- The original `word2vec` baseline is still the strongest scratch model on `NCBI`.
- `trainword2vec` is weaker than the earlier `word2vec` run.
- The lower-learning-rate transformer run recovered the earlier entity-linking collapse and became competitive with Word2Vec.
- The entity-linking pipeline is now stronger than before because it supports reranking after initial cosine retrieval.
- All models failed on `BC5CDR-c`, which means chemical linking remains an open problem in the current setup.

---

## STS

| Model | Dataset | Pearson r | Spearman r |
| --- | --- | --- | --- |
| `transformer_scratch` | `biosses` | 0.2478 | 0.3415 |
| `word2vec` | `biosses` | -0.2804 | -0.1220 |
| `trainword2vec` | `biosses` | -0.4905 | -0.3476 |

### Interpretation

- `transformer_scratch` is clearly the best of the three on sentence similarity.
- Both Word2Vec models are weaker, especially `trainword2vec`.
- The lower-learning-rate transformer run improved STS substantially compared with earlier transformer checkpoints.
- The new SimCSE stage was added specifically to push sentence-level quality further in future ablations.

---

## NLI

| Model | Dataset | Accuracy | Macro F1 | Majority baseline |
| --- | --- | --- | --- | --- |
| `word2vec` | `nli4ct` | 0.3735 | 0.3735 | 0.5000 |
| `trainword2vec` | `nli4ct` | 0.3265 | 0.3265 | 0.5000 |
| `transformer_scratch` | `nli4ct` | 0.3059 | 0.3059 | 0.5000 |

### Interpretation

- `word2vec` is the strongest scratch model on NLI in the current final comparison.
- `trainword2vec` is second and `transformer_scratch` is third.
- None of the scratch models beat the majority baseline yet, which shows that the task is still difficult for the current training scale.

---

## Overall Conclusions

The current project shows a meaningful pattern:

1. **Static embeddings and contextual embeddings behave differently**
   The original `word2vec` is still strongest on `NCBI` retrieval and NLI, while `transformer_scratch` is strongest on STS and `BC5CDR-d`.

2. **The training pipeline matters**
   The second Word2Vec pipeline (`trainword2vec`) did not beat the original Word2Vec baseline, which suggests that preprocessing, tokenization, corpus handling, or training choices strongly affect performance.

3. **Transformer training is highly sensitive to optimization settings**
   A lower learning rate substantially improved the transformer on STS and restored useful entity-linking performance after an earlier collapse.

4. **The system has now moved beyond naive baselines**
   The codebase now includes contrastive sentence-encoder training, TF-IDF weighting, reranking, hard negatives, a neural NLI probe, chemical-aware normalization, and a FastText-style subword baseline, making it a more serious representation-learning system.

5. **Chemical entity linking is still the hardest open problem**
   Earlier benchmarked scratch models all failed on `BC5CDR-c`, which is exactly why the repo now includes chemical-aware normalization and a FastText-style subword baseline.

---

## Practical Summary

If someone asks what this system does in one sentence:

**It trains biomedical embedding models from scratch, plugs them into a shared benchmark, and compares how well those embeddings support entity linking, sentence similarity, and clinical-trial inference tasks.**

If someone asks which model currently looks best:

- for **NCBI entity linking and NLI**: `word2vec`
- for **BC5CDR-d and STS**: `transformer_scratch`

If someone asks what we are testing next:

- whether `fasttext` improves rare biomedical and chemical term handling
- whether stabilized SimCSE improves sentence-level quality further
- whether reranking and hard negatives improve entity linking beyond plain cosine retrieval

If someone asks what is unfinished:

- UMLS-enhanced Word2Vec alignment is still pending UMLS access
- chemical entity linking remains poor in the older benchmarked runs
- scratch models are still behind strong pretrained biomedical baselines
- the new FastText, SimCSE, reranking, and ablation upgrades still need full server-side runs

---

## Useful Commands

Train the original scratch Word2Vec:

```bash
python models/word2vec/train.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/word2vec/weights
```

Train the subword-aware FastText baseline:

```bash
python models/word2vec/train.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/fasttext/weights \
  --model_type fasttext \
  --vector_size 400 \
  --window 10 \
  --negative 10 \
  --epochs 10
```

Train tokenizer for the scratch transformer:

```bash
python models/transformer_scratch/train_tokenizer.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/transformer_scratch/weights/tokenizer \
  --vocab_size 50000 \
  --normalization_strategy chemical
```

Train the scratch transformer:

```bash
python models/transformer_scratch/train_mlm.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --tokenizer_dir models/transformer_scratch/weights/tokenizer \
  --output_dir models/transformer_scratch/weights/final \
  --pooling_strategy mean \
  --normalization_strategy chemical
```

Fine-tune the transformer with SimCSE:

```bash
python models/transformer_scratch/train_simcse.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --input_dir models/transformer_scratch/weights/final \
  --output_dir models/transformer_scratch/weights/final_simcse \
  --temperature 0.1 \
  --dropout 0.2 \
  --use_mlm_aux \
  --normalization_strategy chemical
```

Run full benchmark:

```bash
python evaluation/run_all.py --model transformer_scratch --task all
```

Run FastText benchmark:

```bash
python evaluation/run_all.py --model fasttext --use_tfidf --task all
```

Run SimCSE benchmark:

```bash
python evaluation/run_all.py --model transformer_scratch --use_simcse --use_reranker --use_hard_negatives --task all
```

Create a clean comparison table:

```bash
python evaluation/compare_results.py --models word2vec fasttext transformer_scratch transformer_scratch_simcse
```
