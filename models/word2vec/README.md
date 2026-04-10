# Word2Vec Scratch Baseline

This folder now contains a scratch-training baseline for your course project.

## Recommended training dataset

Use **PubMed abstracts** as the first corpus.

Why this dataset:

- biomedical domain, so it matches the benchmark tasks
- large enough to train useful static embeddings
- much more appropriate than generic text like Wikipedia for this project
- easy to justify academically as a domain-specific unsupervised pretraining corpus

## Expected workflow

### 1. Put raw biomedical text on disk

Recommended location:

```bash
training_data/pubmed/raw/
```

Recommended contents:

- PubMed XML files
- PubMed XML.GZ files

The prep script also accepts plain `.txt` files where each line is one document.

### 2. Prepare the corpus

This extracts abstracts, lowercases them, keeps alphanumeric biomedical tokens, and writes:

```bash
training_data/pubmed/processed/pubmed_abstracts.txt
```

Run:

```bash
python models/word2vec/prepare_pubmed.py \
  --input_dir training_data/pubmed/raw \
  --output_file training_data/pubmed/processed/pubmed_abstracts.txt
```

### 3. Train Word2Vec from scratch

Run:

```bash
python models/word2vec/train.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/word2vec/weights \
  --vector_size 300 \
  --window 10 \
  --min_count 5 \
  --epochs 10 \
  --workers 8
```

The trained vectors are saved to:

```bash
models/word2vec/weights/word2vec.bin
```

## Recommended first baseline settings

- `vector_size=300`
- `window=10`
- `min_count=5`
- `sg=1` for skip-gram
- `negative=10`
- `epochs=10`

These are reasonable first-pass values for a domain-specific static embedding baseline.

## How this connects to evaluation

The benchmark loader in `models/word2vec/model.py` reads:

```bash
models/word2vec/weights/word2vec.bin
```

So once training finishes, we can wire this model into the evaluator registry and benchmark it against the downstream tasks.
