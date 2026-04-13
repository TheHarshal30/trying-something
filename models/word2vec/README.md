# Word2Vec Scratch Baseline

This folder contains the main scratch Word2Vec baseline used in the project.

The baseline now supports:

- standard skip-gram training
- optional TF-IDF-weighted sentence pooling during inference

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

The trained outputs are saved to:

```bash
models/word2vec/weights/word2vec.bin
```

and, unless disabled:

```bash
models/word2vec/weights/tfidf_idf.json
```

## Recommended first baseline settings

- `vector_size=300`
- `window=10`
- `min_count=5`
- `sg=1` for skip-gram
- `negative=10`
- `epochs=10`

These are reasonable first-pass values for a domain-specific static embedding baseline.

## TF-IDF weighted pooling

Previously, a sentence was embedded by simple averaging:

```text
mean(word vectors)
```

Now the embedder can also use TF-IDF weighting:

```text
sum(idf(token) * vector(token)) / sum(idf(token))
```

Why this helps:

- common low-information words get downweighted
- rarer biomedical terms get more influence
- sentence similarity and retrieval can improve without changing the underlying Word2Vec vectors

Example:

```text
type 2 diabetes treatment
```

With TF-IDF weighting, words like `diabetes` and `treatment` matter more than generic words that appear in many documents.

If you want a pure no-TF-IDF ablation:

```bash
python models/word2vec/train.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/word2vec/weights \
  --disable_tfidf
```

## How this connects to evaluation

The benchmark loader in `models/word2vec/model.py` reads:

```bash
models/word2vec/weights/word2vec.bin
```

and automatically uses `tfidf_idf.json` if it is present.

So once training finishes, the model can be benchmarked against:

- entity linking
- STS
- NLI

Example:

```bash
python evaluation/run_all.py --model word2vec --task all
```
