# Training Pipeline

End-to-end instructions for producing the `word2vec` and `word2vec_umls` model weights.

---

## Prerequisites

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

---

## Step 0 — Data acquisition

### PubMed abstracts (for Word2Vec training)

```bash
mkdir -p data/pubmed
wget -r -nd -np -A "*.xml.gz" \
     https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ \
     -P data/pubmed/
```

> Full baseline: ~1100 files, ~35 GB compressed, ~4.5B tokens.
> For a quick smoke-test, download just one or two files.

### UMLS MRCONSO.RRF (for pair extraction)

1. Create a free UMLS account: https://uts.nlm.nih.gov/uts/signup-login
2. Download the latest UMLS Metathesaurus release
3. Extract and locate `META/MRCONSO.RRF`

---

## Step 1 — Train Word2Vec on PubMed

```bash
python 01_train_word2vec.py \
    --abstracts data/pubmed \
    --output    models/word2vec/weights/word2vec.bin \
    --dim       300 \
    --window    5 \
    --min_count 5 \
    --epochs    5 \
    --workers   12
```

**Smoke-test** (first 100k sentences only):
```bash
python 01_train_word2vec.py \
    --abstracts     data/pubmed \
    --max_sentences 100000 \
    --output        models/word2vec/weights/word2vec.bin
```

Expected output size: ~500 MB–1.5 GB (varies with corpus and min_count).

---

## Step 2 — Extract UMLS synonym pairs

```bash
python 02_extract_umls_pairs.py \
    --mrconso /path/to/META/MRCONSO.RRF \
    --vocab_bin models/word2vec/weights/word2vec.bin \
    --pairs_out data/umls_pairs.txt \
    --vocab_out models/word2vec_umls/weights/umls_vocab.json \
    --max_pairs_per_cui 10
```

The `--vocab_bin` argument filters out pairs where neither side has
any in-vocabulary token (they would produce zero vectors and waste training).

Outputs:
- `data/umls_pairs.txt` — tab-separated (anchor, positive) synonym pairs
- `models/word2vec_umls/weights/umls_vocab.json` — `{CUI: canonical_name}`

---

## Step 3 — NT-Xent alignment

```bash
python 03_align_ntxent.py \
    --w2v_bin    models/word2vec/weights/word2vec.bin \
    --pairs      data/umls_pairs.txt \
    --output     models/word2vec_umls/weights/word2vec_umls.bin \
    --proj_dim   256 \
    --temperature 0.07 \
    --batch_size  512 \
    --epochs      10 \
    --lr          3e-4
```

Alignment with `--freeze_embedding` (faster, trains only the projection head,
embeddings updated at extraction time via the head):
```bash
python 03_align_ntxent.py \
    ... \
    --freeze_embedding
```

> **Note:** The projection head is only used during training.  After alignment,
> the updated embedding table is extracted and saved directly as a Word2Vec
> `.bin` file, so the inference path in `model.py` is identical to the baseline.

---

## Output layout

After all three steps, the folder structure should match the contract in `CONTRIBUTING_MODELS.md`:

```
models/
├── word2vec/
│   ├── model.py
│   └── weights/
│       └── word2vec.bin          ← produced by step 1
│
└── word2vec_umls/
    ├── model.py
    └── weights/
        ├── word2vec_umls.bin     ← produced by step 3
        └── umls_vocab.json       ← produced by step 2
```

---

## Hyperparameter guidance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--dim` | 300 | Match Google News / BioWordVec for comparability |
| `--window` | 5 | Standard for biomedical text |
| `--min_count` | 5 | Lower → larger vocab but noisier vectors |
| `--temperature` | 0.07 | SimCLR default; lower = sharper negatives |
| `--batch_size` | 512 | Larger batches = more in-batch negatives = harder learning signal |
| `--proj_dim` | 256 | Projection head output; doesn't affect saved vector dim |
| `--freeze_embedding` | off | Turn on to preserve distributional semantics; turn off for full fine-tuning |

---

## Sanity check

Run this from the project root before submitting a PR:

```python
import sys
sys.path.append('evaluation/')

from models.word2vec.model import Word2VecEmbedder
from models.word2vec_umls.model import Word2VecUMLSEmbedder
import numpy as np

for Cls, path in [(Word2VecEmbedder, 'models/word2vec'),
                  (Word2VecUMLSEmbedder, 'models/word2vec_umls')]:
    emb = Cls()
    emb.load(path)
    texts = ['diabetes mellitus', 'heart failure',
             'Patients received oral capecitabine twice daily.']
    v = emb.encode(texts)
    assert v.shape == (3, 300)
    assert v.dtype == np.float32
    assert not np.isnan(v).any()
    print(f'{emb.name}: OK  shape={v.shape}')
```

---

## Recommended server workflow

From the `TrainWord2Vec/TrainWord2Vec` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m nltk.downloader punkt
```

Then run steps 1 → 3 in order.
