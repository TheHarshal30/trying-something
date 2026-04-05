# Contributing a Model

How to add your model to the benchmark pipeline so it gets tested automatically across all downstream tasks.

---

## How the pipeline works

You write the model. Dawood runs the benchmarks. The only contract between you and the evaluation pipeline is a single Python class with two methods defined in `evaluation/base_embedder.py`:

| Method | Description |
|--------|-------------|
| `load(model_path: str)` | Load your weights from the given folder path. Called once before encoding. |
| `encode(texts: list[str], batch_size: int) → np.ndarray` | Convert N texts into a `(N, dim)` float32 numpy array. This is the only function the eval scripts call. |
| `name → str` *(optional)* | Human readable name used in results files. Defaults to class name. |

Your `encode()` receives any text — a 2-word medical mention or a 300-word clinical paragraph — and must return a fixed-size vector. How you produce that vector is entirely up to you.

---

## Folder structure

Each model gets its own folder inside `models/`. The folder name is what gets passed to the benchmark runner as `--model`.

```
models/
├── word2vec/                        ← --model word2vec
│   ├── model.py                     ← required
│   └── weights/
│       └── word2vec.bin             ← required
│
├── word2vec_umls/                   ← --model word2vec_umls
│   ├── model.py                     ← required
│   └── weights/
│       ├── word2vec_umls.bin        ← required
│       └── umls_vocab.json          ← required
│
├── transformer/                     ← --model transformer
│   ├── model.py                     ← required
│   ├── architecture.py              ← required
│   └── weights/
│       ├── config.json              ← required
│       └── model.pt                 ← required
│
└── transformer_umls/                ← --model transformer_umls
    ├── model.py                     ← required
    ├── architecture.py              ← required
    └── weights/
        ├── config.json              ← required
        ├── model.pt                 ← required
        └── umls_vocab.json          ← required
```

### Rules

- `model.py` — your embedder class. Must extend `BaseEmbedder` and implement `load()` and `encode()`
- `weights/` — all binary files (`.pt`, `.bin`, `.npy`). Any filename is fine as long as `load()` knows where to look
- **Folder name = model name.** Use lowercase with underscores only
- **Never put weights in git.** Add your `weights/` folder to `.gitignore` and share via Google Drive. Link in your PR description
- The difference between base and UMLS variants is that UMLS variants also include `umls_vocab.json` — a mapping of UMLS CUIs to canonical names so the model can incorporate medical KB knowledge

---

## What your model.py must look like

Your `model.py` must import `BaseEmbedder` and implement exactly these two methods. Nothing else is required.

```python
import sys
sys.path.append('evaluation/')

from base_embedder import BaseEmbedder
import numpy as np


class YourModelEmbedder(BaseEmbedder):

    def __init__(self):
        self.model = None
        self._name = 'your_model'  # must match folder name

    def load(self, model_path: str) -> None:
        """load weights from model_path folder"""
        # your loading code here
        pass

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        input  : list of N strings (words, phrases, or full sentences)
        output : np.ndarray shape (N, embedding_dim), dtype float32
        """
        # your encoding code here
        pass

    @property
    def name(self) -> str:
        return self._name
```

> **Important:** `encode()` must always return `float32`. Add `.astype(np.float32)` at the end if your model outputs `float64` or torch tensors.

---

## Example — Word2Vec model

### model.py

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):

    def __init__(self):
        self.wv    = None   # word vectors (KeyedVectors)
        self._name = 'word2vec'

    def load(self, model_path: str) -> None:
        """
        Load word vectors from weights folder.
        Expects: models/word2vec/weights/word2vec.bin
        """
        weights_file = os.path.join(model_path, 'weights', 'word2vec.bin')
        print(f'loading Word2Vec from {weights_file}...')
        self.wv = KeyedVectors.load_word2vec_format(
            weights_file, binary=True
        )
        print(f'loaded — vocab size: {len(self.wv)}, dim: {self.wv.vector_size}')

    def _embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single text by averaging word vectors.
        Ignores OOV words. Returns zero vector if all words are OOV.
        """
        tokens  = text.lower().split()
        vectors = []

        for token in tokens:
            if token in self.wv:
                vectors.append(self.wv[token])

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        input  : list of N strings
        output : (N, vector_size) float32
        batch_size is ignored for Word2Vec (no GPU batching needed)
        """
        if self.wv is None:
            raise RuntimeError('call load() before encode()')

        embeddings = [self._embed_one(text) for text in texts]
        return np.vstack(embeddings).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
```

### Saving your weights

```python
# after training your Word2Vec model, save weights like this:
from gensim.models import Word2Vec

# your trained model
model = Word2Vec(...)
model.train(...)

# save in binary format — this is what load() expects
model.wv.save_word2vec_format(
    'models/word2vec/weights/word2vec.bin',
    binary=True
)
print('saved')
```

---

## Example — Custom Transformer model

### architecture.py

```python
import torch
import torch.nn as nn


class MedicalTransformer(nn.Module):
    """
    Simple transformer encoder for medical text embedding.
    Produces a single vector per input via mean pooling.
    """

    def __init__(
        self,
        vocab_size  : int   = 30522,  # BERT vocab size
        embed_dim   : int   = 256,
        num_heads   : int   = 4,
        num_layers  : int   = 4,
        ff_dim      : int   = 512,
        max_seq_len : int   = 512,
        dropout     : float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_seq_len, embed_dim)
        self.dropout   = nn.Dropout(dropout)

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        input_ids      : (batch, seq_len) token ids
        attention_mask : (batch, seq_len) 1=real token, 0=padding
        returns        : (batch, embed_dim) via mean pooling
        """
        B, L      = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x        = self.token_emb(input_ids) + self.pos_emb(positions)
        x        = self.dropout(x)
        pad_mask = (attention_mask == 0)
        x        = self.encoder(x, src_key_padding_mask=pad_mask)
        x        = self.norm(x)

        # mean pooling — ignore padding tokens
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (batch, embed_dim)


def save_weights(model, save_dir: str):
    """save architecture config + weights after training"""
    import json, os
    os.makedirs(save_dir, exist_ok=True)

    config = {
        'vocab_size'  : model.token_emb.num_embeddings,
        'embed_dim'   : model.embed_dim,
        'num_heads'   : model.encoder.layers[0].self_attn.num_heads,
        'num_layers'  : len(model.encoder.layers),
        'ff_dim'      : model.encoder.layers[0].linear1.out_features,
        'max_seq_len' : model.pos_emb.num_embeddings,
    }

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    print(f'saved config + weights to {save_dir}')
```

### model.py

```python
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))

import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from base_embedder import BaseEmbedder
from architecture import MedicalTransformer


class TransformerEmbedder(BaseEmbedder):

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.device    = self._get_device()
        self._name     = 'transformer'

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')    # M1 Mac
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def load(self, model_path: str) -> None:
        """
        Loads config.json + model.pt from model_path/weights/
        model_path: 'models/transformer'
        """
        weights_dir = os.path.join(model_path, 'weights')

        with open(os.path.join(weights_dir, 'config.json')) as f:
            config = json.load(f)

        self.model = MedicalTransformer(**config)

        state = torch.load(
            os.path.join(weights_dir, 'model.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # use BERT tokenizer (matches vocab_size=30522)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print(f'loaded transformer — dim: {config["embed_dim"]}, device: {self.device}')

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('call load() before encode()')

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f'encoding with {self.name}', unit='batch'):

            batch   = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            input_ids      = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                embeddings = self.model(input_ids, attention_mask)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
```

### Saving your weights

```python
from architecture import MedicalTransformer, save_weights

# your trained model
model = MedicalTransformer(
    vocab_size=30522,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    ff_dim=512,
)
# ... train model ...

# save config + weights
save_weights(model, 'models/transformer/weights')
```

---

## PR checklist

Before opening a pull request verify all of these:

- [ ] `model.py` exists in your model folder and extends `BaseEmbedder`
- [ ] `encode()` returns `float32` numpy array of shape `(N, dim)`
- [ ] `load()` works with just the model folder path as argument
- [ ] `name` property returns the same string as your folder name
- [ ] weights are **not** in git — shared via Drive with link in PR description
- [ ] sanity check below passes

### Sanity check — run this before opening PR

```python
# run from project root
# replace YourModelEmbedder + path with your actual class + folder
import sys
sys.path.append('evaluation/')

from models.your_model.model import YourModelEmbedder
import numpy as np

embedder = YourModelEmbedder()
embedder.load('models/your_model')

texts = [
    'diabetes mellitus',
    'heart failure',
    'Patients received oral capecitabine twice daily.',
]

vectors = embedder.encode(texts)

assert vectors.shape[0] == len(texts), 'wrong number of embeddings'
assert vectors.dtype == np.float32,    'must be float32'
assert vectors.shape[1] > 0,           'embedding dim must be > 0'
assert not np.isnan(vectors).any(),    'NaN in embeddings'
assert not np.isinf(vectors).any(),    'Inf in embeddings'

print(f'shape  : {vectors.shape}')
print(f'dtype  : {vectors.dtype}')
print(f'all checks passed — ready to push')
```

Once your PR is merged, I will add your model to `evaluation/run_all.py` and run:

```bash
python evaluation/run_all.py --model your_model_name
```

Results will appear automatically in `results/your_model_name/` and the leaderboard will update.
