import os
import re
import sys

import numpy as np
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

from base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):

    def __init__(self):
        self.wv    = None
        self._name = 'word2vec'

    def load(self, model_path: str) -> None:
        weights_file = os.path.join(model_path, 'weights', 'word2vec.bin')
        print(f'Loading from {weights_file}...')
        self.wv = KeyedVectors.load_word2vec_format(weights_file, binary=True)
        print(f'Loaded — vocab: {len(self.wv)}, dim: {self.wv.vector_size}')

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.lower())
        vectors = [self.wv[t] for t in tokens if t in self.wv]
        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        if self.wv is None:
            raise RuntimeError('Call load() before encode()')
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
