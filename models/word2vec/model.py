import os
import re
import sys
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

from base_embedder import BaseEmbedder
from tfidf import load_tfidf


class Word2VecEmbedder(BaseEmbedder):

    def __init__(self):
        self.wv    = None
        self._name = 'word2vec'
        self.idf   = None

    def load(self, model_path: str) -> None:
        weights_file = os.path.join(model_path, 'weights', 'word2vec.bin')
        tfidf_file = os.path.join(model_path, 'weights', 'tfidf_idf.json')
        print(f'Loading from {weights_file}...')
        self.wv = KeyedVectors.load_word2vec_format(weights_file, binary=True)
        print(f'Loaded — vocab: {len(self.wv)}, dim: {self.wv.vector_size}')
        if os.path.exists(tfidf_file):
            self.idf = load_tfidf(Path(tfidf_file))
            print(f'Loaded TF-IDF weights from {tfidf_file} ({len(self.idf)} terms)')

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.lower())
        if self.idf:
            weighted_vectors = []
            weights = []
            for token in tokens:
                if token not in self.wv:
                    continue
                weight = float(self.idf.get(token, 1.0))
                weighted_vectors.append(self.wv[token] * weight)
                weights.append(weight)
            if not weighted_vectors:
                return np.zeros(self.wv.vector_size, dtype=np.float32)
            return (np.sum(weighted_vectors, axis=0) / max(np.sum(weights), 1e-8)).astype(np.float32)

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
