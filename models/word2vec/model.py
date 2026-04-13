import os
import re
import sys
from pathlib import Path
import __main__

import numpy as np
from gensim.models import FastText, KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

from base_embedder import BaseEmbedder
from tfidf import load_tfidf


class Word2VecEmbedder(BaseEmbedder):

    def __init__(self):
        self.wv    = None
        self.model = None
        self._name = 'word2vec'
        self.idf   = None
        self.use_tfidf = True

    def load(self, model_path: str) -> None:
        weights_file = os.path.join(model_path, 'weights', 'word2vec.bin')
        fasttext_file = os.path.join(model_path, 'weights', 'fasttext.model')
        tfidf_file = os.path.join(model_path, 'weights', 'tfidf_idf.json')
        if os.path.exists(fasttext_file):
            print(f'Loading fastText from {fasttext_file}...')
            # Older gensim checkpoints may reference the training callback class
            # from the original __main__ module. Register a harmless placeholder
            # so evaluation can still load those checkpoints.
            if not hasattr(__main__, 'EpochLogger'):
                class EpochLogger:  # noqa: N801 - match pickled callback name
                    def __init__(self, *args, **kwargs):
                        pass

                __main__.EpochLogger = EpochLogger
            self.model = FastText.load(fasttext_file)
            if hasattr(self.model, 'callbacks'):
                self.model.callbacks = []
            self.wv = self.model.wv
        else:
            print(f'Loading from {weights_file}...')
            self.wv = KeyedVectors.load_word2vec_format(weights_file, binary=True)
            self.model = None
        print(f'Loaded — vocab: {len(self.wv)}, dim: {self.wv.vector_size}')
        if os.path.exists(tfidf_file):
            self.idf = load_tfidf(Path(tfidf_file))
            print(f'Loaded TF-IDF weights from {tfidf_file} ({len(self.idf)} terms)')

    def _lookup(self, token: str) -> np.ndarray | None:
        if self.wv is None:
            return None
        if self.model is not None:
            return np.asarray(self.wv[token], dtype=np.float32)
        if token in self.wv:
            return np.asarray(self.wv[token], dtype=np.float32)
        return None

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.lower())
        if self.idf and self.use_tfidf:
            weighted_vectors = []
            weights = []
            for token in tokens:
                vector = self._lookup(token)
                if vector is None:
                    continue
                weight = float(self.idf.get(token, 1.0))
                weighted_vectors.append(vector * weight)
                weights.append(weight)
            if not weighted_vectors:
                return np.zeros(self.wv.vector_size, dtype=np.float32)
            return (np.sum(weighted_vectors, axis=0) / max(np.sum(weights), 1e-8)).astype(np.float32)

        vectors = []
        for token in tokens:
            vector = self._lookup(token)
            if vector is not None:
                vectors.append(vector)
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
