import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    """
    Baseline Word2Vec embedder trained on PubMed abstracts.
    Encodes text by mean-pooling over in-vocabulary word vectors.
    OOV tokens are silently skipped; all-OOV inputs return a zero vector.
    """

    def __init__(self):
        self.wv    = None        # gensim KeyedVectors
        self._name = 'word2vec'

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        """
        Load word vectors from the weights folder.
        Expects: <model_path>/weights/word2vec.bin  (Word2Vec binary format)
        """
        weights_file = os.path.join(model_path, 'weights', 'word2vec.bin')
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f'weights not found at {weights_file}\n'
                f'Run training/01_train_word2vec.py first, then copy the output here.'
            )
        print(f'[word2vec] loading from {weights_file} ...')
        self.wv = KeyedVectors.load_word2vec_format(weights_file, binary=True)
        print(f'[word2vec] ready — vocab: {len(self.wv):,}  dim: {self.wv.vector_size}')

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _embed_one(self, text: str) -> np.ndarray:
        """
        Mean-pool word vectors for a single text string.
        Lowercases and whitespace-splits before lookup.
        Returns zero vector for fully OOV input.
        """
        tokens  = text.lower().split()
        vectors = [self.wv[t] for t in tokens if t in self.wv]

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # encode  (the only method the eval pipeline calls)
    # ------------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert N text strings to (N, vector_size) float32 numpy array.
        batch_size is accepted for API compatibility but ignored
        (Word2Vec lookup is already O(1) per token, no GPU batching needed).
        """
        if self.wv is None:
            raise RuntimeError('call load() before encode()')

        embeddings = [self._embed_one(t) for t in texts]
        return np.vstack(embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # name property
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name
