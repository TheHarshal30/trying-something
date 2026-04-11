import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecUMLSEmbedder(BaseEmbedder):
    """
    UMLS-grounded Word2Vec embedder.

    Built in two stages:
      1.  Trained as a standard Skip-gram Word2Vec on PubMed abstracts
          (identical corpus to the baseline word2vec model).
      2.  Embedding matrix fine-tuned with NT-Xent contrastive loss using
          UMLS synonym pairs (same CUI, different surface strings) as
          positive pairs.  All negatives are drawn in-batch.

    At inference time the model behaves identically to the baseline:
    texts are mean-pooled over their in-vocabulary tokens.  The difference
    is that semantically equivalent medical terms (e.g. "heart attack" and
    "myocardial infarction") are pulled much closer together in the vector
    space than they would be by distributional training alone.

    Required files (relative to model_path):
        weights/word2vec_umls.bin   –– aligned vectors in Word2Vec binary fmt
        weights/umls_vocab.json     –– {CUI: canonical_name, ...} mapping
    """

    def __init__(self):
        self.wv         = None   # gensim KeyedVectors (UMLS-aligned)
        self.umls_vocab = None   # dict: CUI -> canonical name
        self._name      = 'word2vec_umls'

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        """
        Load aligned word vectors + UMLS vocab from the weights folder.

        Expected layout:
            <model_path>/weights/word2vec_umls.bin
            <model_path>/weights/umls_vocab.json
        """
        weights_dir = os.path.join(model_path, 'weights')

        bin_path   = os.path.join(weights_dir, 'word2vec_umls.bin')
        vocab_path = os.path.join(weights_dir, 'umls_vocab.json')

        for p in (bin_path, vocab_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f'Required file not found: {p}\n'
                    f'Run the training pipeline (01 → 02 → 03) first.'
                )

        print(f'[word2vec_umls] loading aligned vectors from {bin_path} ...')
        self.wv = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        print(f'[word2vec_umls] ready — vocab: {len(self.wv):,}  dim: {self.wv.vector_size}')

        print(f'[word2vec_umls] loading UMLS vocab from {vocab_path} ...')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.umls_vocab = json.load(f)
        print(f'[word2vec_umls] UMLS vocab loaded — {len(self.umls_vocab):,} CUIs')

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _embed_one(self, text: str) -> np.ndarray:
        """
        Mean-pool aligned word vectors for a single text string.
        Lowercases and whitespace-splits before lookup.
        Returns zero vector for fully OOV input.
        """
        tokens  = text.lower().split()
        vectors = [self.wv[t] for t in tokens if t in self.wv]

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert N text strings to (N, vector_size) float32 numpy array.
        batch_size is accepted for API compatibility but ignored.
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
