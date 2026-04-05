"""
evaluation/pubmedbert_embedder.py

Pretrained baseline embedder using PubMedBERT.
Uses CLS token as the sentence/mention embedding.

Usage:
    embedder = PubMedBERTEmbedder()
    embedder.load('models/pubmedbert-local')
    vectors = embedder.encode(['diabetes mellitus', 'heart failure'])
    # vectors.shape -> (2, 768)
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from base_embedder import BaseEmbedder


class PubMedBERTEmbedder(BaseEmbedder):

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.device    = self._get_device()
        self._name     = 'pubmedbert'

    def _get_device(self) -> torch.device:
        # M1 Mac uses MPS, fallback to CPU
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def load(self, model_path: str) -> None:
        """
        Load tokenizer and model from local path.
        model_path: 'models/pubmedbert-local'
        """
        print(f'loading PubMedBERT from {model_path} on {self.device}...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f'loaded — embedding dim: {self.model.config.hidden_size}')

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts using CLS token.

        Args:
            texts      : list of strings
            batch_size : keep at 32 or lower on 8GB RAM

        Returns:
            np.ndarray of shape (N, 768), dtype float32
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('call load() before encode()')

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f'encoding with {self.name}',
                      unit='batch'):

            batch = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,      
                return_tensors='pt'
            )

            # move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            # CLS token is the first token of last_hidden_state
            cls_embeddings = output.last_hidden_state[:, 0, :]

            # move back to CPU and convert to numpy
            all_embeddings.append(cls_embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings)

    @property
    def name(self) -> str:
        return self._name


if __name__ == '__main__':
    # quick sanity check
    embedder = PubMedBERTEmbedder()
    embedder.load('models/pubmedbert-local')

    texts = [
        'diabetes mellitus',
        'heart failure',
        'skin tumour',
        'adenomatous polyposis coli',
    ]

    vectors = embedder.encode(texts, batch_size=2)
    print(f'output shape : {vectors.shape}')
    print(f'dtype        : {vectors.dtype}')

    # cosine similarity between first two
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f'sim(diabetes, heart failure)     : {cosine(vectors[0], vectors[1]):.4f}')
    print(f'sim(diabetes, diabetes)          : {cosine(vectors[0], vectors[0]):.4f}')
    print(f'sim(skin tumour, polyposis coli) : {cosine(vectors[2], vectors[3]):.4f}')
