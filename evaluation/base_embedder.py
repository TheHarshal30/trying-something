"""
evaluation/base_embedder.py

Interface contract for ALL models — pretrained baselines and team models.
Every embedder must implement load() and encode().
Evaluation scripts never touch model internals.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):

    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model weights from model_path.
        e.g. 'models/pubmedbert-local'
             'models/word2vec/weights/model.bin'
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert a list of strings to embeddings.

        Args:
            texts      : list of N strings
            batch_size : process in batches (keep low on 8GB RAM)

        Returns:
            np.ndarray shape (N, embedding_dim), dtype float32
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Model name used in results files. Override in subclass."""
        return self.__class__.__name__
