import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

from base_embedder import BaseEmbedder


class TransformerScratchEmbedder(BaseEmbedder):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._name = "transformer_scratch"
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self, model_path: str) -> None:
        weights_dir = os.path.join(model_path, "weights", "final")
        print(f"loading TransformerScratch from {weights_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        self.model = AutoModel.from_pretrained(weights_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"loaded — hidden size: {self.model.config.hidden_size}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load() before encode()")

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                model_out = self.model(**encoded)
                hidden = model_out.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu().float().numpy())

        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
