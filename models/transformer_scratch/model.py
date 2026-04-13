import os
import sys
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

from base_embedder import BaseEmbedder
from preprocess import normalize_text


class TransformerScratchEmbedder(BaseEmbedder):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._name = "transformer_scratch"
        self.device = self._get_device()
        self.pooling_strategy = "mean"
        self.normalization_strategy = "chemical"

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self, model_path: str) -> None:
        if os.path.exists(os.path.join(model_path, "config.json")):
            weights_dir = model_path
        else:
            weights_dir = os.path.join(model_path, "weights", "final")
        print(f"loading TransformerScratch from {weights_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        self.model = AutoModel.from_pretrained(weights_dir)
        config_path = os.path.join(weights_dir, "embedding_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
                self.pooling_strategy = config.get("pooling_strategy", "mean")
                self.normalization_strategy = config.get("normalization_strategy", self.normalization_strategy)
        self.model.to(self.device)
        self.model.eval()
        print(f"loaded — hidden size: {self.model.config.hidden_size}")
        print(f"pooling strategy: {self.pooling_strategy}")
        print(f"normalization strategy: {self.normalization_strategy}")

    def _masked_mean(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return outputs.last_hidden_state[:, 0]
        if self.pooling_strategy == "mean":
            return self._masked_mean(outputs.last_hidden_state, attention_mask)
        if self.pooling_strategy == "last4_mean":
            hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
            return self._masked_mean(hidden, attention_mask)
        raise ValueError(f"unknown pooling strategy: {self.pooling_strategy}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load() before encode()")

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = [
                normalize_text(text, strategy=self.normalization_strategy)
                for text in texts[start:start + batch_size]
            ]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                model_out = self.model(**encoded, output_hidden_states=True, return_dict=True)
                pooled = self._pool(model_out, encoded["attention_mask"])
            outputs.append(pooled.cpu().float().numpy())

        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name
