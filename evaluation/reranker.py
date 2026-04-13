from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Reranker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        return self.mlp(x).squeeze(-1)


@dataclass
class RerankerConfig:
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hard_negatives: int = 8
    use_hard_negatives: bool = True
    max_train_mentions: int | None = 2000
    seed: int = 42


def _build_examples(
    mention_embeddings: np.ndarray,
    gold_ids: list[str],
    kb_embeddings: np.ndarray,
    kb_ids: list[str],
    hard_negatives: int,
    use_hard_negatives: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = mention_embeddings @ kb_embeddings.T
    kb_index = {mesh_id: idx for idx, mesh_id in enumerate(kb_ids)}

    left = []
    right = []
    labels = []

    for idx, gold_id in enumerate(gold_ids):
        if gold_id not in kb_index:
            continue

        mention = mention_embeddings[idx]
        pos_idx = kb_index[gold_id]

        left.append(mention)
        right.append(kb_embeddings[pos_idx])
        labels.append(1.0)

        if use_hard_negatives:
            ranked = np.argsort(sim[idx])[::-1]
            negative_indices = []
            for cand_idx in ranked:
                if kb_ids[cand_idx] == gold_id:
                    continue
                negative_indices.append(cand_idx)
                if len(negative_indices) >= hard_negatives:
                    break
        else:
            candidates = [cand_idx for cand_idx, mesh_id in enumerate(kb_ids) if mesh_id != gold_id]
            sample_size = min(hard_negatives, len(candidates))
            negative_indices = random.sample(candidates, sample_size) if sample_size else []

        for cand_idx in negative_indices:
            left.append(mention)
            right.append(kb_embeddings[cand_idx])
            labels.append(0.0)

    return (
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def train_reranker(
    mention_embeddings: np.ndarray,
    gold_ids: list[str],
    kb_embeddings: np.ndarray,
    kb_ids: list[str],
    config: RerankerConfig,
    device: torch.device | None = None,
) -> Reranker | None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.max_train_mentions is not None and len(mention_embeddings) > config.max_train_mentions:
        chosen = np.random.choice(len(mention_embeddings), size=config.max_train_mentions, replace=False)
        mention_embeddings = mention_embeddings[chosen]
        gold_ids = [gold_ids[i] for i in chosen]

    left, right, labels = _build_examples(
        mention_embeddings=mention_embeddings,
        gold_ids=gold_ids,
        kb_embeddings=kb_embeddings,
        kb_ids=kb_ids,
        hard_negatives=config.hard_negatives,
        use_hard_negatives=config.use_hard_negatives,
    )

    if len(labels) == 0:
        return None

    dataset = TensorDataset(
        torch.from_numpy(left),
        torch.from_numpy(right),
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = Reranker(dim=kb_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()
    for epoch in range(config.epochs):
        running_loss = 0.0
        for mention_batch, cand_batch, label_batch in loader:
            mention_batch = mention_batch.to(device)
            cand_batch = cand_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(mention_batch, cand_batch)
            loss = F.binary_cross_entropy_with_logits(logits, label_batch)

            if torch.isnan(loss):
                print("NaN detected, stopping reranker training")
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        print(
            f"reranker epoch {epoch + 1}/{config.epochs} | "
            f"loss {running_loss / max(len(loader), 1):.4f}"
        )

    model.eval()
    return model


def rerank_candidates(
    model: Reranker,
    mention_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mention = torch.from_numpy(mention_embedding.astype(np.float32)).unsqueeze(0).to(device)
    candidates = torch.from_numpy(candidate_embeddings.astype(np.float32)).to(device)

    with torch.no_grad():
        mention = mention.expand(candidates.size(0), -1)
        scores = model(mention, candidates).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    return candidate_indices[order]
