"""
03_align_ntxent.py
──────────────────
Fine-tune Word2Vec embeddings with NT-Xent contrastive loss
using UMLS synonym pairs as positive examples.

Architecture
────────────
  ┌─────────────────────────────────────────────────┐
  │  Frozen / warm embedding table (from Word2Vec)  │
  │   → mean-pool over tokens → (batch, dim)        │
  │   → small MLP projection head (dim → proj_dim)  │
  │   → L2 normalise                                │
  │   → NT-Xent loss over in-batch negatives        │
  └─────────────────────────────────────────────────┘

After alignment we project the embedding table back to the original
vector space and save it as a new Word2Vec .bin file.

NT-Xent loss (SimCLR formulation)
──────────────────────────────────
For a batch of N synonym pairs (a_i, b_i):
  - Construct 2N embeddings z_1…z_2N
  - For each z_i, the positive is its pair, all other 2N-1 are negatives
  - loss_i = -log( exp(sim(z_i, z_pos) / τ) / Σ_j≠i exp(sim(z_i, z_j) / τ) )
  - Total loss = mean over all 2N anchors

Usage
─────
    # Minimal (uses all defaults)
    python training/03_align_ntxent.py \\
        --w2v_bin   models/word2vec/weights/word2vec.bin \\
        --pairs     data/umls_pairs.txt \\
        --output    models/word2vec_umls/weights/word2vec_umls.bin

    # Full control
    python training/03_align_ntxent.py \\
        --w2v_bin   models/word2vec/weights/word2vec.bin \\
        --pairs     data/umls_pairs.txt \\
        --output    models/word2vec_umls/weights/word2vec_umls.bin \\
        --proj_dim  256  --temperature 0.07 \\
        --batch_size 512 --epochs 10 --lr 3e-4 \\
        --freeze_embedding          # keep embedding table frozen, only train head

Requirements
────────────
    pip install torch gensim tqdm numpy
"""

import argparse
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UMLSPairDataset(Dataset):
    """
    Reads the tab-separated pairs file produced by 02_extract_umls_pairs.py.
    Each item is (anchor_str, positive_str).
    """
    def __init__(self, pairs_path: str):
        self.pairs: List[Tuple[str, str]] = []
        with open(pairs_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    self.pairs.append((parts[0].strip(), parts[1].strip()))
        log.info(f'Loaded {len(self.pairs):,} synonym pairs from {pairs_path}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Text → vector using the embedding table
# ─────────────────────────────────────────────────────────────────────────────

def mean_pool(
    texts: List[str],
    word2idx: dict,
    embedding: nn.Embedding,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a list of text strings to (N, embed_dim) tensors by
    mean-pooling token embeddings (ignoring OOV tokens).

    Returns a tensor on `device`.
    """
    vectors = []
    for text in texts:
        indices = [word2idx[t] for t in text.lower().split() if t in word2idx]
        if indices:
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            vec   = embedding(idx_t).mean(dim=0)
        else:
            vec = torch.zeros(embedding.embedding_dim, device=device)
        vectors.append(vec)
    return torch.stack(vectors)   # (N, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Projection head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head (following SimCLR).
    Embed → Linear → BN → ReLU → Linear → L2-norm
    """
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent loss
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent loss for a batch of N positive pairs.

    Parameters
    ----------
    z_a : (N, proj_dim) — L2-normalised projections of anchors
    z_b : (N, proj_dim) — L2-normalised projections of positives
    temperature : float — τ; lower = sharper distribution

    Returns
    -------
    scalar loss (mean over 2N anchors)
    """
    N = z_a.size(0)

    # Stack into (2N, proj_dim)
    z = torch.cat([z_a, z_b], dim=0)

    # Cosine similarity matrix (2N, 2N)
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive indices:
    #   for each i in [0, N)  → positive is i+N
    #   for each i in [N, 2N) → positive is i-N
    targets = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])

    loss = F.cross_entropy(sim, targets)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info(f'Device: {device}')

    # ── load Word2Vec weights ─────────────────────────────────────────────────
    log.info(f'Loading Word2Vec from {args.w2v_bin} ...')
    wv = KeyedVectors.load_word2vec_format(args.w2v_bin, binary=True)
    vocab     = list(wv.key_to_index.keys())
    word2idx  = {w: i for i, w in enumerate(vocab)}
    embed_dim = wv.vector_size

    # Build embedding matrix from Word2Vec weights
    init_weights = torch.tensor(wv.vectors, dtype=torch.float32)   # (V, dim)
    embedding = nn.Embedding.from_pretrained(
        init_weights,
        freeze=args.freeze_embedding,
        padding_idx=None,
    ).to(device)

    log.info(
        f'  vocab: {len(vocab):,}  dim: {embed_dim}'
        f'  embedding frozen: {args.freeze_embedding}'
    )

    # ── model components ──────────────────────────────────────────────────────
    head = ProjectionHead(embed_dim, args.proj_dim).to(device)

    params = list(head.parameters())
    if not args.freeze_embedding:
        params += list(embedding.parameters())

    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    total_steps = (len(UMLSPairDataset(args.pairs)) // args.batch_size) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps, eta_min=1e-6
    )

    # ── dataset / loader ─────────────────────────────────────────────────────
    dataset = UMLSPairDataset(args.pairs)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,      # text collation is fast on CPU; no need for workers
        drop_last   = True,   # NT-Xent needs consistent batch size for BN
    )

    # ── training ─────────────────────────────────────────────────────────────
    log.info(
        f'Training NT-Xent alignment for {args.epochs} epochs  '
        f'batch={args.batch_size}  τ={args.temperature}  proj_dim={args.proj_dim}'
    )

    for epoch in range(1, args.epochs + 1):
        embedding.train()
        head.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
        for anchors, positives in pbar:

            # Mean-pool token embeddings for each side
            z_a_raw = mean_pool(anchors,   word2idx, embedding, device)  # (N, dim)
            z_b_raw = mean_pool(positives, word2idx, embedding, device)  # (N, dim)

            # Project + L2-normalise
            z_a = head(z_a_raw)   # (N, proj_dim)
            z_b = head(z_b_raw)   # (N, proj_dim)

            loss = nt_xent_loss(z_a, z_b, args.temperature)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             lr=f'{scheduler.get_last_lr()[0]:.2e}')

        avg = total_loss / len(loader)
        log.info(f'Epoch {epoch}  avg_loss={avg:.4f}')

    # ── extract aligned embedding matrix ─────────────────────────────────────
    log.info('Extracting aligned embeddings ...')
    embedding.eval()
    with torch.no_grad():
        aligned_matrix = embedding.weight.cpu().numpy()  # (V, dim)

    # ── build new KeyedVectors with aligned weights ───────────────────────────
    aligned_kv = KeyedVectors(vector_size=embed_dim)
    aligned_kv.add_vectors(vocab, aligned_matrix)

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    aligned_kv.save_word2vec_format(args.output, binary=True)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    log.info(f'Saved aligned vectors → {args.output}  ({size_mb:.1f} MB)')

    # ── quick sanity: cosine sim of a known synonym pair ─────────────────────
    test_pairs = [
        ('heart attack', 'myocardial infarction'),
        ('diabetes',     'diabetes mellitus'),
        ('aspirin',      'acetylsalicylic acid'),
    ]
    log.info('Sanity check — cosine similarities after alignment:')

    def embed_phrase(phrase):
        tokens = [t for t in phrase.split() if t in aligned_kv]
        if not tokens:
            return None
        return np.mean([aligned_kv[t] for t in tokens], axis=0)

    for a, b in test_pairs:
        va, vb = embed_phrase(a), embed_phrase(b)
        if va is not None and vb is not None:
            cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            log.info(f'  "{a}" ↔ "{b}"  cosine={cos:.4f}')
        else:
            log.info(f'  "{a}" ↔ "{b}"  (one or both OOV)')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='NT-Xent alignment of Word2Vec with UMLS pairs')

    p.add_argument('--w2v_bin',          required=True,
                   help='Path to trained word2vec.bin (from step 01)')
    p.add_argument('--pairs',            required=True,
                   help='Path to UMLS synonym pairs .txt (from step 02)')
    p.add_argument('--output',           required=True,
                   help='Output path for aligned word2vec_umls.bin')

    p.add_argument('--proj_dim',         type=int,   default=256,
                   help='Projection head output dimension (default: 256)')
    p.add_argument('--temperature',      type=float, default=0.07,
                   help='NT-Xent temperature τ (default: 0.07)')
    p.add_argument('--batch_size',       type=int,   default=512,
                   help='Pairs per batch (default: 512)')
    p.add_argument('--epochs',           type=int,   default=10,
                   help='Training epochs (default: 10)')
    p.add_argument('--lr',               type=float, default=3e-4,
                   help='AdamW learning rate (default: 3e-4)')
    p.add_argument('--freeze_embedding', action='store_true',
                   help='If set, freeze the embedding table and only train the projection head')
    p.add_argument('--seed',             type=int,   default=42)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args)
