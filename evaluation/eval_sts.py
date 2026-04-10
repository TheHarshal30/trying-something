"""
evaluation/eval_sts.py

Evaluates any BaseEmbedder on Semantic Textual Similarity:
  - BIOSSES

How it works:
  1. encode(sentence1) -> vector A
  2. encode(sentence2) -> vector B
  3. cosine_similarity(A, B) -> model score
  4. compare model scores against human scores via Pearson r + Spearman r

Usage:
    python eval_sts.py --model pubmedbert --dataset biosses
"""

import json
import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from assets import ensure_sts_assets

# ─── paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
BIOSSES_DIR = DATA_DIR / 'raw' / 'biosses'


# ─── dataset loader ───────────────────────────────────────────────────────────

def load_biosses(split: str = 'test') -> pd.DataFrame:
    """
    Load BIOSSES split.
    Returns DataFrame with columns: text_1, text_2, label
    """
    file_map = {
        'train'     : BIOSSES_DIR / 'train.jsonl',
        'validation': BIOSSES_DIR / 'validation.jsonl',
        'test'      : BIOSSES_DIR / 'test.jsonl',
    }
    rows = []
    with open(file_map[split]) as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                'text_1': row['text_1'].strip(),
                'text_2': row['text_2'].strip(),
                'label' : float(row['label'])
            })
    df = pd.DataFrame(rows)
    print(f'BIOSSES {split}: {len(df)} pairs loaded')
    return df


# ─── cosine similarity ────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between two (N, dim) matrices.
    Returns (N,) array of similarity scores.
    """
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    return np.sum((a / norm_a) * (b / norm_b), axis=1)


# ─── main evaluation function ─────────────────────────────────────────────────

def evaluate(
    embedder,
    dataset     : str  = 'biosses',
    split       : str  = 'test',
    batch_size  : int  = 32,
    save_figures: bool = True
):
    """
    Run STS evaluation for one model + dataset combo.

    Args:
        embedder    : any BaseEmbedder instance (already loaded)
        dataset     : 'biosses'
        split       : 'train' | 'validation' | 'test'
        batch_size  : passed to embedder.encode()
        save_figures: save plots to results/
    """

    ensure_sts_assets(dataset)

    # ── load dataset ──
    if dataset == 'biosses':
        df          = load_biosses(split)
        dataset_tag = 'biosses'
    else:
        raise ValueError(f'unknown dataset: {dataset}. choose biosses')

    # ── encode both sentences ──
    # encode() handles everything internally regardless of model type
    # word2vec → averages word vectors
    # transformer → pools token embeddings
    # output is always (N, dim)
    print(f'\nencoding sentence 1...')
    t0    = time.time()
    emb_1 = embedder.encode(df['text_1'].tolist(), batch_size=batch_size)

    print(f'encoding sentence 2...')
    emb_2 = embedder.encode(df['text_2'].tolist(), batch_size=batch_size)
    encode_time = time.time() - t0

    print(f'encoded {len(df)*2} sentences in {encode_time:.1f}s')
    print(f'embedding shape: {emb_1.shape}')

    # ── compute cosine similarity for each pair ──
    model_scores = cosine_sim(emb_1, emb_2)   # (N,)
    human_scores = df['label'].values          # (N,)

    # ── compute metrics ──
    pearson_r,  pearson_p  = pearsonr(model_scores,  human_scores)
    spearman_r, spearman_p = spearmanr(model_scores, human_scores)

    result = {
        'model'      : embedder.name,
        'dataset'    : dataset_tag,
        'split'      : split,
        'pearson_r'  : round(float(pearson_r),  4),
        'pearson_p'  : round(float(pearson_p),  4),
        'spearman_r' : round(float(spearman_r), 4),
        'spearman_p' : round(float(spearman_p), 4),
        'num_pairs'  : len(df),
        'runtime_sec': round(encode_time, 2),
        'date'       : str(date.today()),
    }

    print(f'\n{"="*50}')
    print(f'  model      : {result["model"]}')
    print(f'  dataset    : {result["dataset"]}')
    print(f'  split      : {result["split"]}')
    print(f'  Pearson r  : {result["pearson_r"]:.4f}')
    print(f'  Spearman r : {result["spearman_r"]:.4f}')
    print(f'  runtime    : {result["runtime_sec"]}s')
    print(f'{"="*50}\n')

    # ── save results ──
    out_dir = RESULTS_DIR / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f'sts_{dataset_tag}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'results saved to {result_path}')

    # ── save figures ──
    if save_figures:
        fig_dir = out_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # scatter: model scores vs human scores
        axes[0].scatter(human_scores, model_scores, alpha=0.6, s=30)
        axes[0].set_xlabel('human score (0-5)')
        axes[0].set_ylabel('cosine similarity')
        axes[0].set_title(f'Human vs Model scores\nPearson r = {pearson_r:.4f}')

        # add trend line
        z = np.polyfit(human_scores, model_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(human_scores.min(), human_scores.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r--', alpha=0.8)

        # histogram of model cosine scores
        axes[1].hist(model_scores, bins=20, edgecolor='white')
        axes[1].set_xlabel('cosine similarity')
        axes[1].set_ylabel('count')
        axes[1].set_title('Model score distribution')

        plt.suptitle(f'{embedder.name} | {dataset_tag} | {split}', fontsize=13)
        plt.tight_layout()

        fig_path = fig_dir / f'{dataset_tag}_results.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f'figure saved to {fig_path}')

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True,
                        help='pubmedbert | word2vec | transformer etc.')
    parser.add_argument('--dataset',    default='biosses',
                        help='biosses (default: biosses)')
    parser.add_argument('--split',      default='test',
                        help='train | validation | test (default: test)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # ── load embedder ──
    if args.model == 'pubmedbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'pubmedbert-local'))

    # team models added here when pushed
    # elif args.model == 'word2vec':
    #     from word2vec_embedder import Word2VecEmbedder
    #     embedder = Word2VecEmbedder()
    #     embedder.load(str(ROOT / 'models' / 'word2vec' / 'weights'))

    else:
        raise ValueError(f'unknown model: {args.model}')

    evaluate(
        embedder   = embedder,
        dataset    = args.dataset,
        split      = args.split,
        batch_size = args.batch_size,
    )
