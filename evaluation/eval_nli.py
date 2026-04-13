"""
evaluation/eval_nli.py

Evaluates any BaseEmbedder on Natural Language Inference:
  - NLI4CT (Clinical Trial NLI)

How it works:
  1. encode(sentence1) -> vector A  (clinical trial context)
  2. encode(sentence2) -> vector B  (hypothesis)
  3. build feature vector: [A, B, A-B, A*B]
  4. train logistic regression classifier on train split
  5. evaluate on test split -> Accuracy + per-class F1

Why [A, B, A-B, A*B]?
  - A, B          : raw representations of both sentences
  - A - B         : captures difference (important for contradiction)
  - A * B         : captures similarity (important for entailment)
  Together they give the classifier enough signal to distinguish
  entailment vs contradiction vs neutral.

Usage:
    python eval_nli.py --model pubmedbert --dataset nli4ct
"""

import json
import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from assets import ensure_nli_assets

# ─── paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
NLI4CT_DIR  = DATA_DIR / 'raw' / 'nli4ct'


# ─── dataset loader ───────────────────────────────────────────────────────────

def load_nli4ct(split: str = 'train') -> pd.DataFrame:
    """
    Load NLI4CT split.
    Returns DataFrame with columns: sentence1, sentence2, label
    """
    file_map = {
        'train'     : NLI4CT_DIR / 'train.jsonl',
        'validation': NLI4CT_DIR / 'validation.jsonl',
        'test'      : NLI4CT_DIR / 'test.jsonl',
    }

    # use train if requested split doesn't exist
    path = file_map.get(split)
    if not path or not path.exists():
        available = [s for s, p in file_map.items() if p.exists()]
        print(f'split "{split}" not found. available: {available}')
        path = file_map[available[0]]
        split = available[0]

    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            label = row.get('gold_label') or row.get('label', '')
            rows.append({
                'sentence1': row['sentence1'].strip(),
                'sentence2': row['sentence2'].strip(),
                'label'    : label.strip()
            })

    df = pd.DataFrame(rows)
    print(f'NLI4CT {split}: {len(df)} examples loaded')
    print(f'label distribution:\n{df["label"].value_counts().to_string()}')
    return df, split


# ─── feature builder ──────────────────────────────────────────────────────────

def build_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Build NLI feature vector from two embeddings.

    Args:
        emb_a : (N, dim) embeddings of sentence1
        emb_b : (N, dim) embeddings of sentence2

    Returns:
        (N, 4*dim) feature matrix
    """
    diff = np.abs(emb_a - emb_b)
    return np.hstack([emb_a, emb_b, diff])


class NLIMLP(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        return self.net(x)


# ─── main evaluation function ─────────────────────────────────────────────────

def evaluate(
    embedder,
    dataset     : str  = 'nli4ct',
    batch_size  : int  = 32,
    save_figures: bool = True,
    max_iter    : int  = 1000,
    classifier  : str  = 'mlp',
    mlp_epochs  : int  = 25,
    mlp_lr      : float = 1e-3,
    mlp_batch_size: int = 128,
):
    """
    Run NLI evaluation for one model.

    Since NLI4CT only has a train split in your folder,
    we do an 80/20 train/test split internally.

    Args:
        embedder    : any BaseEmbedder instance (already loaded)
        dataset     : 'nli4ct'
        batch_size  : passed to embedder.encode()
        save_figures: save plots to results/
        max_iter    : logistic regression max iterations
    """

    if dataset != 'nli4ct':
        raise ValueError(f'unknown dataset: {dataset}. choose nli4ct')

    ensure_nli_assets(dataset)

    # ── load data ──
    df, actual_split = load_nli4ct('train')

    # 80/20 split since we only have one file
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f'\ntrain: {len(train_df)}  test: {len(test_df)}')

    # ── encode sentences ──
    # encode() handles this for any model type:
    # word2vec   → mean of word vectors
    # transformer → mean pool or CLS
    # output always (N, dim)
    print(f'\nencoding train sentences...')
    t0 = time.time()

    train_emb_a = embedder.encode(train_df['sentence1'].tolist(), batch_size=batch_size)
    train_emb_b = embedder.encode(train_df['sentence2'].tolist(), batch_size=batch_size)

    print(f'encoding test sentences...')
    test_emb_a  = embedder.encode(test_df['sentence1'].tolist(), batch_size=batch_size)
    test_emb_b  = embedder.encode(test_df['sentence2'].tolist(), batch_size=batch_size)

    encode_time = time.time() - t0
    print(f'encoded in {encode_time:.1f}s — embedding dim: {train_emb_a.shape[1]}')

    # ── build features ──
    # [A, B, A-B, A*B] → (N, 4*dim)
    X_train = build_features(train_emb_a, train_emb_b)
    X_test  = build_features(test_emb_a,  test_emb_b)

    # encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_test  = le.transform(test_df['label'])

    print(f'\nfeature shape: {X_train.shape}')
    print(f'classes: {le.classes_}')

    # ── train classifier ──
    print(f'\ntraining {classifier} classifier...')
    t1  = time.time()
    if classifier == 'mlp':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NLIMLP(dim=train_emb_a.shape[1], num_classes=len(le.classes_)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_lr)
        loss_fn = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(train_emb_a.astype(np.float32)),
            torch.from_numpy(train_emb_b.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int64)),
        )
        train_loader = DataLoader(train_ds, batch_size=mlp_batch_size, shuffle=True)

        model.train()
        for epoch in range(mlp_epochs):
            running_loss = 0.0
            for batch_a, batch_b, batch_y in train_loader:
                batch_a = batch_a.to(device)
                batch_b = batch_b.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_a, batch_b)
                loss = loss_fn(logits, batch_y)
                if torch.isnan(loss):
                    print("NaN detected, stopping training")
                    break

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == mlp_epochs - 1:
                print(f'epoch {epoch + 1}/{mlp_epochs} | loss {running_loss / max(len(train_loader), 1):.4f}')

        model.eval()
        with torch.no_grad():
            test_a = torch.from_numpy(test_emb_a.astype(np.float32)).to(device)
            test_b = torch.from_numpy(test_emb_b.astype(np.float32)).to(device)
            y_pred = model(test_a, test_b).argmax(dim=1).cpu().numpy()
    else:
        raise ValueError(f'unknown classifier: {classifier}')

    train_time = time.time() - t1
    print(f'trained in {train_time:.1f}s')

    # ── evaluate ──
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    # majority baseline
    majority_label = train_df['label'].value_counts().index[0]
    majority_acc   = (test_df['label'] == majority_label).mean()

    total_time = encode_time + train_time

    result = {
        'model'           : embedder.name,
        'dataset'         : dataset,
        'accuracy'        : round(float(accuracy), 4),
        'majority_baseline': round(float(majority_acc), 4),
        'improvement'     : round(float(accuracy - majority_acc), 4),
        'per_class_f1'    : {
            cls: round(report[cls]['f1-score'], 4)
            for cls in le.classes_
        },
        'macro_f1'        : round(report['macro avg']['f1-score'], 4),
        'num_train'       : len(train_df),
        'num_test'        : len(test_df),
        'classifier'      : classifier,
        'runtime_sec'     : round(total_time, 2),
        'date'            : str(date.today()),
    }

    print(f'\n{"="*50}')
    print(f'  model            : {result["model"]}')
    print(f'  dataset          : {result["dataset"]}')
    print(f'  accuracy         : {result["accuracy"]:.4f}')
    print(f'  majority baseline: {result["majority_baseline"]:.4f}')
    print(f'  improvement      : +{result["improvement"]:.4f}')
    print(f'  macro F1         : {result["macro_f1"]:.4f}')
    print(f'  per class F1:')
    for cls, f1 in result['per_class_f1'].items():
        print(f'    {cls:<20}: {f1:.4f}')
    print(f'  runtime          : {result["runtime_sec"]}s')
    print(f'{"="*50}\n')

    # ── save results ──
    out_dir = RESULTS_DIR / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f'nli_{dataset}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'results saved to {result_path}')

    # ── save figures ──
    if save_figures:
        fig_dir = out_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[0], cmap='Blues'
        )
        axes[0].set_title('Confusion matrix')
        axes[0].set_ylabel('true label')
        axes[0].set_xlabel('predicted label')

        # per class F1 bar chart
        classes = list(result['per_class_f1'].keys())
        f1s     = list(result['per_class_f1'].values())
        axes[1].bar(classes, f1s)
        axes[1].axhline(y=result['macro_f1'], color='r',
                        linestyle='--', label=f'macro F1={result["macro_f1"]:.3f}')
        axes[1].set_title('Per-class F1 score')
        axes[1].set_ylabel('F1')
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        plt.suptitle(f'{embedder.name} | {dataset} | acc={accuracy:.4f}', fontsize=13)
        plt.tight_layout()

        fig_path = fig_dir / f'{dataset}_results.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f'figure saved to {fig_path}')

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True,
                        help='pubmedbert | word2vec | transformer etc.')
    parser.add_argument('--dataset',    default='nli4ct',
                        help='nli4ct (default: nli4ct)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iter',   type=int, default=1000)
    parser.add_argument('--classifier', default='mlp')
    parser.add_argument('--mlp_epochs', type=int, default=25)
    parser.add_argument('--mlp_lr', type=float, default=1e-3)
    parser.add_argument('--mlp_batch_size', type=int, default=128)
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
        batch_size = args.batch_size,
        max_iter   = args.max_iter,
        classifier = args.classifier,
        mlp_epochs = args.mlp_epochs,
        mlp_lr = args.mlp_lr,
        mlp_batch_size = args.mlp_batch_size,
    )
