"""
evaluation/eval_entity_linking.py

Evaluates any BaseEmbedder on entity linking tasks:
  - NCBI Disease     → KB: CTD_diseases.tsv
  - BC5CDR-d         → KB: CTD_diseases.tsv  (Disease entities only)
  - BC5CDR-c         → KB: CTD_chemicals.tsv (Chemical entities only)

Usage:
    python eval_entity_linking.py --model pubmedbert --dataset ncbi
    python eval_entity_linking.py --model pubmedbert --dataset bc5cdr_d
    python eval_entity_linking.py --model pubmedbert --dataset bc5cdr_c
    python eval_entity_linking.py --model pubmedbert --dataset all
"""

import os
import json
import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# ─── paths ────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / 'data'
RESULTS_DIR   = ROOT / 'results'

NCBI_DIR      = DATA_DIR / 'raw'  / 'ncbi_disease'
BC5CDR_DIR    = DATA_DIR / 'raw'  / 'bc5cdr'
DISEASE_KB    = DATA_DIR / 'lookups' / 'mesh' / 'CTD_diseases.tsv'
CHEMICAL_KB   = DATA_DIR / 'lookups' / 'mesh' / 'CTD_chemicals.tsv'


# ─── KB loader ────────────────────────────────────────────────────────────────

def load_kb(kb_path: Path, entity_type: str) -> dict[str, str]:
    """
    Load CTD KB file into {mesh_id: canonical_name} dict.
    entity_type: 'Disease' or 'Chemical'
    """
    if entity_type == 'Disease':
        col_names = [
            'DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition',
            'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings'
        ]
        name_col = 'DiseaseName'
        id_col   = 'DiseaseID'
    else:
        col_names = [
            'ChemicalName', 'ChemicalID', 'CasRN', 'Definition',
            'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms'
        ]
        name_col = 'ChemicalName'
        id_col   = 'ChemicalID'

    df = pd.read_csv(kb_path, sep='\t', comment='#', header=None,
                     names=col_names, on_bad_lines='skip')

    df['clean_id'] = df[id_col].apply(
        lambda x: x.split(':')[-1] if isinstance(x, str) else x
    )

    kb = {}
    for _, row in df.iterrows():
        mid  = row['clean_id']
        name = row[name_col]
        if isinstance(mid, str) and isinstance(name, str):
            kb[mid] = name.lower().strip()

            # also add synonyms as alternative surface forms
            if isinstance(row.get('Synonyms'), str):
                for syn in row['Synonyms'].split('|'):
                    syn = syn.strip().lower()
                    if syn:
                        kb.setdefault(mid, syn)

    print(f'loaded KB: {len(kb)} entries from {kb_path.name}')
    return kb


# ─── dataset loaders ──────────────────────────────────────────────────────────

def load_ncbi(split: str = 'test') -> list[dict]:
    """
    Load NCBI Disease corpus.
    Returns list of {text, mesh_id} dicts.
    split: 'train' | 'dev' | 'test'
    """
    file_map = {
        'train': NCBI_DIR / 'NCBItrainset_corpus.txt',
        'dev'  : NCBI_DIR / 'NCBIdevelopset_corpus.txt',
        'test' : NCBI_DIR / 'NCBItestset_corpus.txt',
    }
    mentions = []
    with open(file_map[split]) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                text     = parts[3]
                mesh_raw = parts[5]
                for mid in mesh_raw.split('|'):
                    mid = mid.strip()
                    if mid.startswith('D'):
                        mentions.append({'text': text, 'mesh_id': mid})
    print(f'NCBI {split}: {len(mentions)} mentions loaded')
    return mentions


def load_bc5cdr(split: str = 'test', entity_type: str = 'Disease') -> list[dict]:
    """
    Load BC5CDR and filter by entity_type.
    entity_type: 'Disease' | 'Chemical'
    split: 'train' | 'validation' | 'test'
    """
    file_map = {
        'train'     : BC5CDR_DIR / 'train.jsonl',
        'validation': BC5CDR_DIR / 'validation.jsonl',
        'test'      : BC5CDR_DIR / 'test.jsonl',
    }
    mentions = []
    with open(file_map[split]) as f:
        for line in f:
            row = json.loads(line)
            for ent in row['entities']:
                if ent['type'] != entity_type:
                    continue
                if not ent['normalized']:
                    continue   # skip NIL
                text   = ent['text'][0] if ent['text'] else ''
                mesh_id = ent['normalized'][0]['db_id']
                if text and mesh_id:
                    mentions.append({'text': text, 'mesh_id': mesh_id})
    print(f'BC5CDR-{"d" if entity_type=="Disease" else "c"} {split}: {len(mentions)} mentions loaded')
    return mentions


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(ranks: list[int], k_values: list[int] = [1, 5, 10]) -> dict:
    """
    Given a list of ranks (1-indexed, 0 = not found),
    compute Acc@k and MRR.
    """
    metrics = {}
    for k in k_values:
        metrics[f'acc@{k}'] = float(np.mean([1 if 0 < r <= k else 0 for r in ranks]))
    # MRR
    metrics['mrr'] = float(np.mean([1/r if r > 0 else 0 for r in ranks]))
    return metrics


# ─── main evaluation function ─────────────────────────────────────────────────

def evaluate(
    embedder,
    dataset    : str,
    split      : str = 'test',
    batch_size : int = 32,
    top_k      : int = 10,
    save_figures: bool = True
):
    """
    Run entity linking evaluation for one model + dataset combo.

    Args:
        embedder    : any BaseEmbedder instance (already loaded)
        dataset     : 'ncbi' | 'bc5cdr_d' | 'bc5cdr_c'
        split       : 'train' | 'dev'/'validation' | 'test'
        batch_size  : passed to embedder.encode()
        top_k       : max k for Acc@k
        save_figures: save plots to results/
    """

    # ── resolve dataset config ──
    if dataset == 'ncbi':
        mentions    = load_ncbi(split)
        kb_path     = DISEASE_KB
        entity_type = 'Disease'
        dataset_tag = 'ncbi'

    elif dataset == 'bc5cdr_d':
        bc_split    = 'validation' if split == 'dev' else split
        mentions    = load_bc5cdr(bc_split, entity_type='Disease')
        kb_path     = DISEASE_KB
        entity_type = 'Disease'
        dataset_tag = 'bc5cdr_d'

    elif dataset == 'bc5cdr_c':
        bc_split    = 'validation' if split == 'dev' else split
        mentions    = load_bc5cdr(bc_split, entity_type='Chemical')
        kb_path     = CHEMICAL_KB
        entity_type = 'Chemical'
        dataset_tag = 'bc5cdr_c'

    else:
        raise ValueError(f'unknown dataset: {dataset}. choose ncbi | bc5cdr_d | bc5cdr_c')

    # ── load KB ──
    kb = load_kb(kb_path, entity_type)
    kb_ids    = list(kb.keys())
    kb_names  = list(kb.values())

    # ── embed KB ──
    print(f'\nembedding KB ({len(kb_names)} terms)...')
    t0 = time.time()
    kb_embeddings = embedder.encode(kb_names, batch_size=batch_size)  # (KB_size, dim)
    kb_embed_time = time.time() - t0
    print(f'KB embedded in {kb_embed_time:.1f}s — shape: {kb_embeddings.shape}')

    # ── embed mentions ──
    mention_texts = [m['text'] for m in mentions]
    gold_ids      = [m['mesh_id'] for m in mentions]

    print(f'\nembedding {len(mention_texts)} mentions...')
    t0 = time.time()
    mention_embeddings = embedder.encode(mention_texts, batch_size=batch_size)  # (N, dim)
    mention_embed_time = time.time() - t0
    print(f'mentions embedded in {mention_embed_time:.1f}s')

    # ── cosine similarity + ranking ──
    print('\ncomputing similarities and ranks...')
    t0 = time.time()

    # process in chunks to avoid OOM on 8GB RAM
    chunk_size = 256
    ranks      = []
    top1_sims  = []

    for i in tqdm(range(0, len(mention_embeddings), chunk_size), desc='ranking'):
        chunk      = mention_embeddings[i : i + chunk_size]
        chunk_gold = gold_ids[i : i + chunk_size]

        # (chunk_size, KB_size)
        sims = cosine_similarity(chunk, kb_embeddings)

        for j, (sim_row, gold_id) in enumerate(zip(sims, chunk_gold)):
            # argsort descending
            sorted_indices = np.argsort(sim_row)[::-1][:top_k]
            top_ids        = [kb_ids[idx] for idx in sorted_indices]
            top1_sims.append(sim_row[sorted_indices[0]])

            # rank of correct answer (1-indexed, 0 = not in top_k)
            if gold_id in top_ids:
                rank = top_ids.index(gold_id) + 1
            else:
                rank = 0
            ranks.append(rank)

    rank_time = time.time() - t0
    print(f'ranking done in {rank_time:.1f}s')

    # ── compute metrics ──
    metrics = compute_metrics(ranks, k_values=[1, 5, 10])
    total_time = kb_embed_time + mention_embed_time + rank_time

    result = {
        'model'          : embedder.name,
        'dataset'        : dataset_tag,
        'split'          : split,
        **metrics,
        'total_mentions' : len(mentions),
        'found_in_top_k' : int(sum(1 for r in ranks if r > 0)),
        'not_found'      : int(sum(1 for r in ranks if r == 0)),
        'runtime_sec'    : round(total_time, 2),
        'date'           : str(date.today()),
    }

    print(f'\n{"="*50}')
    print(f'  model   : {result["model"]}')
    print(f'  dataset : {result["dataset"]}')
    print(f'  split   : {result["split"]}')
    print(f'  Acc@1   : {result["acc@1"]:.4f}')
    print(f'  Acc@5   : {result["acc@5"]:.4f}')
    print(f'  Acc@10  : {result["acc@10"]:.4f}')
    print(f'  MRR     : {result["mrr"]:.4f}')
    print(f'  runtime : {result["runtime_sec"]}s')
    print(f'{"="*50}\n')

    # ── save results ──
    out_dir = RESULTS_DIR / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f'entity_linking_{dataset_tag}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'results saved to {result_path}')

    # ── save figures ──
    if save_figures:
        fig_dir = out_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        # figure 1: Acc@k curve
        k_vals  = [1, 2, 3, 4, 5, 10]
        acc_vals = [compute_metrics(ranks, [k])[f'acc@{k}'] for k in k_vals]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(k_vals, acc_vals, marker='o')
        axes[0].set_title(f'Acc@k — {embedder.name} on {dataset_tag}')
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('accuracy')
        axes[0].set_xticks(k_vals)
        axes[0].grid(True)

        # figure 2: top-1 similarity score distribution
        axes[1].hist(top1_sims, bins=40, edgecolor='white')
        axes[1].set_title(f'Top-1 similarity distribution')
        axes[1].set_xlabel('cosine similarity')
        axes[1].set_ylabel('count')

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
    parser.add_argument('--model',   required=True,
                        help='model name: pubmedbert | word2vec | transformer etc.')
    parser.add_argument('--dataset', required=True,
                        help='ncbi | bc5cdr_d | bc5cdr_c | all')
    parser.add_argument('--split',   default='test',
                        help='train | dev | test (default: test)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # ── load the right embedder based on --model flag ──
    if args.model == 'pubmedbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'pubmedbert-local'))

    # team models will be added here as they are pushed
    # elif args.model == 'word2vec':
    #     from word2vec_embedder import Word2VecEmbedder
    #     embedder = Word2VecEmbedder()
    #     embedder.load(str(ROOT / 'models' / 'word2vec' / 'weights'))

    else:
        raise ValueError(f'unknown model: {args.model}')

    # ── run evaluation ──
    datasets = ['ncbi', 'bc5cdr_d', 'bc5cdr_c'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        evaluate(
            embedder   = embedder,
            dataset    = ds,
            split      = args.split,
            batch_size = args.batch_size,
        )
