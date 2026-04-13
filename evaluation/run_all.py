"""
evaluation/run_all.py

Master script — runs all evaluations for a given model
across all 4 downstream tasks and builds leaderboard.csv

Usage:
    # run all tasks for pubmedbert
    python run_all.py --model pubmedbert

    # run specific task only
    python run_all.py --model pubmedbert --task entity_linking
    python run_all.py --model pubmedbert --task sts
    python run_all.py --model pubmedbert --task nli

    # run for team model (once they push)
    python run_all.py --model word2vec
    python run_all.py --model transformer
    python run_all.py --model transformer_umls
"""

import json
import argparse
import importlib.util
from pathlib import Path
from datetime import date

import pandas as pd

from assets import ensure_model, ensure_task_assets

# ─── paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'


# ─── embedder registry ────────────────────────────────────────────────────────
# add your team's models here as they are pushed to models/

def load_embedder(model_name: str):
    """
    Load and return the correct embedder for a given model name.
    This is the only place you need to add new models.
    """
    if model_name == 'pubmedbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(ensure_model('pubmedbert'))
        return embedder
    elif model_name == 'sapbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(ensure_model('sapbert'))
        embedder._name = 'sapbert'
        return embedder

    elif model_name == 'biobert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(ensure_model('biobert'))
        embedder._name = 'biobert'
        return embedder

    elif model_name == 'minilm':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(ensure_model('minilm'))
        embedder._name = 'minilm'
        return embedder

    elif model_name == 'word2vec':
        import sys
        sys.path.append(str(ROOT / 'models' / 'word2vec'))
        from model import Word2VecEmbedder
        embedder = Word2VecEmbedder()
        embedder.load(str(ROOT / 'models' / 'word2vec'))
        return embedder

    elif model_name == 'trainword2vec':
        model_dir = ROOT / 'TrainWord2Vec' / 'TrainWord2Vec' / 'models' / 'word2vec'
        model_file = model_dir / 'model.py'
        spec = importlib.util.spec_from_file_location('trainword2vec_model', model_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        embedder = module.Word2VecEmbedder()
        embedder._name = 'trainword2vec'
        embedder.load(str(model_dir))
        return embedder

    elif model_name == 'transformer_scratch':
        import sys
        sys.path.append(str(ROOT / 'models' / 'transformer_scratch'))
        from model import TransformerScratchEmbedder
        embedder = TransformerScratchEmbedder()
        embedder.load(str(ROOT / 'models' / 'transformer_scratch'))
        return embedder

    elif model_name == 'transformer_scratch_simcse':
        import sys
        sys.path.append(str(ROOT / 'models' / 'transformer_scratch'))
        from model import TransformerScratchEmbedder
        embedder = TransformerScratchEmbedder()
        embedder._name = 'transformer_scratch_simcse'
        embedder.load(str(ROOT / 'models' / 'transformer_scratch' / 'weights' / 'final_simcse'))
        return embedder

    # elif model_name == 'word2vec_umls':
    #     from word2vec_umls_embedder import Word2VecUMLSEmbedder
    #     embedder = Word2VecUMLSEmbedder()
    #     embedder.load(str(ROOT / 'models' / 'word2vec_umls' / 'weights'))
    #     return embedder

    # elif model_name == 'transformer':
    #     from transformer_embedder import TransformerEmbedder
    #     embedder = TransformerEmbedder()
    #     embedder.load(str(ROOT / 'models' / 'transformer' / 'weights'))
    #     return embedder

    # elif model_name == 'transformer_umls':
    #     from transformer_umls_embedder import TransformerUMLSEmbedder
    #     embedder = TransformerUMLSEmbedder()
    #     embedder.load(str(ROOT / 'models' / 'transformer_umls' / 'weights'))
    #     return embedder

    else:
        raise ValueError(
            f'unknown model: {model_name}\n'
            f'add it to the load_embedder() registry in run_all.py'
        )


# ─── leaderboard builder ──────────────────────────────────────────────────────

def build_leaderboard():
    """
    Read all result JSONs from results/ and compile into leaderboard.csv
    """
    rows = []

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for result_file in model_dir.glob('*.json'):
            with open(result_file) as f:
                result = json.load(f)

            task = result_file.stem  # e.g. entity_linking_ncbi

            row = {'model': model_name, 'task': task, 'date': result.get('date', '')}

            # entity linking metrics
            if 'acc@1' in result:
                row['acc@1']  = result['acc@1']
                row['acc@5']  = result['acc@5']
                row['mrr']    = result['mrr']

            # sts metrics
            if 'pearson_r' in result:
                row['pearson_r']  = result['pearson_r']
                row['spearman_r'] = result['spearman_r']

            # nli metrics
            if 'accuracy' in result:
                row['accuracy']          = result['accuracy']
                row['macro_f1']          = result['macro_f1']
                row['majority_baseline'] = result['majority_baseline']

            rows.append(row)

    if not rows:
        print('no results found yet — run evaluations first')
        return

    df = pd.DataFrame(rows).sort_values(['task', 'model'])
    leaderboard_path = RESULTS_DIR / 'leaderboard.csv'
    df.to_csv(leaderboard_path, index=False)
    print(f'\nleaderboard saved to {leaderboard_path}')
    print(df.to_string(index=False))
    return df


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='pubmedbert | word2vec | transformer | transformer_umls')
    parser.add_argument('--task',  default='all',
                        help='all | entity_linking | sts | nli  (default: all)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_simcse', action='store_true')
    parser.add_argument('--use_reranker', action='store_true')
    parser.add_argument('--use_hard_negatives', action='store_true')
    parser.add_argument('--use_tfidf', action='store_true')
    parser.add_argument('--download_only', action='store_true',
                        help='download datasets / KBs / model, then exit')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'  model : {args.model}')
    print(f'  task  : {args.task}')
    print(f'{"="*60}\n')

    ensure_task_assets(task=args.task, model_name=args.model)

    if args.download_only:
        print('downloads complete')
        return

    # load embedder once — reused across all tasks
    effective_model_name = args.model
    if args.use_simcse and args.model == 'transformer_scratch':
        effective_model_name = 'transformer_scratch_simcse'

    embedder = load_embedder(effective_model_name)
    if hasattr(embedder, 'use_tfidf'):
        embedder.use_tfidf = args.use_tfidf

    results = {}

    # ── entity linking ──
    if args.task in ('all', 'entity_linking'):
        from eval_entity_linking import evaluate as eval_el

        print('\n--- entity linking: NCBI ---')
        results['ncbi'] = eval_el(
            embedder=embedder, dataset='ncbi',
            batch_size=args.batch_size,
            use_reranker=args.use_reranker,
            use_hard_negatives=args.use_hard_negatives,
        )

        print('\n--- entity linking: BC5CDR-d ---')
        results['bc5cdr_d'] = eval_el(
            embedder=embedder, dataset='bc5cdr_d',
            batch_size=args.batch_size,
            use_reranker=args.use_reranker,
            use_hard_negatives=args.use_hard_negatives,
        )

        print('\n--- entity linking: BC5CDR-c ---')
        results['bc5cdr_c'] = eval_el(
            embedder=embedder, dataset='bc5cdr_c',
            batch_size=args.batch_size,
            use_reranker=args.use_reranker,
            use_hard_negatives=args.use_hard_negatives,
        )

    # ── sts ──
    if args.task in ('all', 'sts'):
        from eval_sts import evaluate as eval_sts

        print('\n--- STS: BIOSSES ---')
        results['biosses'] = eval_sts(
            embedder=embedder, dataset='biosses',
            batch_size=args.batch_size
        )

    # ── nli ──
    if args.task in ('all', 'nli'):
        from eval_nli import evaluate as eval_nli

        print('\n--- NLI: NLI4CT ---')
        results['nli4ct'] = eval_nli(
            embedder=embedder, dataset='nli4ct',
            batch_size=args.batch_size
        )
    

    # ── build leaderboard ──
    print('\n--- building leaderboard ---')
    build_leaderboard()

    print(f'\ndone. all results saved to results/{args.model}/')


if __name__ == '__main__':
    main()
