"""
02_extract_umls_pairs.py
────────────────────────
Extract synonym pairs from UMLS MRCONSO.RRF and save:
  - data/umls_pairs.txt        — positive (anchor, positive) pairs, one per line
  - models/word2vec_umls/weights/umls_vocab.json  — {CUI: canonical_name, ...}

A "synonym pair" is any two distinct surface strings that share the same CUI
(Concept Unique Identifier) in UMLS.  These become the positive pairs for
NT-Xent contrastive training in step 03.

Usage
-----
    python training/02_extract_umls_pairs.py \\
        --mrconso /path/to/UMLS/META/MRCONSO.RRF \\
        [--lang ENG] \\
        [--max_pairs_per_cui 10] \\
        [--vocab_bin models/word2vec/weights/word2vec.bin]

UMLS Licence Note
-----------------
MRCONSO.RRF is part of the UMLS Metathesaurus.
Download requires a free UMLS licence from:  https://uts.nlm.nih.gov/uts/signup-login
File location inside the downloaded release:  META/MRCONSO.RRF

MRCONSO.RRF column layout (pipe-delimited):
    0  CUI     — Concept Unique Identifier  (e.g. C0027051)
    1  LAT     — Language                   (e.g. ENG)
    2  TS      — Term status
    3  LUI     — Lexical Unique Identifier
    4  STT     — String type
    5  SUI     — String Unique Identifier
    6  ISPREF  — Is preferred (Y/N)
    7  AUI     — Atom Unique Identifier
    8  SAUI    — Source atom identifier
    9  SCUI    — Source concept identifier
   10  SDUI    — Source descriptor identifier
   11  SAB     — Source abbreviation
   12  TTY     — Term type
   13  CODE    — Source code
   14  STR     — The actual string / surface form  ← we use this
   15  SRL     — Source restriction level
   16  SUPPRESS— Suppression flag  (N = not suppressed)
   17  CVF     — Content View Flag

We keep only rows where:
  - LAT == ENG  (or --lang value)
  - SUPPRESS == N  (not suppressed / deprecated atoms)

Then for each CUI we collect all unique surface strings and emit all
combinations (capped at --max_pairs_per_cui to avoid O(n^2) explosion on
CUIs with hundreds of synonyms).
"""

import argparse
import itertools
import json
import os
import random
import re
from collections import defaultdict

from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase + collapse whitespace.  Matches the tokeniser in step 01."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def load_word2vec_vocab(bin_path: str) -> set:
    """
    Return the vocabulary set from a trained Word2Vec .bin file without
    loading the full matrix.  Uses gensim KeyedVectors.
    """
    from gensim.models import KeyedVectors
    print(f'Loading W2V vocab from {bin_path} ...')
    wv = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    vocab = set(wv.key_to_index.keys())
    print(f'  W2V vocab size: {len(vocab):,}')
    return vocab


def tokens_in_vocab(text: str, vocab: set) -> bool:
    """Return True if at least one token of text is in the W2V vocab."""
    return any(t in vocab for t in text.split())


# ── main ─────────────────────────────────────────────────────────────────────

def extract(args):
    # ── read MRCONSO.RRF ────────────────────────────────────────────────────
    print(f'Reading MRCONSO.RRF from {args.mrconso} ...')

    # cui → set of normalised surface strings
    cui_to_strings: dict[str, set] = defaultdict(set)
    # cui → preferred string (ISPREF == Y, first one found)
    cui_to_pref:    dict[str, str] = {}

    with open(args.mrconso, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in tqdm(fh, desc='MRCONSO rows', unit=' rows'):
            cols = line.rstrip('\n').split('|')
            if len(cols) < 17:
                continue

            cui      = cols[0]
            lang     = cols[1]
            is_pref  = cols[6]   # 'Y' or 'N'
            suppress = cols[16]  # 'N' = active
            string   = cols[14]

            if lang     != args.lang:   continue
            if suppress != 'N':         continue

            norm = normalise(string)
            if not norm:
                continue

            cui_to_strings[cui].add(norm)

            if is_pref == 'Y' and cui not in cui_to_pref:
                cui_to_pref[cui] = norm

    print(f'Loaded {len(cui_to_strings):,} CUIs with ENG strings')

    # ── optionally filter to W2V vocab ──────────────────────────────────────
    w2v_vocab: set | None = None
    if args.vocab_bin:
        w2v_vocab = load_word2vec_vocab(args.vocab_bin)

    # ── build pairs ─────────────────────────────────────────────────────────
    pairs   = []
    skipped = 0

    for cui, strings in tqdm(cui_to_strings.items(), desc='Building pairs', unit='CUI'):
        string_list = sorted(strings)     # deterministic ordering

        if len(string_list) < 2:
            skipped += 1
            continue

        # All combinations up to cap
        combos = list(itertools.combinations(string_list, 2))
        if len(combos) > args.max_pairs_per_cui:
            combos = random.sample(combos, args.max_pairs_per_cui)

        for a, b in combos:
            # Both members must be non-empty and distinct after normalisation
            if a == b:
                continue
            # If a W2V vocab is available: skip pairs where neither member
            # has any in-vocab token (they would produce zero vectors anyway)
            if w2v_vocab is not None:
                if not tokens_in_vocab(a, w2v_vocab) or \
                   not tokens_in_vocab(b, w2v_vocab):
                    continue
            pairs.append((a, b))

    print(f'Pairs generated: {len(pairs):,}  (CUIs with <2 strings skipped: {skipped:,})')

    # ── shuffle and save pairs ───────────────────────────────────────────────
    random.seed(42)
    random.shuffle(pairs)

    os.makedirs(os.path.dirname(args.pairs_out), exist_ok=True)
    with open(args.pairs_out, 'w', encoding='utf-8') as fh:
        for a, b in pairs:
            fh.write(f'{a}\t{b}\n')
    print(f'Saved pairs → {args.pairs_out}')

    # ── build and save umls_vocab.json ───────────────────────────────────────
    # { CUI: canonical_name }  — uses preferred string when available,
    # otherwise alphabetically first string
    umls_vocab = {}
    for cui, strings in cui_to_strings.items():
        canonical = cui_to_pref.get(cui) or sorted(strings)[0]
        umls_vocab[cui] = canonical

    os.makedirs(os.path.dirname(args.vocab_out), exist_ok=True)
    with open(args.vocab_out, 'w', encoding='utf-8') as fh:
        json.dump(umls_vocab, fh, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(args.vocab_out) / 1024 / 1024
    print(f'Saved UMLS vocab → {args.vocab_out}  ({size_mb:.1f} MB, {len(umls_vocab):,} CUIs)')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Extract UMLS synonym pairs from MRCONSO.RRF')

    p.add_argument(
        '--mrconso',
        required=True,
        help='Path to UMLS META/MRCONSO.RRF',
    )
    p.add_argument(
        '--lang',
        default='ENG',
        help='Language filter (default: ENG)',
    )
    p.add_argument(
        '--max_pairs_per_cui',
        type=int,
        default=10,
        help='Max synonym pairs sampled per CUI (prevents O(n^2) blow-up on prolific CUIs)',
    )
    p.add_argument(
        '--vocab_bin',
        default=None,
        help='(Optional) Path to trained word2vec.bin — pairs where neither '
             'side has any in-vocab token will be filtered out',
    )
    p.add_argument(
        '--pairs_out',
        default=os.path.join('data', 'umls_pairs.txt'),
        help='Output path for tab-separated synonym pairs',
    )
    p.add_argument(
        '--vocab_out',
        default=os.path.join('models', 'word2vec_umls', 'weights', 'umls_vocab.json'),
        help='Output path for umls_vocab.json',
    )
    return p.parse_args()


if __name__ == '__main__':
    random.seed(42)
    extract(parse_args())
