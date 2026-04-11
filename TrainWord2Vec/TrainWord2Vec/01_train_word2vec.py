"""
01_train_word2vec.py
────────────────────
Train a Skip-gram Word2Vec model on PubMed abstracts.

Output
------
    models/word2vec/weights/word2vec.bin   (Word2Vec binary format)

Usage
-----
    python training/01_train_word2vec.py
    python training/01_train_word2vec.py --abstracts /path/to/abstracts.txt
    python training/01_train_word2vec.py --max_sentences 500000   # smoke-test

Requirements
------------
    pip install gensim==4.3.2 nltk tqdm requests
    python -m nltk.downloader punkt

PubMed download note
--------------------
PubMed Baseline FTP: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
Each .xml.gz file contains ~30,000 abstracts.  Full baseline is ~1100 files
(~35 GB compressed, ~4.5B tokens).  Download with:

    wget -r -nd -np -A "*.xml.gz" \
         ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ \
         -P data/pubmed/

Then point --abstracts at the folder or pass a pre-extracted .txt file
(one sentence per line, already tokenised / lowercased) with --pretokenised.
"""

import argparse
import logging
import os
import re
import gzip
import multiprocessing
from pathlib import Path

from gensim.models import Word2Vec
from tqdm import tqdm

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_OUT   = os.path.join('models', 'word2vec', 'weights', 'word2vec.bin')
DEFAULT_DIM   = 300
DEFAULT_WIN   = 5
DEFAULT_MIN   = 5          # min_count — prunes rare tokens
DEFAULT_EPOCH = 5
DEFAULT_NEG   = 10         # negative samples per positive
DEFAULT_ALPHA = 0.025      # initial learning rate


# ── sentence iterators ───────────────────────────────────────────────────────

class PretokenisedIterator:
    """
    Reads a plain-text file where each line is already one sentence,
    tokens separated by whitespace, already lowercased.
    Yields list[str] per line.
    """

    def __init__(self, path: str, max_sentences: int | None = None):
        self.path          = path
        self.max_sentences = max_sentences

    def __iter__(self):
        n = 0
        opener = gzip.open if self.path.endswith('.gz') else open
        with opener(self.path, 'rt', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield line.split()
                n += 1
                if self.max_sentences and n >= self.max_sentences:
                    break


class PubMedXMLIterator:
    """
    Streams sentences from a folder of PubMed .xml.gz baseline files.
    Extracts <AbstractText> content, lowercases, and does whitespace
    tokenisation + simple punctuation splitting.
    Yields list[str] per sentence.
    """
    _ABSTRACT_RE = re.compile(
        r'<AbstractText[^>]*>(.*?)</AbstractText>', re.DOTALL
    )
    _SENT_SPLIT  = re.compile(r'(?<=[.!?])\s+')
    _TOKEN_SPLIT = re.compile(r'[\s\-/]+')
    _STRIP_CHARS = re.compile(r'[^a-z0-9]')

    def __init__(self, folder: str, max_sentences: int | None = None):
        self.files         = sorted(Path(folder).glob('**/*.xml.gz'))
        self.max_sentences = max_sentences
        if not self.files:
            raise FileNotFoundError(f'No .xml.gz files found under {folder}')
        log.info(f'Found {len(self.files):,} PubMed XML files')

    def _tokenise(self, text: str):
        """Very lightweight tokeniser — lowercase + split on whitespace/punct."""
        text = text.lower()
        # Remove HTML entities and tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        tokens = []
        for raw in self._TOKEN_SPLIT.split(text):
            tok = self._STRIP_CHARS.sub('', raw)
            if len(tok) >= 2:          # drop single chars and empty strings
                tokens.append(tok)
        return tokens

    def __iter__(self):
        n = 0
        for xml_path in tqdm(self.files, desc='PubMed XML files', unit='file'):
            try:
                with gzip.open(xml_path, 'rt', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
            except Exception as e:
                log.warning(f'Could not read {xml_path}: {e}')
                continue

            for m in self._ABSTRACT_RE.finditer(content):
                abstract = m.group(1).strip()
                for sent in self._SENT_SPLIT.split(abstract):
                    tokens = self._tokenise(sent)
                    if len(tokens) >= 3:    # skip trivially short sentences
                        yield tokens
                        n += 1
                        if self.max_sentences and n >= self.max_sentences:
                            return


# ── training ─────────────────────────────────────────────────────────────────

def train(args):
    # ── build sentence iterator ──────────────────────────────────────────────
    if args.pretokenised:
        log.info(f'Using pre-tokenised file: {args.pretokenised}')
        sentences = PretokenisedIterator(args.pretokenised, args.max_sentences)
    else:
        log.info(f'Streaming PubMed XML from: {args.abstracts}')
        sentences = PubMedXMLIterator(args.abstracts, args.max_sentences)

    # ── train ────────────────────────────────────────────────────────────────
    log.info('Building vocab + training Word2Vec (Skip-gram) ...')
    log.info(
        f'  dim={args.dim}  window={args.window}  min_count={args.min_count}'
        f'  epochs={args.epochs}  negative={args.negative}'
        f'  workers={args.workers}'
    )

    model = Word2Vec(
        sentences   = sentences,
        vector_size = args.dim,
        window      = args.window,
        min_count   = args.min_count,
        sg          = 1,           # Skip-gram
        hs          = 0,           # negative sampling (not hierarchical softmax)
        negative    = args.negative,
        alpha       = args.alpha,
        epochs      = args.epochs,
        workers     = args.workers,
        seed        = 42,
        compute_loss= True,
    )

    log.info(
        f'Training complete — vocab size: {len(model.wv):,}'
        f'  loss: {model.get_latest_training_loss():.4f}'
    )

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.wv.save_word2vec_format(args.output, binary=True)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    log.info(f'Saved → {args.output}  ({size_mb:.1f} MB)')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Word2Vec on PubMed abstracts')

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        '--abstracts',
        default='data/pubmed',
        help='Folder of PubMed .xml.gz files (default: data/pubmed)',
    )
    src.add_argument(
        '--pretokenised',
        default=None,
        help='Path to a pre-tokenised .txt or .txt.gz file (one sentence per line)',
    )

    p.add_argument('--output',       default=DEFAULT_OUT,   help='Output .bin path')
    p.add_argument('--dim',          type=int, default=DEFAULT_DIM)
    p.add_argument('--window',       type=int, default=DEFAULT_WIN)
    p.add_argument('--min_count',    type=int, default=DEFAULT_MIN)
    p.add_argument('--epochs',       type=int, default=DEFAULT_EPOCH)
    p.add_argument('--negative',     type=int, default=DEFAULT_NEG)
    p.add_argument('--alpha',        type=float, default=DEFAULT_ALPHA)
    p.add_argument('--workers',      type=int, default=max(1, multiprocessing.cpu_count() - 2))
    p.add_argument('--max_sentences',type=int, default=None,
                   help='Cap on sentences to read — useful for smoke-testing')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
