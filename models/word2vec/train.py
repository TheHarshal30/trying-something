"""
Train a scratch biomedical Word2Vec baseline from a local corpus file.

Expected input:
    one normalized document / sentence per line

Recommended corpus:
    PubMed abstracts prepared with prepare_pubmed.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gensim.models import Word2Vec


class CorpusIterator:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path

    def __iter__(self):
        with open(self.corpus_path, "r", encoding="utf-8") as handle:
            for line in handle:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        default="training_data/pubmed/processed/pubmed_abstracts.txt",
        help="plain text corpus: one tokenized document per line",
    )
    parser.add_argument(
        "--output_dir",
        default="models/word2vec/weights",
        help="directory where weights and metadata are saved",
    )
    parser.add_argument("--vector_size", type=int, default=300)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--sg", type=int, default=1, help="1=skipgram, 0=cbow")
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=float, default=1e-4)
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"corpus_path not found: {corpus_path}\n"
            "Run models/word2vec/prepare_pubmed.py first."
        )

    sentence_count = count_lines(corpus_path)
    print(f"corpus path    : {corpus_path}")
    print(f"sentence count : {sentence_count}")
    print("training Word2Vec from scratch...")

    sentences = CorpusIterator(corpus_path)
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "word2vec.bin"
    metadata_path = output_dir / "training_metadata.json"

    model.wv.save_word2vec_format(weights_path, binary=True)

    metadata = {
        "corpus_path": str(corpus_path),
        "sentence_count": sentence_count,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "sg": args.sg,
        "negative": args.negative,
        "sample": args.sample,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
        "vocab_size": len(model.wv),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"vocab size     : {len(model.wv)}")
    print(f"weights saved  : {weights_path}")
    print(f"metadata saved : {metadata_path}")


if __name__ == "__main__":
    main()
