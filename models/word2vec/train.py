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
import logging
import time
from pathlib import Path

from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from tfidf import compute_tfidf, save_tfidf

LOGGER = logging.getLogger(__name__)


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


class EpochLogger(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch = 0
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, model):
        self.epoch_start_time = time.time()
        LOGGER.info("epoch %d/%d started", self.epoch + 1, self.total_epochs)

    def on_epoch_end(self, model):
        elapsed = time.time() - self.epoch_start_time
        LOGGER.info(
            "epoch %d/%d finished in %.2f seconds",
            self.epoch + 1,
            self.total_epochs,
            elapsed,
        )
        self.epoch += 1


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()

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
    parser.add_argument("--model_type", choices=["word2vec", "fasttext"], default="fasttext")
    parser.add_argument("--vector_size", type=int, default=400)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--sg", type=int, default=1, help="1=skipgram, 0=cbow")
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--min_n", type=int, default=3, help="minimum character n-gram length for FastText")
    parser.add_argument("--max_n", type=int, default=6, help="maximum character n-gram length for FastText")
    parser.add_argument("--disable_tfidf", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"corpus_path not found: {corpus_path}\n"
            "Run models/word2vec/prepare_pubmed.py first."
        )

    sentence_count = count_lines(corpus_path)
    LOGGER.info("corpus path    : %s", corpus_path)
    LOGGER.info("sentence count : %d", sentence_count)
    LOGGER.info("model type     : %s", args.model_type)
    LOGGER.info("vector size    : %d", args.vector_size)
    LOGGER.info("window         : %d", args.window)
    LOGGER.info("min count      : %d", args.min_count)
    LOGGER.info("negative       : %d", args.negative)
    LOGGER.info("epochs         : %d", args.epochs)
    LOGGER.info("workers        : %d", args.workers)
    if args.model_type == "fasttext":
        LOGGER.info("min n          : %d", args.min_n)
        LOGGER.info("max n          : %d", args.max_n)
    LOGGER.info("starting %s training from scratch", args.model_type)

    sentences = CorpusIterator(corpus_path)
    total_start = time.time()
    model_cls = FastText if args.model_type == "fasttext" else Word2Vec
    model_kwargs = dict(
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
        callbacks=[EpochLogger(total_epochs=args.epochs)],
    )
    if args.model_type == "fasttext":
        model_kwargs["min_n"] = args.min_n
        model_kwargs["max_n"] = args.max_n
    model = model_cls(**model_kwargs)
    total_elapsed = time.time() - total_start

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "word2vec.bin"
    fasttext_path = output_dir / "fasttext.model"
    metadata_path = output_dir / "training_metadata.json"
    tfidf_path = output_dir / "tfidf_idf.json"

    model.wv.save_word2vec_format(weights_path, binary=True)
    if args.model_type == "fasttext":
        model.save(str(fasttext_path))
        LOGGER.info("fastText model saved: %s", fasttext_path)

    tfidf_terms = 0
    if not args.disable_tfidf:
        LOGGER.info("computing TF-IDF weights for weighted pooling")
        tfidf_start = time.time()
        idf = compute_tfidf(CorpusIterator(corpus_path))
        save_tfidf(idf, tfidf_path)
        tfidf_terms = len(idf)
        LOGGER.info("TF-IDF saved in %.2f seconds to %s", time.time() - tfidf_start, tfidf_path)

    metadata = {
        "corpus_path": str(corpus_path),
        "sentence_count": sentence_count,
        "model_type": args.model_type,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "sg": args.sg,
        "negative": args.negative,
        "min_n": args.min_n if args.model_type == "fasttext" else None,
        "max_n": args.max_n if args.model_type == "fasttext" else None,
        "sample": args.sample,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
        "vocab_size": len(model.wv),
        "tfidf_enabled": not args.disable_tfidf,
        "tfidf_terms": tfidf_terms,
        "runtime_sec": round(total_elapsed, 2),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info("training finished in %.2f seconds", total_elapsed)
    LOGGER.info("vocab size     : %d", len(model.wv))
    LOGGER.info("weights saved  : %s", weights_path)
    LOGGER.info("metadata saved : %s", metadata_path)


if __name__ == "__main__":
    main()
