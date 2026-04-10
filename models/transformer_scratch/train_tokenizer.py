"""
Train a WordPiece tokenizer from scratch on the processed PubMed corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        default="training_data/pubmed/processed/pubmed_abstracts.txt",
        help="one normalized document per line",
    )
    parser.add_argument(
        "--output_dir",
        default="models/transformer_scratch/weights/tokenizer",
        help="directory for tokenizer files",
    )
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--limit_alphabet", type=int, default=1000)
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus_path not found: {corpus_path}")

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(output_dir))

    fast_tokenizer = BertTokenizerFast(
        vocab_file=str(output_dir / "vocab.txt"),
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(str(output_dir))

    print(f"tokenizer saved to {output_dir}")
    print(f"vocab size: {fast_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
