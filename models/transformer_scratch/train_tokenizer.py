"""
Train a WordPiece tokenizer from scratch on the processed PubMed corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast

from preprocess import normalize_text

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
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--limit_alphabet", type=int, default=1000)
    parser.add_argument("--normalization_strategy", default="chemical",
                        choices=["chemical", "none", "basic"])
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus_path not found: {corpus_path}")

    normalized_corpus_path = output_dir / "_normalized_corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as src, open(
        normalized_corpus_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            normalized = normalize_text(line.strip(), strategy=args.normalization_strategy)
            if normalized:
                dst.write(normalized + "\n")

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[str(normalized_corpus_path)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(output_dir))
    tokenizer.save(str(output_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    fast_tokenizer.model_max_length = 128
    fast_tokenizer.save_pretrained(str(output_dir))

    tokenizer_config = {
        "do_lower_case": True,
        "model_max_length": 128,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "normalization_strategy": args.normalization_strategy,
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2)

    print(f"tokenizer saved to {output_dir}")
    print(f"vocab size: {fast_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
