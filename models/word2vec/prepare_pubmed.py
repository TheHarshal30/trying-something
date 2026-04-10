"""
Prepare a biomedical text corpus for scratch Word2Vec training.

Primary intended source:
    PubMed XML / XML.GZ baseline dumps

Output format:
    One normalized abstract per line in a plain text file.
"""

from __future__ import annotations

import argparse
import gzip
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*")


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def iter_pubmed_abstracts(path: Path):
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            abstract_texts = []
            for abstract in elem.findall(".//Abstract/AbstractText"):
                text = "".join(abstract.itertext()).strip()
                if text:
                    abstract_texts.append(text)

            if abstract_texts:
                yield " ".join(abstract_texts)

            elem.clear()


def iter_plaintext(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def build_corpus(input_dir: Path, output_file: Path, min_tokens: int) -> tuple[int, int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_docs = 0
    num_kept = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix in {".txt", ".text"}:
                iterator = iter_plaintext(path)
            elif path.name.endswith(".xml") or path.name.endswith(".xml.gz"):
                iterator = iter_pubmed_abstracts(path)
            else:
                continue

            for raw_text in iterator:
                num_docs += 1
                normalized = normalize_text(raw_text)
                if not normalized:
                    continue
                if len(normalized.split()) < min_tokens:
                    continue
                out.write(normalized + "\n")
                num_kept += 1

    return num_docs, num_kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="training_data/pubmed/raw",
        help="directory containing PubMed XML/XML.GZ files or plain text files",
    )
    parser.add_argument(
        "--output_file",
        default="training_data/pubmed/processed/pubmed_abstracts.txt",
        help="one normalized abstract per line",
    )
    parser.add_argument("--min_tokens", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"input_dir not found: {input_dir}\n"
            "Place PubMed XML/XML.GZ files there first."
        )

    total_docs, kept_docs = build_corpus(
        input_dir=input_dir,
        output_file=output_file,
        min_tokens=args.min_tokens,
    )

    print(f"documents seen : {total_docs}")
    print(f"documents kept : {kept_docs}")
    print(f"output written : {output_file}")


if __name__ == "__main__":
    main()
