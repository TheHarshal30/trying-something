"""
Train a small BERT-style masked language model from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


class LineByLineIterableDataset(IterableDataset):
    def __init__(self, corpus_path: Path, tokenizer: BertTokenizer, max_length: int):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        with open(self.corpus_path, "r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx % num_workers != worker_id:
                    continue
                text = line.strip()
                if not text:
                    continue
                yield self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def save_checkpoint(output_dir: Path, model, tokenizer, step: int, metadata: dict) -> None:
    ckpt_dir = output_dir / f"checkpoint_step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    with open(ckpt_dir / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path",
        default="training_data/pubmed/processed/pubmed_abstracts.txt",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default="models/transformer_scratch/weights/tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        default="models/transformer_scratch/weights/final",
    )
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    corpus_path = Path(args.corpus_path)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus_path not found: {corpus_path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer_dir not found: {tokenizer_dir}")

    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_dir))
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_sequences = count_lines(corpus_path)
    steps_per_epoch = math.ceil(total_sequences / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"device                 : {device}")
    print(f"corpus path            : {corpus_path}")
    print(f"total sequences        : {total_sequences}")
    print(f"tokenizer vocab size   : {tokenizer.vocab_size}")
    print(f"hidden size            : {args.hidden_size}")
    print(f"layers                 : {args.num_hidden_layers}")
    print(f"heads                  : {args.num_attention_heads}")
    print(f"batch size             : {args.batch_size}")
    print(f"epochs                 : {args.epochs}")
    print(f"steps per epoch        : {steps_per_epoch}")
    print(f"total optimizer steps  : {total_steps}")

    dataset = LineByLineIterableDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_length=args.max_position_embeddings,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_cuda_amp = device.type == "cuda"
    use_bf16 = use_cuda_amp and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp and not use_bf16)

    optimizer_step = 0
    running_loss = 0.0
    train_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        progress = tqdm(
            dataloader,
            total=steps_per_epoch,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=use_cuda_amp,
            ):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % args.log_every_steps == 0:
                    avg_loss = running_loss / args.log_every_steps
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    print(
                        f"step {optimizer_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

                if optimizer_step % args.save_every_steps == 0:
                    metadata = {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch + 1,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    save_checkpoint(output_dir.parent, model, tokenizer, optimizer_step, metadata)

                if optimizer_step >= total_steps:
                    break

        epoch_elapsed = time.time() - epoch_start
        print(f"epoch {epoch + 1} finished in {epoch_elapsed:.2f}s")

    total_elapsed = time.time() - train_start
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "corpus_path": str(corpus_path),
        "tokenizer_dir": str(tokenizer_dir),
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
        "max_position_embeddings": args.max_position_embeddings,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mlm_probability": args.mlm_probability,
        "seed": args.seed,
        "total_sequences": total_sequences,
        "total_optimizer_steps": optimizer_step,
        "runtime_sec": round(total_elapsed, 2),
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"training finished in {total_elapsed:.2f}s")
    print(f"final model saved to {output_dir}")


if __name__ == "__main__":
    main()
