from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, corpus_path: Path, tokenizer, max_length: int):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(corpus_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    self.examples.append(text)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]

    def collate_fn(self, batch: list[str]):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def pool_outputs(outputs, attention_mask: torch.Tensor, pooling_strategy: str) -> torch.Tensor:
    if pooling_strategy == "cls":
        return outputs.last_hidden_state[:, 0]
    if pooling_strategy == "mean":
        return masked_mean(outputs.last_hidden_state, attention_mask)
    if pooling_strategy == "last4_mean":
        hidden_states = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
        return masked_mean(hidden_states, attention_mask)
    raise ValueError(f"unknown pooling strategy: {pooling_strategy}")


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim.masked_fill_(mask, -1e9)

    labels = torch.arange(z1.size(0), device=z.device)
    labels = torch.cat([labels + z1.size(0), labels], dim=0)
    return F.cross_entropy(sim, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", default="training_data/pubmed/processed/pubmed_abstracts.txt")
    parser.add_argument("--input_dir", default="models/transformer_scratch/weights/final")
    parser.add_argument("--output_dir", default="models/transformer_scratch/weights/final_simcse")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--pooling_strategy", default="last4_mean", choices=["cls", "mean", "last4_mean"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    corpus_path = Path(args.corpus_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(input_dir))
    model = AutoModel.from_pretrained(str(input_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataset = TextDataset(corpus_path=corpus_path, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_cuda_amp = device.type == "cuda"
    use_bf16 = use_cuda_amp and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp and not use_bf16)

    print(f"device               : {device}")
    print(f"input checkpoint     : {input_dir}")
    print(f"output checkpoint    : {output_dir}")
    print(f"examples             : {len(dataset)}")
    print(f"steps per epoch      : {steps_per_epoch}")
    print(f"total optimizer step : {total_steps}")
    print(f"pooling strategy     : {args.pooling_strategy}")

    train_start = time.time()
    optimizer_step = 0
    running_loss = 0.0
    running_cos = 0.0
    running_norm = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        progress = tqdm(dataloader, total=steps_per_epoch, desc=f"simcse epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=use_cuda_amp,
            ):
                out1 = model(**batch, output_hidden_states=True, return_dict=True)
                out2 = model(**batch, output_hidden_states=True, return_dict=True)
                z1 = F.normalize(pool_outputs(out1, batch["attention_mask"], args.pooling_strategy), dim=1)
                z2 = F.normalize(pool_outputs(out2, batch["attention_mask"], args.pooling_strategy), dim=1)
                loss = nt_xent(z1, z2, temperature=args.temperature) / args.gradient_accumulation_steps

            if torch.isnan(loss):
                print("NaN detected, stopping training")
                break

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.gradient_accumulation_steps
            running_cos += F.cosine_similarity(z1, z2, dim=1).mean().item()
            running_norm += z1.norm(dim=1).mean().item()

            if step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                    avg_cos = running_cos / args.log_every_steps
                    avg_norm = running_norm / args.log_every_steps
                    progress.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        cos=f"{avg_cos:.4f}",
                        norm=f"{avg_norm:.4f}",
                    )
                    print(
                        f"step {optimizer_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"cos {avg_cos:.4f} | "
                        f"norm {avg_norm:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    running_cos = 0.0
                    running_norm = 0.0

                if optimizer_step % args.save_every_steps == 0:
                    ckpt_dir = output_dir.parent / f"simcse_checkpoint_step_{optimizer_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    with open(ckpt_dir / "embedding_config.json", "w", encoding="utf-8") as handle:
                        json.dump({"pooling_strategy": args.pooling_strategy}, handle, indent=2)

                if optimizer_step >= total_steps:
                    break

        print(f"epoch {epoch + 1} finished in {time.time() - epoch_start:.2f}s")

    total_elapsed = time.time() - train_start
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "embedding_config.json", "w", encoding="utf-8") as handle:
        json.dump({"pooling_strategy": args.pooling_strategy}, handle, indent=2)

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "objective": "simcse_unsupervised",
                "input_dir": str(input_dir),
                "corpus_path": str(corpus_path),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "pooling_strategy": args.pooling_strategy,
                "total_optimizer_steps": optimizer_step,
                "runtime_sec": round(total_elapsed, 2),
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    print(f"training finished in {total_elapsed:.2f}s")
    print(f"final model saved to {output_dir}")


if __name__ == "__main__":
    main()
