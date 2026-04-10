# Scratch Transformer Pipeline

This folder contains the next model after the Word2Vec baseline: a small BERT-style encoder trained from scratch on processed PubMed text.

If you trained the tokenizer before commit `dc72038`, delete the old tokenizer directory and retrain it after pulling the latest code. The updated script saves a Hugging Face-compatible tokenizer more reliably on some server setups.

## Recommended architecture

Start with this configuration:

- hidden size: `384`
- layers: `6`
- attention heads: `6`
- intermediate size: `1536`
- max sequence length: `128`
- vocab size: `30000`

This is intentionally small enough to be realistic for an A100-backed course project while still being a proper contextual encoder.

## Training flow

### 1. Train tokenizer

```bash
python models/transformer_scratch/train_tokenizer.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --output_dir models/transformer_scratch/weights/tokenizer \
  --vocab_size 30000
```

### 2. Train MLM model from scratch

```bash
python models/transformer_scratch/train_mlm.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --tokenizer_dir models/transformer_scratch/weights/tokenizer \
  --output_dir models/transformer_scratch/weights/final \
  --hidden_size 384 \
  --num_hidden_layers 6 \
  --num_attention_heads 6 \
  --intermediate_size 1536 \
  --max_position_embeddings 128 \
  --batch_size 64 \
  --epochs 3 \
  --learning_rate 5e-4 \
  --save_every_steps 5000 \
  --log_every_steps 100
```

## Notes

- This training loop is GPU-aware and will use CUDA automatically when available.
- On an A100, `bfloat16` autocast should be used automatically when supported.
- Checkpoints are written under `models/transformer_scratch/weights/checkpoint_step_*`.
- Final model and tokenizer are written under `models/transformer_scratch/weights/final`.

## Evaluation integration

The embedder implementation is in `models/transformer_scratch/model.py`.

Once training is complete, we can register `transformer_scratch` in `evaluation/run_all.py` and benchmark it against:

- STS
- NLI
- Entity Linking
