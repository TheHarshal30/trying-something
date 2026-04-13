# Scratch Transformer Pipeline

This folder contains the contextual biomedical encoder trained from scratch on processed PubMed text.

The pipeline now has **two stages**:

1. masked language modeling (MLM) pretraining
2. optional unsupervised SimCSE-style contrastive fine-tuning

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
  --pooling_strategy last4_mean \
  --save_every_steps 5000 \
  --log_every_steps 100
```

### 3. Optional SimCSE sentence-encoder fine-tuning

This stage starts from the MLM checkpoint and improves sentence embeddings using unsupervised contrastive learning:

- same sentence + two dropout passes = positive pair
- other sentences in the batch = negatives

```bash
python models/transformer_scratch/train_simcse.py \
  --corpus_path training_data/pubmed/processed/pubmed_abstracts.txt \
  --input_dir models/transformer_scratch/weights/final \
  --output_dir models/transformer_scratch/weights/final_simcse \
  --batch_size 64 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --temperature 0.05 \
  --pooling_strategy last4_mean
```

## Pooling options

The embedder now supports three ablation-friendly pooling strategies:

- `cls`
- `mean`
- `last4_mean`

Recommended default:

- `last4_mean`

This usually gives more stable sentence embeddings than plain CLS pooling.

## Notes

- This training loop is GPU-aware and will use CUDA automatically when available.
- On an A100, `bfloat16` autocast should be used automatically when supported.
- Checkpoints are written under `models/transformer_scratch/weights/checkpoint_step_*`.
- Final model and tokenizer are written under `models/transformer_scratch/weights/final`.
- SimCSE checkpoints are written under `models/transformer_scratch/weights/final_simcse`.
- Both MLM and SimCSE loops include gradient clipping and NaN checks for safety.
- `embedding_config.json` stores the pooling strategy used at inference time.

## Evaluation integration

The embedder implementation is in `models/transformer_scratch/model.py`.

Available benchmark model names:

- `transformer_scratch`
- `transformer_scratch_simcse`

Example:

```bash
python evaluation/run_all.py --model transformer_scratch --task all
python evaluation/run_all.py --model transformer_scratch_simcse --task all
```

## Why this upgrade matters

MLM teaches the encoder to predict masked tokens from context.

SimCSE then reshapes the embedding space so that:

- two views of the same sentence are close
- different sentences are pushed apart

That makes the model much more useful as an actual sentence encoder for:

- STS
- entity linking retrieval
- downstream pair classification
