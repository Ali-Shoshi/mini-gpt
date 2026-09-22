# Mini GPT from Scratch

A roughly 30.6M-parameter, decoder-only GPT built in readable PyTorch. It uses a byte-level BPE tokenizer so every major part of the language-model pipeline is visible: tokenization, embeddings, causal attention, Transformer blocks, next-token loss, mixed-precision training, checkpoints, and sampling.

## What this project is (and is not)

This is a real, trainable language model for learning. It uses TinyStories V2 by default: a 2.23 GB corpus of simple, clean English stories. You can also prepare your own UTF-8 plain-text corpus.

## Project map

```text
config.py      Settings for the model, training run, and file locations.
tokenizer.py   Trains, saves, and loads a byte-level BPE tokenizer.
data.py        Downloads optional data, tokenizes it into disk shards, and memory-maps batches.
model.py       GPT architecture: embeddings, attention, MLPs, and generation.
train.py       Training loop, evaluation, and checkpoint saving.
sample.py      Loads a checkpoint and generates text from a prompt.
```

## Setup

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell says that scripts are disabled, run this once in the same terminal, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Prepare TinyStories

TinyStories V2 is the default dataset. This explicitly downloads approximately 2.3 GB, then trains a BPE tokenizer and writes memory-mapped token shards to disk:

```powershell
python data.py --dataset tinystories
```

After preparation completes, `train.py` automatically uses TinyStories. You do not need to change any other dataset setting.

## Train

```powershell
python train.py
```

Checkpoints are written to `checkpoints/`; loss data and periodic generated samples are written to `logs/`.

```text
checkpoints/best.pt       Best validation-loss checkpoint
checkpoints/latest.pt     Latest periodic checkpoint
logs/loss.png             Training and validation loss graph
logs/metrics.csv          Loss and learning-rate measurements
logs/sample_step_*.txt    Periodic generated text
```

To resume from `checkpoints/latest.pt`, set this in `config.py` before running `python train.py` again:

```python
RESUME_TRAINING = True
```

`latest.pt` is saved every 500 steps, so let the run reach its first save before relying on resume.

## Generate text

After a best checkpoint has been saved, generate text:

```powershell
python sample.py --prompt "Once upon a time" --tokens 400
```

Try more variation with a higher temperature:

```powershell
python sample.py --prompt "A little girl found a mysterious box" --tokens 400 --temperature 1.0
```

## Important settings

`config.py` begins with a 256-token context, micro-batch size 2, gradient accumulation of 2, 12 Transformer blocks, width 432, six attention heads, and an 8,192-token BPE vocabulary. This is about 30.6M parameters and should be a manageable starting point for a 4 GB RTX 3050 Ti.

For a quick smoke test, temporarily set `MAX_STEPS = 100`. Once it works, restore the default `20_000` steps. If you run out of GPU memory, reduce `BATCH_SIZE` from 2 to 1; gradient accumulation preserves a useful effective batch size.

## Using your own corpus

Put a UTF-8 plain-text training file at `data/raw/train.txt`. Optionally put a separate validation file at `data/raw/validation.txt`. Then run `python data.py --force` to rebuild the BPE tokenizer and memory-mapped token shards. The project automatically uses the raw files in preference to the starter corpus.
