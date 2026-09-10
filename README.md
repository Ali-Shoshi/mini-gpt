# Mini GPT from Scratch

A roughly 30.6M-parameter, decoder-only GPT built in readable PyTorch. It uses a byte-level BPE tokenizer so every major part of the language-model pipeline is visible: tokenization, embeddings, causal attention, Transformer blocks, next-token loss, mixed-precision training, checkpoints, and sampling.

## What this project is (and is not)

This is a real, trainable language model for learning. Tiny Shakespeare is deliberately small and only verifies that the pipeline works; it will produce Shakespeare-like text, not broad modern English. For a serious first run, prepare TinyStories V2: it is 2.23 GB of simple, clean English stories and is explicitly downloaded only when requested.

## Project map

```text
config.py      Settings for the model, training run, and file locations.
tokenizer.py   Trains, saves, and loads a byte-level BPE tokenizer.
data.py        Downloads optional data, tokenizes it into disk shards, and memory-maps batches.
model.py       GPT architecture: embeddings, attention, MLPs, and generation.
train.py       Training loop, evaluation, and checkpoint saving.
sample.py      Loads a checkpoint and generates text from a prompt.
```

## Run it

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Prepare the small starter corpus and its tokenizer for a quick smoke test:

```powershell
python data.py --dataset tinyshakespeare
```

For noticeably better English, prepare TinyStories V2 instead. This downloads approximately 2.3 GB and then tokenizes it into disk files:

```powershell
python data.py --dataset tinystories
```

Start training after preparation:

```powershell
python train.py
```

Checkpoints are written to `checkpoints/`; loss data and periodic generated samples are written to `logs/`. Stop safely with `Ctrl+C`; the last saved checkpoint remains available. Set `RESUME_TRAINING = True` in `config.py` to continue from `checkpoints/latest.pt`.

After a best checkpoint has been saved, generate text:

```powershell
python sample.py --prompt "ROMEO:" --tokens 400
```

Try more variation with a higher temperature:

```powershell
python sample.py --prompt "JULIET:" --tokens 400 --temperature 1.0
```

## Important settings

`config.py` begins with a 256-token context, micro-batch size 2, gradient accumulation of 2, 12 Transformer blocks, width 432, six attention heads, and an 8,192-token BPE vocabulary. This is about 30.6M parameters and should be a manageable starting point for a 4 GB RTX 3050 Ti.

For a quick smoke test, temporarily set `MAX_STEPS = 100`. Once it works, restore the default `20_000` steps. If you run out of GPU memory, reduce `BATCH_SIZE` from 2 to 1; gradient accumulation preserves a useful effective batch size.

## Using your own corpus

Put a UTF-8 plain-text training file at `data/raw/train.txt`. Optionally put a separate validation file at `data/raw/validation.txt`. Then run `python data.py --force` to rebuild the BPE tokenizer and memory-mapped token shards. The project automatically uses the raw files in preference to the starter corpus.
