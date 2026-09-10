"""All experiment settings live here so they are easy to change."""

from pathlib import Path

import torch

# Reproducibility
SEED = 1337

# A byte-level BPE tokenizer learns useful word pieces such as "ing" and
# "tion" while still representing every UTF-8 character safely.
VOCAB_SIZE = 8_192
CONTEXT_LENGTH = 256

# This configuration has about 30.6 million trainable parameters.
N_LAYERS = 12
N_HEADS = 6
N_EMBD = 432
DROPOUT = 0.1

# Training
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 20_000
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
SAVE_INTERVAL = 500
SAMPLE_INTERVAL = 1_000
SAMPLE_PROMPT = "Once upon a time"
SAMPLE_TOKENS = 160
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Files and data. The default corpus is deliberately small so the whole pipeline
# can be verified quickly. Use `python data.py --dataset tinystories` to prepare
# the larger, 2.23 GB TinyStories V2 corpus before a serious training run.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CORPUS_PATH = DATA_DIR / "input.txt"
DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.json"
DATA_MANIFEST_PATH = ARTIFACT_DIR / "data_manifest.json"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

TINYSTORIES_REPO_ID = "roneneldan/TinyStories"
TINYSTORIES_TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"

# The code automatically falls back to CPU, but training there will be slow.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_TRAINING = False


def model_settings() -> dict[str, int | float]:
    """Settings saved inside checkpoints for reference."""
    return {
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_embd": N_EMBD,
        "dropout": DROPOUT,
    }
