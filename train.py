"""Train MiniGPT to predict the next byte in a text corpus."""

from __future__ import annotations

import math
import random
import csv
import time
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_CHECKPOINT,
    CHECKPOINT_DIR,
    CONTEXT_LENGTH,
    DEVICE,
    EVAL_BATCHES,
    EVAL_INTERVAL,
    GRAD_ACCUM_STEPS,
    GRAD_CLIP,
    LATEST_CHECKPOINT,
    LEARNING_RATE,
    LOG_DIR,
    MAX_STEPS,
    MIN_LEARNING_RATE,
    RESUME_TRAINING,
    SAVE_INTERVAL,
    SAMPLE_INTERVAL,
    SAMPLE_PROMPT,
    SAMPLE_TOKENS,
    SEED,
    TOKENIZER_PATH,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    model_settings,
)
from data import TokenShard, load_token_shards
from model import MiniGPT, count_parameters
from tokenizer import BPETokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(tokens: TokenShard, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from a memory-mapped shard without loading the corpus into RAM."""
    last_start = len(tokens) - CONTEXT_LENGTH - 1
    if last_start <= 0:
        raise ValueError("The data split is shorter than the configured context length.")

    starts = torch.randint(0, last_start, (BATCH_SIZE,)).tolist()
    inputs = torch.from_numpy(
        np.stack([tokens.window(start, CONTEXT_LENGTH) for start in starts])
    )
    targets = torch.from_numpy(
        np.stack([tokens.window(start + 1, CONTEXT_LENGTH) for start in starts])
    )
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def learning_rate_at(step: int) -> float:
    """Linear warmup followed by cosine decay."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


def autocast_context(use_amp: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_tokens: TokenShard,
    validation_tokens: TokenShard,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Estimate train and validation loss on fresh random batches."""
    model.eval()
    losses: dict[str, float] = {}

    for split_name, split_tokens in (("train", train_tokens), ("validation", validation_tokens)):
        split_losses = []
        for _ in range(EVAL_BATCHES):
            inputs, targets = get_batch(split_tokens, device)
            with autocast_context(use_amp):
                _, loss = model(inputs, targets)
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def save_checkpoint(
    path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "model_settings": model_settings(),
        },
        path,
    )


@torch.no_grad()
def save_sample(model: MiniGPT, tokenizer: BPETokenizer, device: torch.device, step: int) -> None:
    """Write a fixed-prompt sample so language quality is visible over time."""
    was_training = model.training
    model.eval()
    prompt_ids = tokenizer.encode(SAMPLE_PROMPT)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(input_ids, max_new_tokens=SAMPLE_TOKENS, temperature=0.8)
    sample_path = LOG_DIR / f"sample_step_{step + 1:06d}.txt"
    sample_path.write_text(tokenizer.decode(generated[0].tolist()), encoding="utf-8")
    if was_training:
        model.train()


def write_training_logs(rows: list[dict[str, float | int]]) -> None:
    """Persist loss measurements as CSV and a compact plot after every evaluation."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "train_loss", "validation_loss", "learning_rate"))
        writer.writeheader()
        writer.writerows(rows)

    # Import here so core training still works if plotting is intentionally removed.
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    validation_losses = [row["validation_loss"] for row in rows]
    figure, axis = plt.subplots()
    axis.plot(steps, train_losses, label="train")
    axis.plot(steps, validation_losses, label="validation")
    axis.set(xlabel="training step", ylabel="cross-entropy loss", title="MiniGPT training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOG_DIR / "loss.png", dpi=150)
    plt.close(figure)


def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    use_amp = device.type == "cuda"

    if use_amp:
        torch.set_float32_matmul_precision("high")
        print(f"Training on: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA was not found; training on CPU will be very slow.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_tokens, validation_tokens = load_token_shards()
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    model = MiniGPT().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,} BPE tokens")
    print(f"Context length: {CONTEXT_LENGTH} BPE tokens")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    best_validation_loss = float("inf")
    if RESUME_TRAINING and LATEST_CHECKPOINT.exists():
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        best_validation_loss = checkpoint["validation_loss"]
        print(f"Resuming from step {start_step:,}.")

    model.train()
    start_time = time.perf_counter()
    metrics: list[dict[str, float | int]] = []
    progress = tqdm(range(start_step, MAX_STEPS), desc="Training", unit="step")

    for step in progress:
        current_learning_rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate

        optimizer.zero_grad(set_to_none=True)
        unscaled_loss_value = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            inputs, targets = get_batch(train_tokens, device)
            with autocast_context(use_amp):
                _, loss = model(inputs, targets)
                loss = loss / GRAD_ACCUM_STEPS
            unscaled_loss_value += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=f"{unscaled_loss_value:.4f}", lr=f"{current_learning_rate:.2e}")

        should_evaluate = step == start_step or (step + 1) % EVAL_INTERVAL == 0
        if should_evaluate:
            losses = estimate_loss(model, train_tokens, validation_tokens, device, use_amp)
            print(
                f"\nstep {step + 1:,} | train loss {losses['train']:.4f} | "
                f"validation loss {losses['validation']:.4f}"
            )
            metrics.append(
                {
                    "step": step + 1,
                    "train_loss": losses["train"],
                    "validation_loss": losses["validation"],
                    "learning_rate": current_learning_rate,
                }
            )
            write_training_logs(metrics)
            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                save_checkpoint(BEST_CHECKPOINT, model, optimizer, step, best_validation_loss)
                print(f"Saved new best checkpoint: {BEST_CHECKPOINT.name}")

        if (step + 1) % SAMPLE_INTERVAL == 0:
            save_sample(model, tokenizer, device, step)

        if (step + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(LATEST_CHECKPOINT, model, optimizer, step, best_validation_loss)

    final_losses = estimate_loss(model, train_tokens, validation_tokens, device, use_amp)
    save_checkpoint(LATEST_CHECKPOINT, model, optimizer, MAX_STEPS - 1, final_losses["validation"])
    metrics.append(
        {
            "step": MAX_STEPS,
            "train_loss": final_losses["train"],
            "validation_loss": final_losses["validation"],
            "learning_rate": learning_rate_at(MAX_STEPS - 1),
        }
    )
    write_training_logs(metrics)
    save_sample(model, tokenizer, device, MAX_STEPS - 1)
    elapsed_minutes = (time.perf_counter() - start_time) / 60
    print(f"\nTraining finished in {elapsed_minutes:.1f} minutes.")
    print(f"Final validation loss: {final_losses['validation']:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
