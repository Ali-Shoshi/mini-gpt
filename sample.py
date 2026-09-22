"""Generate text from a trained MiniGPT checkpoint."""

from __future__ import annotations

import argparse

import torch

from config import BEST_CHECKPOINT, DEVICE, TOKENIZER_PATH
from model import MiniGPT
from tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT.")
    parser.add_argument(
        "--prompt", default="Once upon a time", help="Text to begin generation with."
    )
    parser.add_argument("--tokens", type=int, default=400, help="Number of new byte tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Higher means more random.")
    parser.add_argument("--top-k", type=int, default=40, help="Keep only the k most likely tokens.")
    parser.add_argument(
        "--checkpoint", default=str(BEST_CHECKPOINT), help="Checkpoint file to load."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(DEVICE)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError as error:
        raise SystemExit(
            "No checkpoint was found. Run `python train.py` first, then try sampling again."
        ) from error

    model = MiniGPT().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise SystemExit("The prompt must contain at least one character.")

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
