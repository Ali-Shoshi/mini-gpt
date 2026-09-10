"""Prepare scalable BPE-tokenized data and expose memory-mapped token shards."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config import (
    CONTEXT_LENGTH,
    CORPUS_PATH,
    DATA_DIR,
    DATA_MANIFEST_PATH,
    DEFAULT_DATA_URL,
    TINYSTORIES_REPO_ID,
    TINYSTORIES_TRAIN_FILE,
    TINYSTORIES_VALID_FILE,
    TOKENIZER_PATH,
    VOCAB_SIZE,
)
from tokenizer import BPETokenizer

RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"
TOKEN_DTYPE = np.dtype(np.uint16)
MANIFEST_VERSION = 2


@dataclass
class TokenShard:
    """A window into a memory-mapped token file; it never loads the full corpus."""

    path: Path
    offset: int
    count: int

    def __post_init__(self) -> None:
        self._tokens = np.memmap(
            self.path,
            dtype=TOKEN_DTYPE,
            mode="r",
            offset=self.offset * TOKEN_DTYPE.itemsize,
            shape=(self.count,),
        )

    def __len__(self) -> int:
        return self.count

    def window(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self._tokens[start : start + length], dtype=np.int64)


def _download_tinyshakespeare() -> Path:
    """Get a tiny starter corpus only when no local corpus already exists."""
    if CORPUS_PATH.exists():
        return CORPUS_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the Tiny Shakespeare starter corpus.")
    request = urllib.request.Request(
        DEFAULT_DATA_URL,
        headers={"User-Agent": "mini-gpt-from-scratch/1.0"},
    )
    with urllib.request.urlopen(request) as response, CORPUS_PATH.open("wb") as output:
        shutil.copyfileobj(response, output)
    return CORPUS_PATH


def download_tinystories() -> tuple[Path, Path]:
    """Download the 2.23 GB TinyStories V2 training text and validation text.

    This is an explicit action, so `python train.py` never silently starts a
    multi-gigabyte download. The dataset's license is CDLA-Sharing-1.0; review
    it before redistributing any derived model or data.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading TinyStories V2. This needs roughly 2.3 GB of download space.")
    train_path = Path(
        hf_hub_download(
            repo_id=TINYSTORIES_REPO_ID,
            repo_type="dataset",
            filename=TINYSTORIES_TRAIN_FILE,
            local_dir=RAW_DIR,
        )
    )
    validation_path = Path(
        hf_hub_download(
            repo_id=TINYSTORIES_REPO_ID,
            repo_type="dataset",
            filename=TINYSTORIES_VALID_FILE,
            local_dir=RAW_DIR,
        )
    )
    return train_path, validation_path


def default_source_paths() -> tuple[Path, Path | None]:
    """Prefer user-supplied raw files, then use the lightweight starter corpus."""
    raw_train = RAW_DIR / "train.txt"
    raw_validation = RAW_DIR / "validation.txt"
    if raw_train.exists():
        return raw_train, raw_validation if raw_validation.exists() else None
    return _download_tinyshakespeare(), None


def _encode_to_binary(source_path: Path, target_path: Path, tokenizer: BPETokenizer) -> int:
    """Encode a text file in batches, writing uint16 token IDs directly to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(".tmp")
    token_count = 0
    batch_size = 1_024
    file_size = source_path.stat().st_size

    with source_path.open("r", encoding="utf-8", errors="replace") as source, temporary_path.open(
        "wb"
    ) as target, tqdm(
        total=file_size, desc=f"Tokenizing {source_path.name}", unit="B", unit_scale=True
    ) as progress:
        lines: list[str] = []
        batch_bytes = 0

        def flush_lines() -> None:
            nonlocal token_count, batch_bytes, lines
            if not lines:
                return
            encoded_lines = tokenizer.encode_batch(lines)
            for token_ids in encoded_lines:
                np.asarray(token_ids, dtype=TOKEN_DTYPE).tofile(target)
                token_count += len(token_ids)
            progress.update(batch_bytes)
            batch_bytes = 0
            lines = []

        for line in source:
            lines.append(line)
            batch_bytes += len(line.encode("utf-8", errors="replace"))
            if len(lines) >= batch_size:
                flush_lines()
        flush_lines()

    temporary_path.replace(target_path)
    return token_count


def _source_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _manifest_is_current(train_path: Path, validation_path: Path | None) -> bool:
    if not DATA_MANIFEST_PATH.exists() or not TOKENIZER_PATH.exists():
        return False
    try:
        manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("version") != MANIFEST_VERSION or manifest.get("requested_vocab_size") != VOCAB_SIZE:
        return False
    if manifest.get("train_source") != _source_signature(train_path):
        return False
    expected_validation = _source_signature(validation_path) if validation_path else None
    if manifest.get("validation_source") != expected_validation:
        return False
    return all(Path(entry["path"]).exists() for entry in manifest["shards"].values())


def _shards_from_manifest() -> tuple[TokenShard, TokenShard]:
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    shards = manifest["shards"]
    train = TokenShard(Path(shards["train"]["path"]), shards["train"]["offset"], shards["train"]["count"])
    validation = TokenShard(
        Path(shards["validation"]["path"]),
        shards["validation"]["offset"],
        shards["validation"]["count"],
    )
    return train, validation


def prepare_token_shards(
    train_path: Path,
    validation_path: Path | None = None,
    force: bool = False,
) -> tuple[TokenShard, TokenShard]:
    """Train BPE and create reusable, memory-mapped train/validation token files."""
    if not force and _manifest_is_current(train_path, validation_path):
        print("Using existing BPE tokenizer and prepared token shards.")
        return _shards_from_manifest()

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training a {VOCAB_SIZE:,}-token BPE tokenizer from {train_path.name}.")
    tokenizer = BPETokenizer.train([train_path])
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer learned {tokenizer.vocab_size:,} tokens.")

    if validation_path is None:
        combined_path = PREPARED_DIR / "combined_tokens.bin"
        total_count = _encode_to_binary(train_path, combined_path, tokenizer)
        split_index = int(total_count * 0.95)
        shard_data = {
            "train": {"path": str(combined_path.resolve()), "offset": 0, "count": split_index},
            "validation": {
                "path": str(combined_path.resolve()),
                "offset": split_index,
                "count": total_count - split_index,
            },
        }
    else:
        train_binary = PREPARED_DIR / "train_tokens.bin"
        validation_binary = PREPARED_DIR / "validation_tokens.bin"
        train_count = _encode_to_binary(train_path, train_binary, tokenizer)
        validation_count = _encode_to_binary(validation_path, validation_binary, tokenizer)
        shard_data = {
            "train": {"path": str(train_binary.resolve()), "offset": 0, "count": train_count},
            "validation": {"path": str(validation_binary.resolve()), "offset": 0, "count": validation_count},
        }

    if shard_data["train"]["count"] <= CONTEXT_LENGTH + 1:
        raise ValueError("Training data is too small for the configured context length.")
    if shard_data["validation"]["count"] <= CONTEXT_LENGTH + 1:
        raise ValueError("Validation data is too small for the configured context length.")

    manifest = {
        "version": MANIFEST_VERSION,
        "requested_vocab_size": VOCAB_SIZE,
        "actual_vocab_size": tokenizer.vocab_size,
        "train_source": _source_signature(train_path),
        "validation_source": _source_signature(validation_path) if validation_path else None,
        "shards": shard_data,
    }
    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {shard_data['train']['count']:,} train and {shard_data['validation']['count']:,} validation tokens.")
    return _shards_from_manifest()


def load_token_shards() -> tuple[TokenShard, TokenShard]:
    """Return prepared shards, preparing the default source only when necessary."""
    train_path, validation_path = default_source_paths()
    return prepare_token_shards(train_path, validation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare BPE-tokenized training data.")
    parser.add_argument(
        "--dataset",
        choices=("tinyshakespeare", "tinystories"),
        default="tinyshakespeare",
        help="Corpus to prepare. TinyStories V2 downloads about 2.3 GB.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild tokenizer and token shards.")
    args = parser.parse_args()

    if args.dataset == "tinystories":
        train_path, validation_path = download_tinystories()
    else:
        train_path, validation_path = _download_tinyshakespeare(), None
    prepare_token_shards(train_path, validation_path, force=args.force)


if __name__ == "__main__":
    main()
