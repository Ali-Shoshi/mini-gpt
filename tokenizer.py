"""Train, save, and use a byte-level BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import VOCAB_SIZE


class BPETokenizer:
    """A byte-level BPE tokenizer with explicit special-token IDs.

    BPE begins from byte-level pieces, then learns frequently occurring merges.
    That lets common words and word fragments become single tokens, making the
    model's fixed context window much more useful than a character tokenizer.
    """

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self._tokenizer.encode_batch(texts)]

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode([int(token_id) for token_id in token_ids])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def train(cls, corpus_paths: list[Path]) -> "BPETokenizer":
        """Train BPE directly from files, without loading a full corpus in RAM."""
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=cls.special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )
        tokenizer.train([str(path) for path in corpus_paths], trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. Run `python data.py --prepare` first."
            )
        return cls(Tokenizer.from_file(str(path)))
