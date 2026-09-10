"""A small, readable decoder-only GPT language model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONTEXT_LENGTH, DROPOUT, N_EMBD, N_HEADS, N_LAYERS, VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    """Multi-head attention that can see only the current and earlier tokens."""

    def __init__(self) -> None:
        super().__init__()
        if N_EMBD % N_HEADS != 0:
            raise ValueError("N_EMBD must be divisible by N_HEADS.")

        self.n_heads = N_HEADS
        self.head_dim = N_EMBD // N_HEADS

        # One projection creates query, key, and value vectors for every token.
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.output_projection = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.residual_dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (batch, tokens, embedding) -> (batch, heads, tokens, head dimension)
        q = q.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch applies the lower-triangular causal mask internally and chooses
        # the most memory-efficient attention kernel available for this GPU.
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True,
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        return self.residual_dropout(self.output_projection(x))


class MLP(nn.Module):
    """The per-token feed-forward network inside every Transformer block."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and MLP, each protected by a residual path."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attention = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """A roughly 29M-parameter, decoder-only GPT trained by next-byte prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
        self.embedding_dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(N_LAYERS))
        self.final_norm = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

        self.apply(self._init_weights)

        # Weight tying saves parameters and aligns input/output token meanings.
        self.lm_head.weight = self.token_embedding.weight

        # Scale residual projections to keep a deep network stable at initialization.
        for name, parameter in self.named_parameters():
            if name.endswith("output_projection.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYERS))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(
                f"Received {sequence_length} tokens, but context is {CONTEXT_LENGTH}."
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """Generate one byte token at a time from an initial prompt."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        for _ in range(max_new_tokens):
            context = token_ids[:, -CONTEXT_LENGTH:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
