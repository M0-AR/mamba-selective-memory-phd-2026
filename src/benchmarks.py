"""Memory-scaling + forgetting-curve math (video § compare the two memories)."""
from __future__ import annotations

import math


def transformer_numbers_per_layer(seq_len: int, d_model: int = 768) -> int:
    """2 lists of d_model per token per layer (K and V)."""
    return 2 * d_model * seq_len


def mamba_numbers_per_layer(d_model: int = 768, expand: int = 2, n_slots: int = 16) -> int:
    """Video: widen 768 -> 1536 channels x 16 slots = 24,576 (length-independent)."""
    return (d_model * expand) * n_slots


def crossover_length(d_model: int = 768, expand: int = 2, n_slots: int = 16) -> float:
    """L where transformer == mamba: 2*d*L = (d*expand)*N -> L = expand*N/2."""
    return expand * n_slots / 2


def ratio_at(seq_len: int, **kw) -> float:
    return transformer_numbers_per_layer(seq_len, kw.get("d_model", 768)) / mamba_numbers_per_layer(
        kw.get("d_model", 768), kw.get("expand", 2), kw.get("n_slots", 16)
    )


def fixed_decay(value: float, keep: float, n_steps: int) -> float:
    return value * (keep ** n_steps)


def selective_retention_after(value: float, step_filler: float, n_fillers: int) -> float:
    keep = math.exp(-step_filler)
    return value * (keep ** n_fillers)
