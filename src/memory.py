"""Core selective-memory recurrence (video transcript § memory rule).

Video rule (scalar, one slot):
    keep = exp(-step),  step in [0, +inf), keep in (0, 1]
    h_new = keep * h_old + (1 - keep) * x_new

- step ~ 0.01  -> keep ~ 0.99 (barely changes, ignore / filler)
- step = 5     -> keep ~ 0.0067 (almost replace)
- step = 20    -> keep ~ 2e-9  (wipe + write, important token)

This module is deliberately dependency-light (numpy only) so every
number in the paper can be hand-checked on a laptop.
"""
from __future__ import annotations

import math
import numpy as np


def keep_from_step(step: float) -> float:
    """keep = e^{-step}. step >= 0."""
    return math.exp(-step)


def step_from_keep(keep: float) -> float:
    """Inverse: step = -ln(keep)."""
    if keep <= 0 or keep > 1:
        raise ValueError("keep must be in (0, 1]")
    return -math.log(keep)


def update_scalar(h: float, x: float, keep: float) -> float:
    """One hand-checkable step: h' = keep*h + (1-keep)*x."""
    return keep * h + (1.0 - keep) * x


def run_sequence(xs: list[float] | np.ndarray, keep: float, h0: float = 0.0) -> list[float]:
    """Run fixed-keep recurrence over a sequence, return full trace."""
    h = h0
    trace = []
    for x in xs:
        h = update_scalar(float(h), float(x), keep)
        trace.append(h)
    return trace


def run_selective(
    xs: list[float],
    is_important: list[bool],
    step_important: float = 20.0,
    step_filler: float = 1e-6,
    h0: float = 0.0,
) -> list[float]:
    """Selective recurrence: important tokens wipe+write, filler ignored.

    This is the video's hand-set Mamba intuition:
      digit  -> big step   -> keep ~ 0 -> h = x
      letter -> tiny step  -> keep ~ 1 -> h unchanged
    """
    h = h0
    trace = []
    for x, imp in zip(xs, is_important):
        step = step_important if imp else step_filler
        keep = keep_from_step(step)
        h = update_scalar(float(h), float(x), keep)
        trace.append(h)
    return trace


class MultiSlotMemory:
    """N slots sharing one input-dependent step but with own timescales.

    keep_i = exp(-step * alpha_i), alpha_i > 0 per-slot rate.
    Slow slots (small alpha) hold long ago; fast slots hold just-now.
    Video example: slowest keeps 0.78/letter, fastest 0.14/letter.
    Gu & Dao paper uses N=16 slots per channel.
    """

    def __init__(self, alphas: list[float], h0: float = 0.0):
        self.alphas = np.array(alphas, dtype=float)
        self.h = np.full_like(self.alphas, float(h0))

    def step(self, x: float, step: float) -> np.ndarray:
        keeps = np.exp(-step * self.alphas)
        self.h = keeps * self.h + (1.0 - keeps) * float(x)
        return self.h.copy()

    def run(self, xs: list[float], steps: list[float]) -> np.ndarray:
        traj = []
        for x, s in zip(xs, steps):
            traj.append(self.step(x, s))
        return np.array(traj)


# Video's 4-slot illustration: per-letter keeps [0.78, ?, ?, 0.14]
# Solve alpha = -ln(keep_letter) / step_letter with step_letter = 1.0 for demo.
FOUR_SLOT_ALPHAS = [-math.log(k) for k in (0.78, 0.50, 0.28, 0.14)]
