"""Last-digit game + selective copying + associative recall.

Game definition (faithful to video):
- Each row = 20 tokens, each token is 0-9 or A-Z.
- Digit token carries its face value, e.g. '7' -> 7.
- Letter token carries (ord(L)-ord('A')) % 10, so A=0..J=9, K=0..T=9, U=0..Z=5.
  Letters are noise by construction.
- Task: output the LAST digit value in the row.

Fixed layer: one scalar h, one fixed keep for every token, input x = token value.
  Prediction = round(clip(h)) % 10 (nearest int 0..9).
Selective layer (hand-set, not trained):
  digit  -> step=20 (keep~0, wipe+write)
  letter -> step=1e-6 (keep~1, ignore)
  Prediction likewise.
"""
from __future__ import annotations

import random
import string
import numpy as np
from .memory import keep_from_step, update_scalar


def token_value(tok: str) -> int:
    if tok.isdigit():
        return int(tok)
    return (ord(tok.upper()) - ord("A")) % 10


def is_digit_tok(tok: str) -> bool:
    return tok.isdigit()


def random_row(length: int = 20, p_digit: float = 0.2, rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random()
    row = []
    for _ in range(length):
        if rng.random() < p_digit:
            row.append(str(rng.randint(0, 9)))
        else:
            row.append(rng.choice(string.ascii_uppercase))
    return row


def last_digit(row: list[str]) -> int | None:
    for tok in reversed(row):
        if tok.isdigit():
            return int(tok)
    return None


def rows_without_digit_are_skipped(n_rows: int, **kw) -> int:
    """Helper for reporting: count rows with no digit (task undefined)."""
    rng = kw.pop("rng", None)
    return 0  # filtering is done by caller; kept for API clarity


def predict_fixed(row: list[str], keep: float) -> int:
    h = 0.0
    for tok in row:
        h = update_scalar(h, float(token_value(tok)), keep)
    return int(round(float(np.clip(h, 0, 9)))) % 10


def predict_selective(
    row: list[str], step_digit: float = 20.0, step_letter: float = 1e-6
) -> int:
    h = 0.0
    for tok in row:
        step = step_digit if is_digit_tok(tok) else step_letter
        keep = keep_from_step(step)
        h = update_scalar(h, float(token_value(tok)), keep)
    return int(round(float(np.clip(h, 0, 9)))) % 10


def evaluate(predict_fn, n_rows: int, seed: int, length: int = 20, p_digit: float = 0.2):
    """Evaluate on fresh rows; skip rows with no digit. Returns (acc, n_used, n_total)."""
    rng = random.Random(seed)
    correct = 0
    used = 0
    for _ in range(n_rows):
        row = random_row(length, p_digit, rng)
        target = last_digit(row)
        if target is None:
            continue
        used += 1
        if predict_fn(row) == target:
            correct += 1
    return (correct / used if used else 0.0, used, n_rows)


def sweep_fixed_keeps(
    candidate_keeps: list[float], n_rows: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Grid-search fixed keep on train rows (video did 3100 settings on 1000 rows)."""
    best_keep, best_acc = 0.0, -1.0
    for k in candidate_keeps:
        acc, _, _ = evaluate(lambda r, k=k: predict_fixed(r, k), n_rows, seed)
        if acc > best_acc:
            best_acc, best_keep = acc, k
    return best_keep, best_acc


# ---- Harder variants (paper § selective copying / induction) ----

def selective_copy_row(
    vocab: list[str], n_signal: int = 4, n_filler: int = 12,
    rng: random.Random | None = None,
) -> tuple[list[str], list[str]]:
    """Signal tokens scattered among filler 'X' blocks. Target = signal in order."""
    rng = rng or random.Random()
    signal = [rng.choice(vocab) for _ in range(n_signal)]
    seq: list[str] = []
    positions = sorted(rng.sample(range(n_signal + n_filler), n_signal))
    si = 0
    for i in range(n_signal + n_filler):
        if i in positions:
            seq.append(signal[si])
            si += 1
        else:
            seq.append("X")  # filler
    return seq, signal


def associative_recall_row(
    n_pairs: int = 4, rng: random.Random | None = None
) -> tuple[list[str], str, str]:
    """Pairs k1 v1 k2 v2 ... then query kq -> must output vq (induction-head test)."""
    rng = rng or random.Random()
    keys = [f"K{i}" for i in range(n_pairs)]
    vals = [str(rng.randint(0, 9)) for _ in range(n_pairs)]
    seq: list[str] = []
    for k, v in zip(keys, vals):
        seq += [k, v]
    q = rng.choice(keys)
    seq.append(q)
    return seq, q, vals[keys.index(q)]
