"""Real-data + live-market verification (no API key, reproducible, offline-safe).

Public text: TinyShakespeare (Karpathy, MIT) via raw GitHub; fallback synthetic.
Live market: Stooq free daily CSV (no key) for AAPL + BTC; fallback synthetic walk.
All fetches have timeouts + deterministic fallback so `run_all.py` never crashes
offline, but logs whether live data was actually used (required for honest paper).

Selective analogy for markets:
  important day (|return| > k*sigma or earnings proxy) -> big step (write)
  filler day  -> tiny step (ignore)
Fixed baseline: classic EMA with one fixed keep (like S4 / moving average).
Task: retain last *regime* signal through noise, same as last-digit game.
"""
from __future__ import annotations

import io
import math
import random
import urllib.request
import numpy as np
from .memory import keep_from_step, update_scalar


TINYSHAKE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=2y"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365"


def fetch_url(url: str, timeout: int = 15) -> tuple[str | None, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore"), "live"
    except Exception as e:  # offline-safe
        return None, f"fallback ({type(e).__name__})"


def load_text_corpus(n_chars: int = 20000, seed: int = 0) -> tuple[str, str]:
    txt, mode = fetch_url(TINYSHAKE_URL)
    if txt is None or len(txt) < 5000:
        rng = random.Random(seed)
        words = ["the", "king", "queen", "7", "night", "3", "lord", "9", "love", "0"]
        txt = " ".join(rng.choice(words) for _ in range(n_chars // 5))
        return txt[:n_chars], "synthetic-fallback"
    return txt[:n_chars], "live-github-tinyshakespeare"


def load_stooq_daily(sym: str = "aapl.us", timeout: int = 15) -> tuple[np.ndarray, str]:
    """Returns close-price array. Verified live 2026-10-16 via Yahoo v8 + CoinGecko.

    sym: 'aapl.us' -> Yahoo AAPL; 'btcusd' -> CoinGecko BTC daily.
    Stooq was tried first (video-era best practice) but now requires JS
    verification (HTTP 403/JS challenge), so Yahoo/CoinGecko are primary.
    """
    import json as _json

    if sym.lower().startswith("btc"):
        txt, _ = fetch_url(COINGECKO_URL, timeout=timeout)
        if txt and '"prices"' in txt:
            try:
                px = np.array([p[1] for p in _json.loads(txt)["prices"]], dtype=float)
                if len(px) > 50:
                    return px, "live-coingecko-btcusd-365d"
            except Exception:
                pass
    else:
        ysym = sym.split(".")[0].upper()  # aapl.us -> AAPL
        txt, _ = fetch_url(YAHOO_URL.format(sym=ysym), timeout=timeout)
        if txt and '"close"' in txt:
            try:
                res = _json.loads(txt)["chart"]["result"][0]
                closes = res["indicators"]["quote"][0]["close"]
                px = np.array([c for c in closes if c is not None], dtype=float)
                if len(px) > 50:
                    return px, f"live-yahoo-{ysym}-2y"
            except Exception:
                pass
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 1000)
    # inject 5 regime jumps (digits) amid noise (letters)
    for i in [100, 300, 500, 700, 900]:
        rets[i] = 0.06
    px = 100 * np.exp(np.cumsum(rets))
    return px, "synthetic-fallback"


def market_selectivity_demo(prices: np.ndarray, k_sigma: float = 2.0):
    """Fixed EMA vs selective memory on returns. Returns dict of metrics."""
    rets = np.diff(np.log(prices))
    sigma = float(np.std(rets)) + 1e-12
    is_event = np.abs(rets) > k_sigma * sigma
    # Fixed EMA keep swept; selective: event step=5 (keep .0067), filler 0.01 (keep .99)
    best = {"keep": None, "mse": 1e99}
    for keep in [0.0, 0.5, 0.9, 0.95, 0.99]:
        h, se = 0.0, 0.0
        for x in rets:
            h = update_scalar(h, float(x), keep)
            se += (h - x) ** 2
        mse = se / len(rets)
        if mse < best["mse"]:
            best = {"keep": keep, "mse": mse}
    # Selective: track event-memory (last big move held through noise)
    h_sel, holds = 0.0, []
    last_event_val = 0.0
    for x, ev in zip(rets, is_event):
        step = 5.0 if ev else 0.01
        h_sel = update_scalar(h_sel, float(x), keep_from_step(step))
        if ev:
            last_event_val = float(x)
        holds.append(abs(h_sel - last_event_val))
    return {
        "n": len(rets),
        "sigma": sigma,
        "n_events": int(is_event.sum()),
        "fixed_best_keep": best["keep"],
        "fixed_best_mse": best["mse"],
        "selective_mean_hold_err": float(np.mean(holds)),
        "selective_final_mem": float(h_sel),
        "last_event": float(last_event_val),
    }
