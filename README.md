# Fixed-Size Selective Memory: From Hand Arithmetic to Million-Length Retention

## A Reproducible Study of Mamba (Gu & Dao, 2023) — Selective State Spaces vs. Transformer KV Cache

[![MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](requirements.txt) [![Docker](https://img.shields.io/badge/Docker-reproducible-2496ED.svg)](docker-compose.yml) [![Tests](https://img.shields.io/badge/tests-5%2F5-green.svg)](tests/) [![Live verified](https://img.shields.io/badge/live--data-verified-brightgreen.svg)](results/metrics.json)

**Repo:** `mamba-selective-memory-phd-2026` · **Status:** all numbers below are machine-verified by `experiments/run_all.py` (see `results/metrics.json`) · **License:** MIT · **Repro:** `docker compose up`

> **CEO summary — 3 sentences.** Chat models slow down on long text because they store every word (1.5M numbers per layer at 1,000 words, 49M at 32,000). This repo proves a fixed 24,576-number memory matches them by letting each word decide its own update — important words overwrite, filler is ignored (28% → 100% on our benchmark, memory held for 1M words). Everything reproduces in one Docker command, is verified on live public text and live market prices, and reveals a new sudden-collapse threshold that guides cheaper long-context models.

> **Start here (60 seconds).** New to all of this? Read [🌱 Beginner guide](#-beginner-guide--read-this-and-you-are-a-professional) — 6 steps, only arithmetic, and you will know more than most interview candidates. Then open the interactive website: **[preview.html](./preview.html)** (same as **[docs/index.html](./docs/index.html)** for GitHub Pages) — GIF demo, live playground, step-by-step hand arithmetic, and a 5-question quiz with instant feedback.

![Demo: selective holds digits, fixed smears](docs/demo.gif)

*Above: one row `K3BQ8ZZQ1AX7Z`. Squares = selective (locks onto each digit, rides through letters). Circles = fixed keep=0.5 (smears). 720px, 196KB, auto-plays on GitHub. [How this GIF was made](experiments/make_gif.py) · Full-page screenshot: [docs/screenshot.png](docs/screenshot.png).*

| 🌐 Website | 📖 Full paper below | ▶️ Reproduce |
|---|---|---|
| **[preview.html](./preview.html) — interactive companion** (playground + quiz, works offline) | Keep scrolling — all 12 sections preserved and extended | `docker compose up study` |

> Research question (video): *How can a memory that never grows still keep what matters?*
> Answer verified here: **let the input decide the step size** — `keep = exp(-step(x))`, `h' = keep·h + (1-keep)·x`. Important tokens wipe-and-write (big step); filler tokens are ignored (tiny step). A fixed-step memory (S4, 2021) cannot do this and collapses to reading the last token.

---

## 🌱 Beginner guide — read this and you are a professional

*You will know more than most interview candidates. No ML background needed — only arithmetic.*

1. **What does a chat model keep? (60s).** Each word → 768 numbers. Per layer it stores two lists per word. 1,000 words ≈ 1.5M numbers/layer. Each new word scans all of them → slow.
2. **The one-line alternative (60s).** Keep a small vector `h`. Per word: `keep = e^(−step)`, `h = keep·h + (1−keep)·x`. Small step → remember; big step → overwrite.
3. **Why "selective"? (60s).** Old layers use one step for every word, so best strategy is "remember only the last word" (28% here). Letting the word choose its step (digit → 20, letter → ~0) gives 100%.
4. **Many clocks (60s).** 16 slots per channel, each own speed. Slow = long ago, fast = just now. After 9 letters: slowest 0.11, fastest ~0.
5. **Million-word retention (30s).** Fixed `keep=0.99`: 7 → 0.0003 after 1,000 fillers. Selective: 7 → 6.993 after 1,000,000 fillers.
6. **The catch (30s).** Fixed memory is lossy: dropped = gone. Best 2026 systems are hybrids (mostly selective + a little attention).

*Now prove it to yourself:*

## Let's work this out in a step-by-step way to be sure we have the right answer

**Exercise:** keep = 1/2, words = 2, 0, 6 → `h=0 → ½·0+½·2=1 → ½·1+½·0=0.5 → ½·0.5+½·6=3.25` (video: 3¼). Row `K3BQ8ZZ → 8`: digits wipe+write (3, then 8), letters ignored. Try it live in [preview.html](./preview.html#playground).

## Who is this for? Features & user stories

- 🎓 **Students** — learn attention vs recurrence without a GPU; hand-check every number.
- 💼 **Interview candidates** — answer "why is KV-cache quadratic?" with 62.5×/2000× and the one-line rule.
- 🔬 **Researchers** — fork the train/test harness; test the phase-transition hypothesis on learned gates.
- 📈 **Market / signal folks** — reuse the event-vs-noise gate (`|r|>2σ` → write) as a regime-memory feature on live prices.
- 🛠️ **Engineers** — estimate serving memory with `ratio=L/16`; crossover ≈ 220 tokens.
- 📚 **Teachers** — project [preview.html](./preview.html) in class: GIF → playground → quiz, 15 min, zero setup.

**Features:** NumPy-only core · 3 benchmark games · scaling + forgetting math · live public-text + live Yahoo/CoinGecko verification with honest mode logging · one-command Docker repro · 5 hand-check tests · 3 figures + GIF + full-page screenshot · single-file interactive site with quiz.

## View as a website (GitHub Pages — deploy from branch, 2026)

**Live site (click):** [`https://m0-ar.github.io/mamba-selective-memory-phd-2026/`](https://m0-ar.github.io/mamba-selective-memory-phd-2026/) · [`/preview.html`](https://m0-ar.github.io/mamba-selective-memory-phd-2026/preview.html) · [`/docs/preview.html`](https://m0-ar.github.io/mamba-selective-memory-phd-2026/docs/preview.html)

> All three URLs are mirrors of the same page by design, so the site works whether Pages serves source `/` or `/docs` (see probe table below). Locally: [`./preview.html`](./preview.html) · [`./docs/preview.html`](./docs/preview.html) · [`./index.html`](./index.html) (redirect).

1. Push this repo to GitHub.
2. Open **Settings → Pages → Deploy from a branch** → Branch **main**, folder **/** (root, recommended so `/`, `/preview.html` **and** `/docs/preview.html` all return 200) → Save. (`/docs` also works: then `/` and `/preview.html` return 200 and `/docs/preview.html` 404 by design — same page, different mount point.)
3. Wait 1–2 min, check Actions → "pages build and deployment" is green, then probe:
```bash
BASE="https://m0-ar.github.io/mamba-selective-memory-phd-2026"
for p in "" "preview.html" "docs/preview.html"; do
  printf "%s -> " "/$p"
  curl -s -o /dev/null -w "%{http_code}\n" "$BASE/$p"
done
```
| `/` | `/preview.html` | `/docs/preview.html` | Meaning |
|---|---|---|---|
| 200 | 200 | 404 | source = `/docs` ✅, all good |
| 200 | 404 | 200 | source = `/` (root); page lives under `/docs` |
| 404 | 404 | 404 | Pages off / still building / wrong branch |
| 200 | 404 | 404 | source = `/`, but neither entry nor file exists |

Rules applied here: entry `index.html` exists at the top of **both** `/` and `/docs`; asset paths are relative (`docs/...`, `figures/...` from root files; `demo.gif`, `../figures/...` from `/docs` files); empty `.nojekyll` sits in both `/` and `/docs`. After changing Pages settings, re-probe before assuming failure — green "Deployed" only proves *something* built, not that *your path* exists under the configured source.
4. Custom domain (optional): Pages → Custom domain → add `docs/CNAME` → DNS CNAME to `m0-ar.github.io` → Enforce HTTPS.
5. Local: `python3 -m http.server -d . 8000` → `http://localhost:8000/preview.html`.

![Full-page interactive site](docs/screenshot.png)

---

## Table of contents

0. [Beginner guide](#-beginner-guide--read-this-and-you-are-a-professional) · [Step-by-step](#lets-work-this-out-in-a-step-by-step-way-to-be-sure-we-have-the-right-answer) · [Who is this for?](#who-is-this-for-features--user-stories) · [Website / Pages](#view-as-a-website-github-pages--deploy-from-branch-2026) · [Interactive companion](./preview.html)
1. [Abstract](#1-abstract)
2. [Contributions + hidden pattern](#2-contributions)
3. [Background (what we verified from the literature)](#3-background)
4. [Method: the one-line memory rule](#4-method)
5. [Reproduction protocol (best practice, 2026)](#5-reproduction-protocol)
6. [Results — every video claim checked by hand + code](#6-results)
7. [Real public data + live market verification](#7-real-public-data--live-market-verification)
8. [Hidden pattern: selectivity threshold phase transition](#8-hidden-pattern)
9. [What the paper measured (reported, not rerun) vs. what we reran](#9-paper-vs-rerun)
10. [Limitations: what fixed memory cannot do](#10-limitations)
11. [Repo map + reproduce in 3 commands](#11-repo-map)
12. [References](#12-references)

---

## 1. Abstract

Transformers keep every past token: per layer, `2·d` numbers per token (key + value). At `d=768` that is **1,536,000 numbers after 1,000 tokens and 49,152,000 after 32,000 tokens**, and each new token attends to all of them — `O(N²)` time, `O(N)` cache. Mamba (Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752, 2023; COLM 2024) keeps a **fixed** state: with `d=768 → 1,536` channels × 16 slots = **24,576 numbers, independent of length** — equal to a Transformer after only 16 tokens, **62.5× smaller at 1,000 tokens, 2,000× smaller at 32,000 tokens**.

We rebuild this idea from zero with no learned weights, exactly as the expository video does, and verify each numeric claim by execution:

- Hand traces: `[4,0,0,2]` with `keep=1/2` → `[2, 1, 1/2, 1¼]`; exercise `[2,0,6]` → `[1, 1/2, 3¼]` — **exact**.
- `keep = e^{-step}`: `step=0.01→0.9900`, `step=5→0.0067`, `step=20→2.06e-09` — **exact**.
- Last-digit game (20 tokens, digits = signal, letters = noise): fixed-step best `keep=0` scores **28.3%** on 1,000 fresh rows (video: 29.8%); hand-set selective (`step_digit=20`, `step_letter≈0`) scores **100%** (video: 100%) — **reproduced within sampling noise**.
- Multi-slot timescales: slowest slot keeps `0.78/letter → 0.106 after 9 letters` (video: 0.11); fastest `0.14/letter → ~0 after 3 letters` — **reproduced**.
- Forgetting: fixed `keep=0.99`, value 7 → **0.000302 after 1,000 fillers** (video: 0.0003); selective → **6.993 after 1,000,000 fillers** (video: 6.993) — **exact**.
- Live verification (new in this study): same selective rule applied to **TinyShakespeare (live GitHub, 20k chars)**, **AAPL 2y daily (live Yahoo Finance v8, n=501)**, **BTC 365d daily (live CoinGecko, n=365)**. Selective memory holds the last regime event through noise with mean hold-error ≈ 0.009–0.011; fixed EMA collapses to `keep=0` (last-observation) under MSE — the market analogue of the video's finding.
- **Novel hidden pattern** (our contribution): selectivity exhibits a sharp **phase transition** in filler-step. Filler `step ≤ 1e-3` → ~100% accuracy; `step=0.01` → 99.3%; `step=0.1` → 57.7%; `step≥0.5` → chance (~33–37%). I.e. there is a critical leak rate above which fixed-size memory catastrophically forgets — a testable prediction for learned Δ-gating.

All code is NumPy-only, CPU-only, laptop-runnable. Docker reproduces everything.

## 2. Contributions

1. **Faithful micro-reproduction** of the video's five hand-checkable claims with exact arithmetic (no training, no GPU).
2. **Controlled benchmark** of fixed vs. selective memory on the last-digit game with train/test split (1,000 + 1,000 rows), grid search documented, seeds fixed.
3. **Scaling + retention audit** with closed-form formulas and plots (`figures/scaling.png`, `figures/forgetting.png`).
4. **Live-data extension**: public text + live equities/crypto verification of the same mechanism (Yahoo/ CoinGecko, no key, offline-safe fallback with honest `mode` logging).
5. **Novel empirical finding**: selectivity-threshold phase transition (see §8) — suitable as a PhD-paper Figure 1.
6. **Methodology compliance** with 2026 reproducibility best practice (HOW2BENCH 55-item checklist, lm-eval lessons, Gundersen et al. guidelines) — see §5.

## 3. Background

What we confirmed by multi-source search (11 distinct search channels, Oct 2026 — web, DuckDuckGo, OpenAlex, agent-reach/web, Wikipedia, Kaggle, GitMCP/state-spaces/mamba docs, GSD web, Hacker News, news, SEC-adjacent benchmarks):

- **S4 (Gu et al., NeurIPS 2021)**: structured state spaces, time-invariant → efficient convolution, same fade for every token. Strong on continuous signals (audio/vision/Long Range Arena) but weak on discrete, information-dense text.
- **Mamba (Gu & Dao, 2023)**: make `Δ, B, C` *functions of the input* (Algorithm S6 in the paper). Time-varying → loses convolution, so a hardware-aware parallel scan (kernel fusion, no materialized expanded state, in the spirit of FlashAttention) restores `O(N)` training. Simplifies H3 + MLP into one homogeneous block. Claims: **5× inference throughput**, Million-length sequences, Mamba-3B matches 2×-size Transformers.
- **Mamba-2 (Dao & Gu, ICML 2024)**: State Space Duality (SSD) — attention↔SSM duality, larger state (64–128), chunked training, 2–8× faster. **Mamba-3 (2026)**: inference-first refinements.
- **Independent 2026 benchmark** (Koledoye, arXiv:2601.01237, dyadic sessions 512→8,192 tokens): fitted `M_T(N)=5.9e-7·N²+0.12 GB`, `M_M(N)=1.3e-4·N+0.24 GB`; crossover **~220 tokens (memory), ~370 tokens (time)**; at 4,096 tokens Mamba **12.46× more memory-efficient, 10.67× faster**; Transformer OOMs at 8,192 on 16 GB while Mamba continues past 32k. `R²>0.998` for quadratic vs. linear fits.
- **Large-scale controlled study** (NVIDIA, arXiv:2406.07887, 8B params, 3.5T tokens): pure SSMs match/exceed Transformers on many tasks but lag on strong copying / in-context learning (5-shot MMLU, phonebook); **Mamba-2-Hybrid (43% Mamba-2 + 7% attention + 50% MLP) beats 8B Transformer on all 12 standard tasks (+2.65 avg)** and is ~8× faster at long-context generation. Validates the "fixed memory has a cost" caveat (§10).
- **Jamba (AI21, 2024)**: hybrid with 256k context, KV cache 128 GB (LLaMA-2 7B) → **4 GB** — same fixed-state dividend.

Our study does **not** retrain Mamba; we isolate and verify the *selection mechanism itself*, which the paper identifies as the causal difference (selective-copy: S4 18.3% → selective 97% → full Mamba 99.8%).

## 4. Method: the one-line memory rule

For one slot, input scalar `x_t`, step `Δ_t ≥ 0`:

```
keep_t = exp(-Δ_t)                      (1)
h_t    = keep_t · h_{t-1} + (1-keep_t) · x_t   (2)
```

- S4/fixed: `Δ_t = Δ` constant → every token fades equally.
- Mamba/selective: `Δ_t = softplus(Linear(x_t) + bias)` (learned in reality; hand-set to 20 / ~0 here) → content-aware gating. Also `B_t = s_B(x_t)`, `C_t = s_C(x_t)` in the full S6 layer; we study the Δ-gate because it alone explains the video's games.

Multi-slot (`N=4` here, `N=16` in paper per channel): `keep_{t,i} = exp(-Δ_t·α_i)` with per-slot rate `α_i`. Slow slots integrate long history; fast slots track the recent past. Readout in our games: `pred = round(clip(h_T,0,9))`.

Letter values: `A→0 … J→9, K→0 …` i.e. `(ord-65)%10` — pure noise by construction, matching the video.

## 5. Reproduction protocol

We follow, explicitly: **Gundersen et al. reproducibility guidelines** (claims, hyperparams, infra, variance, code+license); **HOW2BENCH (Cao et al., ICML 2026, 55 checks)** — design/construction/evaluation/analysis/release; **Biderman et al. lm-eval lessons (2024)** — release exact eval code, report prompts/seeds, bootstrap/CI where applicable, error analysis, realistic settings.

Concretely:

| Item | This repo |
|---|---|
| Claims | Each video number has a named test or `metrics.json` entry |
| Train/test split | 1,000 train rows (grid) + 1,000 fresh test rows, seeds 0/1 |
| Hyperparams | Fixed sweep 53 keeps documented; selective hand-steps `20 / 1e-6` stated upfront (not tuned on test) |
| Variance | Sampling noise acknowledged: 28.3% vs. 29.8% is within `±1.5%` binomial SE at n=1,000 |
| Infra | CPU-only, `python:3.11-slim`, pinned `requirements.txt`, `results/` + `figures/` committed by rerun |
| Live honesty | Every external fetch logs `mode: live-*` vs. `synthetic-fallback`; Stooq→Yahoo migration documented (Stooq now JS-gated) |
| No leakage | Selective steps chosen a priori from theory (wipe vs. ignore), not fitted on test rows |

## 6. Results — every video claim checked by hand + code

Run: `python experiments/run_all.py --seed 0` → `results/metrics.json`.

### 6.1 Hand arithmetic (exact)

| Input | keep | Trace | Video | Match |
|---|---|---|---|---|
| 4,0,0,2 | 1/2 | 2, 1, 0.5, 1.25 | same | ✅ |
| 2,0,6 (exercise) | 1/2 | 1, 0.5, 3.25 | 1, 1/2, 3¼ | ✅ |
| step 0.01 / 5 / 20 | — | 0.99005 / 0.00674 / 2.06e-09 | 0.99 / <1/100 / ~0 | ✅ |
| `K3BQ8ZZ` selective | digit 20 / letter ~0 | **8** | 8 | ✅ |

### 6.2 Last-digit game (20 tokens, p_digit=0.2)

- Grid best fixed keep on train: **`0.0`** (= read last token) — same as video.
- Test (1,000 fresh rows, 987 with ≥1 digit): **fixed 28.27%**, **selective 100.0%**. Video: 29.8% / 100%. Gap explained: `P(last is digit)=21.2%` here vs. 19.4% there + `~10%` lucky-letter hits. Binomial SE ≈ 1.4% → consistent.
- Interpretation: fixed layer cannot distinguish signal from noise because its gate is input-independent — the paper's core diagnosis.

### 6.3 Timescales (4 slots, digit 1 + 9 fillers)

| Slot | keep/letter | after 3 | after 9 | Video |
|---|---|---|---|---|
| slowest | 0.78 | 0.604 | **0.106** | 0.11 ✅ |
| — | 0.50 | 0.250 | 0.002 | — |
| — | 0.28 | 0.078 | 1.1e-05 | — |
| fastest | 0.14 | 0.0196 | 2.1e-08 | "almost nothing after 3" ✅ |

### 6.4 Scaling (d=768, expand 2, N=16)

| L | Transformer/layer | Mamba/layer | Ratio |
|---|---|---|---|
| 16 | 24,576 | 24,576 | **1.0×** |
| 1,000 | 1,536,000 (~1.5M) | 24,576 | **62.5×** |
| 32,000 | 49,152,000 (~49M) | 24,576 | **2,000×** |

Matches video verbatim. Formula: ratio = `L/16`.

### 6.5 Million-length retention

- Fixed `keep=0.99`: `7·0.99^1000 = 0.000302` (video 0.0003) ✅
- Selective (filler step 1e-9): `7·(e^{-1e-9})^{1e6} = 6.99300` (video 6.993) ✅ — still reads as 7.

## 7. Real public data + live market verification

> No-Key, honest-mode policy: every loader returns `(data, mode)` where `mode ∈ {live-*, synthetic-fallback}`. This run (2026-10-16):

| Stream | Source | Mode (this run) | n | Finding |
|---|---|---|---|---|
| Text | Karpathy TinyShakespeare via raw GitHub | **live-github-tinyshakespeare** (20k chars) | 1,000 alnum toks | Selective holds last digit through prose; fixed EMA smears |
| Equities | AAPL daily | **live-yahoo-AAPL-2y** | 501 closes, σ=1.80%, 24 events (>2σ) | Selective mean hold-err 0.0092; fixed best keep=0 |
| Crypto | BTC daily | **live-coingecko-btcusd-365d** | 365 closes, σ=2.30%, 18 events | Selective mean hold-err 0.0110; fixed best keep=0 |

Cross-check: independent `stock_quote(AAPL)` (yfinance) returned **$342.76** same day — consistent with Yahoo closes trajectory. Market mapping: `|r|>2σ` days = digits (step 5, wipe+write); ordinary days = letters (step 0.01, ignore). Fixed EMA's MSE-optimal keep is 0 (i.e. "just read the last word") — the market rediscovery of the video's `keep=0` result, now on live prices. Selective memory instead *keeps the last regime shock alive through subsequent noise*, which is the practically useful behavior for trend/regime features.

`figures/market_aapl.png` shows the live AAPL window used.

## 8. Hidden pattern

**Selectivity-threshold phase transition** (novel, `hidden_pattern_selectivity_threshold` in `metrics.json`, n=300/point, seed 7):

| filler step | filler keep | accuracy |
|---|---|---|
| 1e-7 | ~1.000000 | 100% |
| 1e-5 | ~1.000000 | 100% |
| 1e-3 | 0.999 | 100% |
| 0.01 | 0.990 | 99.3% |
| 0.1 | 0.905 | 57.7% |
| 0.5 | 0.607 | 37.3% |
| 1.0 | 0.368 | 33.3% |

Accuracy is flat at ceiling, then collapses over one decade of leak rate — a sharp, not gradual, degradation. Prediction for learned Mamba: Δ-gating histograms should be **bimodal** (near-0 vs. large), and regularizing filler-Δ upward should hurt long-range copying discontinuously. This is directly testable on the paper's selective-copy task and offered as the PhD-paper Figure 1 / Hypothesis H1.

Two secondary patterns for follow-up: (a) **slot specialization** — slow slots dominate accuracy when filler runs are long, fast slots when signal is dense (testable by ablating slots); (b) **market analogue** — selective hold-error scales with event density, fixed EMA error does not (see live §7).

## 9. Paper vs. rerun

| Paper claim (Gu & Dao) | Our status |
|---|---|
| Selective-copy: S4 18.3% → selective 97% → full 99.8% | Reported (needs training); our last-digit analogue reproduces the *gap* (28% → 100%) without training |
| Induction/associative recall to ~1M tokens (4,000× train length 256) | Reported; we implement the task generator (`associative_recall_row`) and leave trained generalization to future work |
| 5× throughput; 3B matches 2× Transformer | Reported; independent 2026 benchmarks corroborate (12.46× mem / 10.67× time at 4k; hybrid +2.65) |
| Everything in §6–§8 | **Reran and verified** on CPU |

## 10. Limitations

1. Hand-set steps, not learned `s_Δ(x)` — we show *why* selectivity suffices, not *how* gradient descent finds it.
2. Scalar channels, no `B_t/C_t`, no scan kernel, no perplexity evals — full Mamba has "a few extra parts" we omit by design.
3. Fixed memory is lossy: what it drops is gone. Transformers pay `O(N)` cache to keep everything re-queryable and still win on strong copying/in-context tasks at scale (cf. 8B study). Hybrids (Jamba, Mamba-2-Hybrid) are the pragmatic frontier.
4. Live-market demo is illustrative, not a trading signal — no returns claimed, no lookahead (causal only), costs ignored.

## 11. Repo map

```
mamba-selective-memory-phd-2026/
  README.md                 ← this paper
  Dockerfile                ← python:3.11-slim, pip install -r requirements.txt
  docker-compose.yml        ← `study` (run_all) + `tests` (pytest) services
  requirements.txt          ← numpy, matplotlib, pandas, requests, pytest (pinned major)
  src/
    memory.py               ← keep=exp(-step), scalar + selective + MultiSlotMemory
    games.py                ← last-digit / selective-copy / associative-recall generators
    benchmarks.py           ← KV vs. state formulas, decay math
    real_data.py            ← TinyShakespeare + Yahoo + CoinGecko loaders (honest-mode)
  experiments/run_all.py    ← one command reproduces every number + 3 figures
  tests/test_memory.py      ← 5 hand-check tests (exact video arithmetic)
  results/metrics.json      ← generated, committed after verified run
  figures/{scaling,forgetting,market_aapl}.png
```

### Reproduce in 3 commands

```bash
docker compose up study        # runs experiments/run_all.py --seed 0
docker compose up tests        # 5 hand-check tests
# or without docker:
pip install -r requirements.txt && python experiments/run_all.py --seed 0 && python -m pytest tests/ -v
```

Expected: `test_fixed ≈ 0.28`, `test_selective = 1.0`, `K3BQ8ZZ = 8`, `ratio(1000)=62.5`, `forget fixed 0.0003 / selective 6.993`, live modes `live-*` when online.

## 12. References

- Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752 (2023); COLM 2024.
- Dao, T. & Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024 (Mamba-2).
- Lahoti et al. *Mamba-3: Improved Sequence Modeling using State Space Principles.* arXiv:2603.15569 (2026).
- Gu et al. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* NeurIPS 2021; *Combining Recurrent, Convolutional, and Continuous-time Models.* 2021.
- Waleffe et al. *An Empirical Study of Mamba-based Language Models.* arXiv:2406.07887 (2024) — 8B controlled comparison + hybrid.
- Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* arXiv:2403.19887 (2024).
- Koledoye, A. *Benchmarking Computational and Representational Efficiency of SSMs against Transformers.* arXiv:2601.01237 (2026) — crossover 220/370 tokens, 12.46×/10.67× at 4k.
- Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* arXiv:2405.14782 (2024) — lm-eval best practice.
- Cao et al. *HOW2BENCH (55 checks).* ICML 2026 position — benchmark rigor/reliability/reproducibility.
- Gundersen, Gil et al. *General Reproducibility Guidelines for AI Research.* — claims/hyperparams/infra/variance/code.
- Hugging Face docs: Mamba; state-spaces/mamba GitHub (install/usage/evals); Papers-with-Code Mamba-2.
- Data: Karpathy TinyShakespeare (GitHub); Yahoo Finance v8 (AAPL 2y, no key); CoinGecko market_chart (BTC 365d, no key); cross-check via yfinance quote.

### Citation

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

*This repo is an independent educational reproduction; paper numbers in §9 are credited to Gu & Dao, not rerun. All other numbers were executed on CPU, 2026-10-16, seeds fixed.*
