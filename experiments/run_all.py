"""End-to-end reproducible study: video claims -> benchmarks -> real/live data.

Outputs (all deterministic given --seed):
  results/metrics.json   all numbers cited in README/paper
  figures/*.png          forgetting curve, scaling, slots, market demo
Prints a console table so Docker logs are self-contained.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.memory import keep_from_step, run_sequence, run_selective, MultiSlotMemory, FOUR_SLOT_ALPHAS
from src.games import evaluate, predict_fixed, predict_selective, sweep_fixed_keeps, token_value
from src.benchmarks import transformer_numbers_per_layer, mamba_numbers_per_layer, ratio_at, fixed_decay
from src.real_data import load_text_corpus, load_stooq_daily, market_selectivity_demo

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

def ensure_dirs():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def exp_hand_rules():
    return {
        "trace_4_0_0_2_keep05": run_sequence([4,0,0,2], keep=0.5),
        "trace_2_0_6_keep05": run_sequence([2,0,6], keep=0.5),
        "keep_step_001": keep_from_step(0.01), "keep_step_5": keep_from_step(5.0),
        "keep_step_20": keep_from_step(20.0),
        "K3BQ8ZZ_selective": predict_selective(list("K3BQ8ZZ")),
    }

def exp_last_digit_game():
    cand = [i/100 for i in range(0,100,2)] + [0.9,0.95,0.99,0.999]
    # video swept 3100 settings on 1000 rows; we sweep 53 representative keeps (documented)
    best_keep, best_train = sweep_fixed_keeps(cand, n_rows=1000, seed=0)
    acc_fixed, used_f, _ = evaluate(lambda r: predict_fixed(r, best_keep), 1000, seed=1)
    acc_sel, used_s, _ = evaluate(predict_selective, 1000, seed=1)
    # digit-ending rate
    rng = random.Random(1); ends_digit = sum(1 for _ in range(1000)
        if __import__("src.games", fromlist=["random_row","last_digit"]).random_row(20,0.2,rng)[-1].isdigit())/1000
    return {"best_keep": best_keep, "train_acc": best_train,
            "test_fixed": acc_fixed, "test_selective": acc_sel,
            "used": used_s, "p_last_is_digit": ends_digit, "n_train": 1000, "n_test": 1000}

def exp_slots_timescales():
    mem = MultiSlotMemory(FOUR_SLOT_ALPHAS)
    traj = mem.run([1.0]+[0.0]*9, [20.0]+[1.0]*9)  # digit 1 then 9 filler letters (step 1.0)
    per_letter = [math.exp(-a*1.0) for a in FOUR_SLOT_ALPHAS]
    return {"alphas": FOUR_SLOT_ALPHAS, "per_letter_keep": per_letter,
            "after9": traj[-1].tolist(), "after3": traj[2].tolist()}

def exp_scaling_forgetting():
    scales = [16,1000,32000]
    table = {L: {"transformer": transformer_numbers_per_layer(L), "mamba": mamba_numbers_per_layer(),
                 "ratio": ratio_at(L)} for L in scales}
    forget = {"fixed_099_1000": fixed_decay(7,0.99,1000),
              "selective_1M": 7*(keep_from_step(1e-9)**1_000_000)}
    # paper numbers (Gu & Dao, reported not rerun): selective-copy S4 18.3 / selective 97 / full 99.8
    paper = {"selective_copy_S4": 18.3, "selective_copy_selective": 97.0, "selective_copy_full": 99.8,
             "throughput_gain": 5.0, "mamba3B_vs_2x": "matches 2x transformer"}
    # plots
    Ns = np.array([16,64,256,1000,4000,8000,16000,32000])
    plt.figure(); plt.plot(Ns, [transformer_numbers_per_layer(int(n)) for n in Ns], marker="o", label="Transformer KV (2*768*N)")
    plt.axhline(mamba_numbers_per_layer(), linestyle="--", label="Mamba fixed 24,576")
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("Sequence length"); plt.ylabel("Numbers held per layer")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,"scaling.png")); plt.close()
    steps = np.arange(0,1200)
    plt.figure(); plt.plot(steps, [fixed_decay(7,0.99,int(s)) for s in steps], label="fixed keep .99")
    plt.axhline(7*(keep_from_step(1e-9)**1_000_000), linestyle="--", label="selective after 1M fillers ~6.99")
    plt.xlabel("filler letters"); plt.ylabel("retained value of 7"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"forgetting.png")); plt.close()
    return {"table": table, "forget": forget, "paper_reported": paper}

def exp_real_live():
    txt, txt_mode = load_text_corpus()
    # text game: map chars to digit-noise like video (digits signal, letters noise)
    import re
    toks = [c for c in txt[:4000] if c.isalnum()][:1000]
    vals = [token_value(c) for c in toks]; imp = [c.isdigit() for c in toks]
    h_fix, h_sel = 0.0, 0.0
    for x, im in zip(vals, imp):
        h_fix = 0.9*h_fix + 0.1*x
        h_sel = keep_from_step(20.0 if im else 1e-6)*h_sel + (1-keep_from_step(20.0 if im else 1e-6))*x
    aapl, m_aapl = load_stooq_daily("aapl.us"); btc, m_btc = load_stooq_daily("btcusd")
    demo_aapl = market_selectivity_demo(aapl); demo_btc = market_selectivity_demo(btc)
    # market plot
    plt.figure(); plt.plot(aapl[-250:]); plt.title(f"AAPL closes ({m_aapl})"); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"market_aapl.png")); plt.close()
    return {"text_mode": txt_mode, "text_len": len(txt),
            "text_demo_fixed_mem": h_fix, "text_demo_sel_mem": h_sel,
            "aapl_mode": m_aapl, "btc_mode": m_btc,
            "aapl": demo_aapl, "btc": demo_btc}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    ensure_dirs()
    out = {"seed": a.seed, "hand": exp_hand_rules(), "game": exp_last_digit_game(),
           "slots": exp_slots_timescales(), "scale": exp_scaling_forgetting(), "live": exp_real_live()}
    # HIDDEN-PATTERN discovery (novel, testable): selectivity threshold phase transition
    # Sweep filler-step; accuracy jumps when filler keep crosses ~0.99 (document + verify).
    th = {}
    for s in [1e-7,1e-5,1e-3,0.01,0.1,0.5,1.0]:
        from src.games import evaluate as ev
        acc,_,_ = ev(lambda r,s=s: predict_selective(r,20.0,s), 300, seed=7)
        th[str(s)] = acc
    out["hidden_pattern_selectivity_threshold"] = th
    with open(os.path.join(RES,"metrics.json"),"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
    print("\nFIGURES:", os.listdir(FIG)); print("RESULTS written to results/metrics.json")

if __name__ == "__main__":
    main()
