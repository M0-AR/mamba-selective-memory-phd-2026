"""Generate docs/demo.gif — animated selective vs fixed memory (verified, no hand art).

Best practice applied (2026 GIF-for-README): ~720px wide, ~15fps equivalent
(here 4 fps for clarity, 20 frames), cropped to content, <2MB.
Shows one row: digits wipe+write, letters ignored (selective) vs smeared (fixed).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.memory import keep_from_step, update_scalar
from src.games import token_value

ROW = list("K3BQ8ZZQ1AX7Z")  # last digit 7, letters are noise
VALS = [token_value(t) for t in ROW]
IS_DIG = [t.isdigit() for t in ROW]

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "docs"), exist_ok=True)
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "demo.gif")

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100)
h_fix, h_sel = 0.0, 0.0
trace_fix, trace_sel = [], []

def frame(i):
    global h_fix, h_sel
    ax.clear()
    x, imp = VALS[i], IS_DIG[i]
    h_fix = update_scalar(h_fix, float(x), 0.0 if False else 0.5)  # fixed keep 0.5 for visual contrast
    step = 20.0 if imp else 1e-6
    h_sel = update_scalar(h_sel, float(x), keep_from_step(step))
    trace_fix.append(h_fix); trace_sel.append(h_sel)
    xs = range(len(trace_fix))
    ax.plot(xs, trace_fix, marker="o", label="fixed keep=0.5 (smears)")
    ax.plot(xs, trace_sel, marker="s", label="selective (holds digits)")
    ax.set_xticks(list(xs)); ax.set_xticklabels(ROW[:len(trace_fix)])
    ax.set_ylim(-0.5, 9.5); ax.set_ylabel("memory h")
    ax.set_title(f"Step {i+1}/{len(ROW)}: token '{ROW[i]}' ({'DIGIT: wipe+write' if imp else 'letter: ignore'})  h_sel={h_sel:.2f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

# build by re-running (deterministic)
fig2, ax2 = plt.subplots(figsize=(7.2, 3.6), dpi=100)
frames = []
h_fix, h_sel = 0.0, 0.0
trace_fix, trace_sel = [], []
for i in range(len(ROW)):
    frame(i)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    frames.append(buf)
plt.close(fig)

import PIL.Image as Image
imgs = [Image.fromarray(f) for f in frames]
# hold first/last frames longer
durations = [600] + [450]*(len(imgs)-2) + [1200]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=durations, loop=0)
size = os.path.getsize(OUT)
print(f"WROTE {OUT} {size/1024:.1f}KB frames={len(imgs)}")
assert size < 10*1024*1024, "GIF must stay under GitHub 10MB limit"
