"""
frame_align/plots_frames.py -- figures from metrics.csv.

Reads results/<run>/data/metrics.csv (written by score_frames.py) and writes
results/<run>/figures/*.png. Nothing is hardcoded: if the numbers change, the
figures change.

Three figures, one per finding worth showing to someone who will not read a table:

  knee.png         useful-accept vs frame length. THE headline. Shows the ~500 ms
                   requirement and the aof/gcc_phat crossover in one picture.
  false_alarm.png  what a matched-data-only calibration would have shipped.
                   Shows that the null test matters at 250-500 ms and stops
                   mattering above 1000 ms.
  precision.png    hit rate at 20 ms vs 1 ms. Shows gcc_phat is sample-exact and
                   aof is grid-quantised -- the two bars are equal for one method
                   and wildly unequal for the other.

Usage:
    python plots_frames.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # no display on a login node
import matplotlib.pyplot as plt            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

RUN_SLUG = "2026-08-18_frame-align-null"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

COLOR = {"gcc_phat": "#c2410c", "aof": "#1d4ed8"}
MARKER = {"gcc_phat": "o", "aof": "s"}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load():
    p = os.path.join(DATA_DIR, "metrics.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing -- run score_frames.py first")
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, section, value_key, method):
    """(frame lengths, values) for one method, sorted by length."""
    pts = []
    for r in rows:
        if r.get("section") != section or r.get("method") != method:
            continue
        L = fnum(r.get("frame_len_ms"))
        v = fnum(r.get(value_key))
        if L == L and v == v:                    # both finite
            pts.append((L, v))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def style(ax, lens):
    ax.set_xscale("log")
    ax.set_xticks(lens)
    ax.set_xticklabels([f"{int(x)}" for x in lens])
    ax.set_xlabel("frame length (ms)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def fig_knee(rows, methods, lens):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in methods:
        x, y = series(rows, "operating_point", "useful_accept", m)
        ax.plot(x, [100 * v for v in y], marker=MARKER.get(m, "o"),
                color=COLOR.get(m), linewidth=2, markersize=6, label=m)
    ax.axvline(500, color="#6b7280", linestyle="--", linewidth=1)
    ax.text(520, 8, "500 ms\nworking point", fontsize=8, color="#6b7280")
    ax.axhline(95, color="#6b7280", linestyle=":", linewidth=1)
    ax.set_ylim(-3, 105)
    ax.set_ylabel("useful-accept rate at 1% false-accept (%)")
    ax.set_title("How much audio does alignment need?", fontsize=12, loc="left")
    style(ax, lens)
    ax.legend(frameon=False)
    fig.text(0.01, -0.02,
             "Correctly located within 50 ms AND confident enough to trust, at a "
             "threshold admitting 1% of unrelated audio.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def fig_false_alarm(rows, methods, lens):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in methods:
        x, y = series(rows, "naive_fa", "fa_keep95", m)
        ax.plot(x, [100 * v for v in y], marker=MARKER.get(m, "o"),
                color=COLOR.get(m), linewidth=2, markersize=6, label=m)
    ax.axhspan(0, 1, color="#16a34a", alpha=0.10)
    ax.text(55, 2.5, "1% target", fontsize=8, color="#15803d")
    ax.set_ylabel("unrelated audio wrongly accepted (%)")
    ax.set_title("What we would have shipped without the null test",
                 fontsize=12, loc="left")
    ax.set_ylim(-3, 100)
    style(ax, lens)
    ax.legend(frameon=False)
    fig.text(0.01, -0.02,
             "Threshold set on genuine audio alone, to keep 95% of it -- then "
             "measured against audio that does not match.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def fig_precision(rows, methods, lens):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.36
    idx = list(range(len(lens)))
    for k, m in enumerate(methods):
        by_tol = {}
        for r in rows:
            if r.get("section") != "hit_rate" or r.get("method") != m:
                continue
            by_tol.setdefault(fnum(r.get("tol_ms")), {})[fnum(r.get("frame_len_ms"))] = \
                fnum(r.get("hit_rate"))
        loose = [100 * by_tol.get(20.0, {}).get(L, float("nan")) for L in lens]
        tight = [100 * by_tol.get(1.0, {}).get(L, float("nan")) for L in lens]
        pos = [i + (k - 0.5) * width for i in idx]
        ax.bar(pos, loose, width * 0.92, color=COLOR.get(m), alpha=0.35,
               label=f"{m} — within 20 ms")
        ax.bar(pos, tight, width * 0.92, color=COLOR.get(m),
               label=f"{m} — within 1 ms")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{int(L)}" for L in lens])
    ax.set_xlabel("frame length (ms)")
    ax.set_ylabel("frames located (%)")
    ax.set_title("Precision: solid bar = exact, pale bar = close enough",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(0.01, -0.02,
             "Equal bars = sample-exact. A tall pale bar over a short solid bar = "
             "the answer is snapped to a coarse grid.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def main():
    rows = load()
    methods = sorted({r["method"] for r in rows if r.get("method")})
    lens = sorted({fnum(r.get("frame_len_ms")) for r in rows
                   if fnum(r.get("frame_len_ms")) == fnum(r.get("frame_len_ms"))})
    os.makedirs(FIG_DIR, exist_ok=True)

    for name, fn in (("knee", fig_knee),
                     ("false_alarm", fig_false_alarm),
                     ("precision", fig_precision)):
        fig = fn(rows, methods, lens)
        out = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
