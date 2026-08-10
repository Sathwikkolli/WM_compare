"""
align_bench/plots.py -- the figures.

matplotlib only (no seaborn) so it runs in a bare cluster env.
Writes PNGs to results/<run>/figures/.

  1. error_cdf.png        Tolerance vs fraction of pairs within it, one curve per
                          method. THE single best figure: read off any tolerance
                          you like, and crossing curves tell you "A wins if you
                          need 1 ms, B wins if 20 ms is fine."
  2. heatmap.png          method x attack, coloured by hit@20ms. The overview.
  3. strength_curves.png  hit rate vs attack strength, one panel per attack.
  4. scatter.png          predicted vs true offset. The VALUE is off-diagonal:
                          a cloud sitting exactly one echo-delay off the line
                          means the method locked onto the echo -- no summary
                          statistic would ever show you that.
  5. reliability.png      confidence vs observed accuracy. Diagonal = trustworthy.

Usage:
    python plots.py
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

RUN_SLUG = "2026-08-05_align-method-bakeoff"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# headline bar = 50 ms: AWARE(20bps) slides its detector in 42 ms steps, so
# finer alignment buys the detector nothing. See score.py for the full rationale.
TOL_MS = 50.0
CODEC_TOL_MS = 100.0
SR = 16000


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "raw_clip*.csv"))):
        with open(p, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _f(r, k):
    v = r.get(k, "")
    try:
        return float(v) if v not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def _hit(r):
    e = abs(_f(r, "error_ms"))
    tol = CODEC_TOL_MS if r["family"] == "codec" else TOL_MS
    return int(int(r.get("ok", 0) or 0) == 1 and np.isfinite(e) and e <= tol)


def fig_error_cdf(rows):
    by_m = defaultdict(list)
    for r in rows:
        if int(r.get("ok", 0) or 0) == 1:
            e = abs(_f(r, "error_ms"))
            if np.isfinite(e):
                by_m[r["method"]].append(max(e, 1e-3))

    grid = np.logspace(-3, 4, 400)          # 1 us .. 10 s
    plt.figure(figsize=(8, 5))
    for m in sorted(by_m):
        e = np.sort(np.array(by_m[m]))
        frac = np.searchsorted(e, grid, side="right") / len(e)
        plt.semilogx(grid, frac * 100, label=f"{m}  (n={len(e)})", lw=2)
    for x, lab, ls in ((1.0, "1 ms", ":"), (20.0, "20 ms", ":"),
                       (50.0, "50 ms  (AWARE 20bps step = 42 ms)", "--")):
        plt.axvline(x, color="k", ls=ls, lw=1, alpha=0.6)
        plt.text(x * 1.08, 4, lab, fontsize=8, rotation=90, va="bottom")
    plt.xlabel("error tolerance (ms, log scale)")
    plt.ylabel("% of pairs aligned within tolerance")
    plt.title("Alignment accuracy -- error CDF")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_cdf.png"), dpi=150)
    plt.close()


def fig_heatmap(rows):
    methods = sorted({r["method"] for r in rows})
    attacks = sorted({r["attack"] for r in rows})
    grid = np.full((len(methods), len(attacks)), np.nan)
    acc = defaultdict(list)
    for r in rows:
        acc[(r["method"], r["attack"])].append(_hit(r))
    for i, m in enumerate(methods):
        for j, a in enumerate(attacks):
            v = acc.get((m, a))
            if v:
                grid[i, j] = float(np.mean(v)) * 100

    plt.figure(figsize=(max(9, 0.55 * len(attacks) + 3), 0.5 * len(methods) + 3))
    plt.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(label=f"hit rate @ {TOL_MS:.0f} ms (%)")
    plt.xticks(range(len(attacks)), attacks, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(methods)), methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(len(attacks)):
            if np.isfinite(grid[i, j]):
                plt.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                         fontsize=7,
                         color="black" if 25 < grid[i, j] < 75 else "white")
    plt.title(f"Hit rate @ {TOL_MS:.0f} ms -- method x attack "
              f"(codec attacks bounded at {CODEC_TOL_MS:.0f} ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "heatmap.png"), dpi=150)
    plt.close()


def fig_strength_curves(rows):
    attacks = sorted({r["attack"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    ncol = 4
    nrow = int(np.ceil(len(attacks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             squeeze=False)
    for k, a in enumerate(attacks):
        ax = axes[k // ncol][k % ncol]
        params = []
        for r in rows:
            if r["attack"] == a and r["param"] not in params:
                params.append(r["param"])
        for m in methods:
            ys = []
            for p in params:
                sel = [_hit(r) for r in rows
                       if r["attack"] == a and r["param"] == p and r["method"] == m]
                ys.append(np.mean(sel) * 100 if sel else np.nan)
            ax.plot(range(len(params)), ys, marker="o", ms=3, lw=1.2, label=m)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right", fontsize=6)
        ax.set_title(a, fontsize=9)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6)
    for k in range(len(attacks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Hit rate @ {TOL_MS:.0f} ms vs attack strength", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "strength_curves.png"), dpi=140)
    plt.close(fig)


def fig_scatter(rows):
    methods = sorted({r["method"] for r in rows})
    ncol = 3
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, m in enumerate(methods):
        ax = axes[k // ncol][k % ncol]
        t = [_f(r, "true_offset") / SR for r in rows
             if r["method"] == m and int(r.get("ok", 0) or 0) == 1]
        p = [_f(r, "pred_offset") / SR for r in rows
             if r["method"] == m and int(r.get("ok", 0) or 0) == 1]
        t, p = np.array(t), np.array(p)
        ok = np.isfinite(t) & np.isfinite(p)
        ax.scatter(t[ok], p[ok], s=6, alpha=0.35, edgecolors="none")
        if ok.any():
            lim = [min(t[ok].min(), p[ok].min()), max(t[ok].max(), p[ok].max())]
            ax.plot(lim, lim, "k--", lw=1, alpha=0.7)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("true offset (s)", fontsize=8)
        ax.set_ylabel("predicted offset (s)", fontsize=8)
        ax.grid(alpha=0.3)
    for k in range(len(methods), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Predicted vs true offset -- off-diagonal clusters show WHY it failed")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scatter.png"), dpi=140)
    plt.close(fig)


def fig_reliability(rows):
    plt.figure(figsize=(6.5, 5))
    edges = np.linspace(0, 1, 11)
    for m in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == m]
        c = np.array([_f(r, "confidence") for r in sel])
        h = np.array([_hit(r) for r in sel], dtype=float)
        ok = np.isfinite(c)
        if ok.sum() < 20:
            continue
        c, h = c[ok], h[ok]
        xs, ys = [], []
        for i in range(len(edges) - 1):
            m_ = (c >= edges[i]) & (c < edges[i + 1])
            if m_.sum() >= 5:
                xs.append((edges[i] + edges[i + 1]) / 2)
                ys.append(h[m_].mean() * 100)
        if xs:
            plt.plot(xs, ys, marker="o", ms=4, lw=1.5, label=m)
    plt.plot([0, 1], [0, 100], "k--", lw=1, alpha=0.6, label="perfectly calibrated")
    plt.xlabel("method's reported confidence")
    plt.ylabel(f"observed hit rate @ {TOL_MS:.0f} ms (%)")
    plt.title("Reliability -- can you trust it to know when it failed?")
    plt.ylim(-5, 105)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reliability.png"), dpi=150)
    plt.close()


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(f"no raw_clip*.csv in {DATA_DIR} -- run run_bench.py first")
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"plotting {len(rows)} rows")
    for fn in (fig_error_cdf, fig_heatmap, fig_strength_curves,
               fig_scatter, fig_reliability):
        try:
            fn(rows)
            print(f"  {fn.__name__} ok")
        except Exception as e:
            print(f"  {fn.__name__} FAILED: {e}")
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
