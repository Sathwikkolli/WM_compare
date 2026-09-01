"""
informed/plots_phase_b.py -- the three Phase B figures.

  gain_curve.png       ONE attack, the explainer. Both detectors' scores against
                       attack strength, each with its own FPR-matched threshold,
                       and the band between the two crossings shaded. That band
                       IS the result. Needs `bisect_sweep.py --curve`.
  gain_summary.png     ALL attacks, the headline. One bar per attack, gain in
                       that attack's own units, zero line marked so a NEGATIVE
                       result -- informed doing worse -- is visible rather than
                       hidden. This is the abstract figure.
  null_separation.png  The honesty figure. Score distributions on watermarked
                       and unwatermarked audio per detector, with the threshold
                       drawn in. Without it the first question at a defence is
                       "how do we know that comparison was fair?"

Every figure degrades gracefully: a missing input is reported and skipped rather
than aborting the others.

Usage:
    python plots_phase_b.py
    python plots_phase_b.py --curve-attack mp3
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
SWEEP_DIR = os.path.join(DATA_DIR, "sweep")
NULL_DIR = os.path.join(DATA_DIR, "null")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

C_BLIND = "#c2410c"
C_INF = "#1d4ed8"
C_BAND = "#16a34a"


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def read_csvs(pattern):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def cross_x(xs, ys, thr_ys):
    """Where score falls through its threshold as strength increases."""
    order = np.argsort(xs)
    x, y, t = np.asarray(xs)[order], np.asarray(ys)[order], np.asarray(thr_ys)[order]
    g = y - t
    for i in range(len(x) - 1):
        if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and g[i] > 0 >= g[i + 1]:
            if g[i] == g[i + 1]:
                return float(x[i + 1])
            return float(x[i] + (0 - g[i]) * (x[i + 1] - x[i]) / (g[i + 1] - g[i]))
    return float("nan")


# --------------------------------------------------------------------------- #
#  1. the gain curve
# --------------------------------------------------------------------------- #
def fig_gain_curve(attack):
    rows = [r for r in read_csvs(os.path.join(SWEEP_DIR, "curve_clip*.csv"))
            if r["attack"] == attack]
    if not rows:
        print(f"  gain_curve: no curve data for '{attack}' "
              f"(run: bisect_sweep.py --clip 0 --attacks {attack} --curve)")
        return None

    unit = rows[0].get("unit", "")
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    crossings = {}

    for ax, arm, colour in ((axes[0], "blind", C_BLIND),
                            (axes[1], "informed", C_INF)):
        by_t = defaultdict(list)
        thr_by_t = {}
        for r in rows:
            if r["arm"] != arm:
                continue
            t, s, th = fnum(r["t"]), fnum(r["score"]), fnum(r["threshold"])
            if np.isfinite(t) and np.isfinite(s):
                by_t[t].append(s)
            if np.isfinite(t) and np.isfinite(th):
                thr_by_t[t] = th
        ts = sorted(by_t)
        if not ts:
            continue
        mean = np.array([np.mean(by_t[t]) for t in ts])
        sd = np.array([np.std(by_t[t], ddof=1) if len(by_t[t]) > 1 else 0.0
                       for t in ts])
        thr = np.array([thr_by_t.get(t, np.nan) for t in ts])
        vals = np.array([fnum(next(r["value"] for r in rows
                                   if fnum(r["t"]) == t)) for t in ts])

        ax.plot(ts, mean, "o-", color=colour, lw=2, ms=4,
                label=f"{arm} score")
        ax.fill_between(ts, mean - sd, mean + sd, color=colour, alpha=0.13)
        ax.plot(ts, thr, ":", color=colour, lw=1.4,
                label="threshold at 1% false positives")

        xc = cross_x(ts, mean, thr)
        crossings[arm] = xc
        if np.isfinite(xc):
            ax.axvline(xc, color=colour, ls="--", lw=1.2)
        ax.set_ylabel(f"{arm} score")
        ax.grid(alpha=0.22, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        # secondary x labels in native units, on the top panel only
        if arm == "blind" and len(vals) == len(ts):
            sec = ax.secondary_xaxis("top")
            sel = np.linspace(0, len(ts) - 1, min(6, len(ts))).astype(int)
            sec.set_xticks([ts[i] for i in sel])
            sec.set_xticklabels([f"{vals[i]:g}" for i in sel], fontsize=8)
            sec.set_xlabel(f"attack strength ({unit})", fontsize=9)

    b, i = crossings.get("blind", np.nan), crossings.get("informed", np.nan)
    if np.isfinite(b) and np.isfinite(i) and i > b:
        for ax in axes:
            ax.axvspan(b, i, color=C_BAND, alpha=0.13, zorder=0)
        axes[0].annotate(f"informed survives\n{i - b:.2f} further along the axis",
                         xy=((b + i) / 2, 0.5), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=9, color="#15803d")

    axes[1].set_xlabel("attack strength  (0 = weakest, 1 = strongest)")
    axes[0].set_title(f"Where each detector fails  [{attack}]",
                      fontsize=12, loc="left")
    fig.text(0.01, -0.01,
             "Both arms see the identical attacked file; only the detector "
             "differs. Thresholds are matched at 1% false positives, so the two "
             "are equally willing to say yes.", fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  2. the summary
# --------------------------------------------------------------------------- #
def fig_gain_summary():
    p = os.path.join(DATA_DIR, "phase_b_metrics.csv")
    if not os.path.exists(p):
        print("  gain_summary: phase_b_metrics.csv missing (run score_phase_b.py)")
        return None
    rows = [r for r in csv.DictReader(open(p, newline=""))
            if np.isfinite(fnum(r.get("gain_native_mean")))]
    if not rows:
        print("  gain_summary: no finite gains")
        return None

    rows.sort(key=lambda r: fnum(r["gain_native_mean"]))
    labels = [f"{r['attack']}  ({r['unit']})" for r in rows]
    gains = np.array([fnum(r["gain_native_mean"]) for r in rows])
    sds = np.array([fnum(r.get("gain_native_sd")) for r in rows])
    sds = np.where(np.isfinite(sds), sds, 0.0)
    ns = [int(fnum(r.get("n_paired")) or 0) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.34 * len(rows) + 1.6)))
    colours = [C_INF if g > 0 else C_BLIND for g in gains]
    y = np.arange(len(rows))
    ax.barh(y, gains, xerr=sds, color=colours, alpha=0.85,
            error_kw=dict(lw=0.8, ecolor="#6b7280"))
    ax.axvline(0, color="#111827", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("gain — how much further informed detection survives "
                  "(each attack in its own units)")
    ax.set_title("Does informed detection extend robustness?",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.22, axis="x", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    for yi, (g, nn) in enumerate(zip(gains, ns)):
        ax.text(g, yi, f"  n={nn}", va="center", fontsize=7.5,
                ha="left" if g >= 0 else "right", color="#4b5563")

    fig.text(0.01, -0.015,
             "Right of zero = informed helps. Left = informed is WORSE, which "
             "is a real possible outcome when subtraction amplifies alignment "
             "or gain errors. Units differ per attack and must not be averaged.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  3. null separation
# --------------------------------------------------------------------------- #
def fig_null_separation(attack):
    raw = [r for r in read_csvs(os.path.join(NULL_DIR, f"nullraw_{attack}.csv"))]
    curve = [r for r in read_csvs(os.path.join(SWEEP_DIR, "curve_clip*.csv"))
             if r["attack"] == attack]
    if not raw:
        print(f"  null_separation: no nullraw_{attack}.csv "
              f"(re-run Stage 1 -- older runs did not save raw scores)")
        return None

    # A mid-strength slice: strong enough to be interesting, weak enough that
    # the watermark should still be findable.
    strengths = sorted({fnum(r["strength"]) for r in raw})
    target = strengths[len(strengths) // 2] if strengths else 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, arm, colour in ((axes[0], "blind", C_BLIND),
                            (axes[1], "informed", C_INF)):
        neg = [fnum(r["score"]) for r in raw
               if r["arm"] == arm and abs(fnum(r["strength"]) - target) < 1e-6]
        neg = [v for v in neg if np.isfinite(v)]
        pos = [fnum(r["score"]) for r in curve
               if r["arm"] == arm and abs(fnum(r["t"]) - target) < 0.05]
        pos = [v for v in pos if np.isfinite(v)]
        if not neg:
            continue

        bins = np.linspace(min(neg + (pos or neg)), max(neg + (pos or neg)), 40)
        ax.hist(neg, bins=bins, color="#9ca3af", alpha=0.75,
                label=f"no watermark (n={len(neg)})")
        if pos:
            ax.hist(pos, bins=bins, color=colour, alpha=0.6,
                    label=f"watermarked (n={len(pos)})")
        thr = np.sort(neg)[int(np.ceil(0.99 * len(neg)))] if len(neg) > 2 else np.nan
        if np.isfinite(thr):
            ax.axvline(thr, color="#111827", ls="--", lw=1.2)
            ax.annotate("1% FPR", xy=(thr, 0.92), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8, ha="right", va="top")
        ax.set_title(f"{arm}", fontsize=11, loc="left")
        ax.set_xlabel("score")
        ax.grid(alpha=0.2, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("count")
    fig.suptitle(f"Are the two detectors equally willing to say yes?  "
                 f"[{attack}, strength {target:.2f}]", fontsize=12, x=0.01,
                 ha="left")
    fig.text(0.01, -0.02,
             "Each threshold is placed where 1% of UNWATERMARKED audio would be "
             "accepted. That is what makes the two arms comparable despite "
             "different score scales.", fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def main(argv):
    os.makedirs(FIG_DIR, exist_ok=True)
    ca = get_arg(argv, "--curve-attack", "music_bed")

    for name, fn in (("gain_curve", lambda: fig_gain_curve(ca)),
                     ("gain_summary", fig_gain_summary),
                     ("null_separation", lambda: fig_null_separation(ca))):
        try:
            fig = fn()
        except Exception as e:
            print(f"  {name}: failed ({type(e).__name__}: {e})")
            continue
        if fig is None:
            continue
        out = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
