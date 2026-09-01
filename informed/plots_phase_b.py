"""
informed/plots_phase_b.py -- the Phase B figures.

Five figures, in the order you would present them:

  1. overview.png         The one-page answer. Who won per attack (stacked), and
                          how big the gain was where it could be measured.
  2. status_map.png       Every clip x attack, coloured by outcome. THE figure
                          the first run needed and did not have: 73% of cases
                          were censored, and a bar chart of the remaining 27%
                          silently misrepresents the whole experiment.
  3. paired_scatter.png   Blind crossing vs informed crossing, one dot per clip,
                          with the diagonal drawn. Shows the pairing directly and
                          exposes spread that a mean hides.
  4. gain_curve.png       One attack in detail: both scores against strength,
                          each with its FPR-matched threshold, gap shaded.
  5. null_separation.png  Score distributions with and without a watermark.
                          The figure that shows the comparison was fair.

Every figure degrades gracefully -- a missing input is reported and skipped
rather than aborting the rest.

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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

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

# One palette across every figure, so a colour always means the same thing.
C_INF = "#1d4ed8"        # informed
C_BLIND = "#c2410c"      # blind
C_BOTH = "#65a30d"       # neither broke -- attack too weak to decide
C_NONE = "#d4d4d8"       # no data
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


def load_metrics():
    p = os.path.join(DATA_DIR, "phase_b_metrics.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p, newline="")))
    for r in rows:
        for k in ("gain_t_mean", "gain_t_sd", "gain_native_mean",
                  "gain_native_sd", "win_fraction", "wilcoxon_p"):
            r[k] = fnum(r.get(k))
        for k in ("n_paired", "n_clips", "decisive_wins", "both_survived",
                  "informed_won_censored", "censored_informed"):
            r[k] = int(fnum(r.get(k)) or 0)
    return rows


def outcome_grid():
    """(attacks, clips, codes) where code is 0 none / 1 blind / 2 both / 3 informed.

    Built from the RAW sweep rows rather than the metrics summary, because the
    per-clip status is exactly what the summary aggregates away.
    """
    rows = read_csvs(os.path.join(SWEEP_DIR, "sweep_clip*.csv"))
    if not rows:
        return [], [], None
    by = defaultdict(dict)
    for r in rows:
        by[(r["attack"], r["clip_id"])][r["arm"]] = r

    attacks = sorted({r["attack"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    grid = np.zeros((len(attacks), len(clips)), dtype=int)

    for ai, a in enumerate(attacks):
        for ci, c in enumerate(clips):
            pair = by.get((a, c), {})
            b, i = pair.get("blind"), pair.get("informed")
            if not b or not i:
                continue
            bs, ist = b["status"], i["status"]
            if bs == "CROSSED" and ist == "CROSSED":
                gt = fnum(i["t_cross"]) - fnum(b["t_cross"])
                grid[ai, ci] = 3 if gt > 0 else 1
            elif ist == "NO_CROSSING_SURVIVED" and bs == "CROSSED":
                grid[ai, ci] = 3                      # informed survived, blind did not
            elif ist == "NO_CROSSING_SURVIVED" and bs == "NO_CROSSING_SURVIVED":
                grid[ai, ci] = 2                      # neither broke
            elif bs == "NO_CROSSING_SURVIVED" and ist == "CROSSED":
                grid[ai, ci] = 1                      # blind survived, informed did not
    return attacks, clips, grid


# --------------------------------------------------------------------------- #
#  1. overview
# --------------------------------------------------------------------------- #
def fig_overview():
    rows = load_metrics()
    if not rows:
        print("  overview: phase_b_metrics.csv missing (run score_phase_b.py)")
        return None

    for r in rows:
        wf = r["win_fraction"] if np.isfinite(r["win_fraction"]) else 0.0
        r["_iw"] = r.get("decisive_wins", 0)
        r["_bw"] = r["n_paired"] - int(round(wf * r["n_paired"]))
        r["_bs"] = r.get("both_survived", 0)
        r["_nd"] = max(0, r["n_clips"] - r["_iw"] - r["_bw"] - r["_bs"])
    rows.sort(key=lambda r: (r["_iw"] - r["_bw"], r["_iw"]))

    labels = [r["attack"] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, max(4.4, 0.36 * len(rows) + 1.8)),
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.32})

    # -- A: who won, per clip ------------------------------------------------
    ax = axes[0]
    iw = np.array([r["_iw"] for r in rows], dtype=float)
    bw = np.array([r["_bw"] for r in rows], dtype=float)
    bs = np.array([r["_bs"] for r in rows], dtype=float)
    nd = np.array([r["_nd"] for r in rows], dtype=float)
    ax.barh(y, iw, color=C_INF, label="informed won")
    ax.barh(y, bw, left=iw, color=C_BLIND, label="blind won")
    ax.barh(y, bs, left=iw + bw, color=C_BOTH, label="neither broke")
    ax.barh(y, nd, left=iw + bw + bs, color=C_NONE, label="no data")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("clips (out of 50)")
    ax.set_title("A. Who won, per clip", fontsize=11, loc="left")
    ax.grid(alpha=0.2, axis="x", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    # -- B: how big, where measurable ---------------------------------------
    ax = axes[1]
    gt = np.array([r["gain_t_mean"] if np.isfinite(r["gain_t_mean"]) else np.nan
                   for r in rows])
    sd = np.array([r["gain_t_sd"] if np.isfinite(r["gain_t_sd"]) else 0.0
                   for r in rows])
    colours = [C_INF if (np.isfinite(g) and g > 0) else C_BLIND for g in gt]
    ax.barh(y, np.nan_to_num(gt), xerr=sd, color=colours, alpha=0.9,
            error_kw=dict(lw=0.8, ecolor="#6b7280"))
    ax.axvline(0, color="#111827", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("gain along the strength axis (0–1 scale)")
    ax.set_title("B. How much further informed survives", fontsize=11, loc="left")
    ax.grid(alpha=0.2, axis="x", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    for yi, r in enumerate(rows):
        if not np.isfinite(r["gain_native_mean"]):
            ax.text(0.01, yi, "  not measurable", va="center", fontsize=7.5,
                    color="#6b7280")
            continue
        g = r["gain_t_mean"]
        p = r["wilcoxon_p"]
        star = " *" if (np.isfinite(p) and p < 0.05) else ""
        ax.text(g, yi, f"  {r['gain_native_mean']:+.3g} {r.get('unit','')}{star}",
                va="center", fontsize=7.5,
                ha="left" if g >= 0 else "right", color="#374151")

    fig.suptitle("Informed vs blind detection — the whole picture",
                 fontsize=13, x=0.008, ha="left", y=1.0)
    fig.text(0.008, -0.02,
             "Left: every clip is a decision. Right: how big the effect was, on "
             "the attacks where BOTH detectors actually broke. * = Wilcoxon "
             "p < 0.05. Bars pointing left mean informed did WORSE.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  2. status map
# --------------------------------------------------------------------------- #
def fig_status_map():
    attacks, clips, grid = outcome_grid()
    if grid is None:
        print("  status_map: no sweep data")
        return None

    cmap = ListedColormap([C_NONE, C_BLIND, C_BOTH, C_INF])
    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.32 * len(attacks) + 1.6)))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest")
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=8.5)
    ax.set_xticks(np.arange(0, len(clips), max(1, len(clips) // 10)))
    ax.set_xticklabels([clips[i] for i in
                        range(0, len(clips), max(1, len(clips) // 10))],
                       fontsize=7.5, rotation=90)
    ax.set_xlabel("clip")
    ax.set_title("Outcome of every clip × attack", fontsize=12, loc="left")
    ax.legend(handles=[Patch(color=C_INF, label="informed won"),
                       Patch(color=C_BLIND, label="blind won"),
                       Patch(color=C_BOTH, label="neither broke"),
                       Patch(color=C_NONE, label="no data")],
              frameon=False, fontsize=8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.text(0.008, -0.06,
             "Green rows mean the attack never got strong enough to decide "
             "anything — widen that axis. Grey rows mean the attack or its "
             "calibration failed, which is a bug, not a result.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  3. paired scatter
# --------------------------------------------------------------------------- #
def fig_paired_scatter():
    p = os.path.join(DATA_DIR, "phase_b_pairs.csv")
    if not os.path.exists(p):
        print("  paired_scatter: phase_b_pairs.csv missing")
        return None
    rows = list(csv.DictReader(open(p, newline="")))
    by = defaultdict(list)
    for r in rows:
        b, i = fnum(r["blind_value"]), fnum(r["informed_value"])
        if np.isfinite(b) and np.isfinite(i):
            by[r["attack"]].append((b, i, r.get("unit", "")))
    by = {k: v for k, v in by.items() if len(v) >= 3}
    if not by:
        print("  paired_scatter: no attack has >=3 paired clips")
        return None

    names = sorted(by, key=lambda k: -len(by[k]))[:6]
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.7 * nrow),
                             squeeze=False)

    for k, name in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        pts = by[name]
        bx = np.array([p[0] for p in pts])
        iy = np.array([p[1] for p in pts])
        unit = pts[0][2]

        lo = min(bx.min(), iy.min())
        hi = max(bx.max(), iy.max())
        pad = 0.08 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
                color="#9ca3af", lw=1)

        # A point is an informed win when it sits on the "stronger attack" side
        # of the diagonal, and which side that is depends on the axis direction.
        import strength_axis as SA
        ax_def = SA.AXIS.get(name, {})
        stronger_is_lower = float(ax_def.get("hi", 1)) < float(ax_def.get("lo", 0))
        win = (iy < bx) if stronger_is_lower else (iy > bx)
        ax.scatter(bx[win], iy[win], s=26, color=C_INF, alpha=0.85,
                   label=f"informed won ({win.sum()})")
        ax.scatter(bx[~win], iy[~win], s=26, color=C_BLIND, alpha=0.85,
                   label=f"blind won ({(~win).sum()})")

        ax.set_title(f"{name}  ({unit})", fontsize=10, loc="left")
        ax.set_xlabel("blind fails at")
        ax.set_ylabel("informed fails at")
        ax.grid(alpha=0.2, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="best")

    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Every clip is its own control", fontsize=13, x=0.008,
                 ha="left")
    fig.text(0.008, -0.015,
             "One dot per clip. The dashed line is 'both detectors failed at the "
             "same strength'. Distance from it is the gain on that clip, and the "
             "spread is why the paired test matters more than the mean.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  4. gain curve
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
        by_t, thr_by_t, val_by_t = defaultdict(list), {}, {}
        for r in rows:
            if r["arm"] != arm:
                continue
            t, s, th = fnum(r["t"]), fnum(r["score"]), fnum(r["threshold"])
            if np.isfinite(t):
                val_by_t[t] = fnum(r["value"])
                if np.isfinite(s):
                    by_t[t].append(s)
                if np.isfinite(th):
                    thr_by_t[t] = th
        ts = sorted(by_t)
        if not ts:
            continue
        mean = np.array([np.mean(by_t[t]) for t in ts])
        sd = np.array([np.std(by_t[t], ddof=1) if len(by_t[t]) > 1 else 0.0
                       for t in ts])
        thr = np.array([thr_by_t.get(t, np.nan) for t in ts])

        ax.plot(ts, mean, "o-", color=colour, lw=2, ms=4, label=f"{arm} score")
        ax.fill_between(ts, mean - sd, mean + sd, color=colour, alpha=0.13)
        ax.plot(ts, thr, ":", color=colour, lw=1.4,
                label="threshold at 1% false positives")

        g = mean - thr
        xc = float("nan")
        for j in range(len(ts) - 1):
            if np.isfinite(g[j]) and np.isfinite(g[j + 1]) and g[j] > 0 >= g[j + 1]:
                xc = ts[j] + (0 - g[j]) * (ts[j + 1] - ts[j]) / (g[j + 1] - g[j])
                break
        crossings[arm] = xc
        if np.isfinite(xc):
            ax.axvline(xc, color=colour, ls="--", lw=1.2)
        ax.set_ylabel(f"{arm} score")
        ax.grid(alpha=0.22, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        if arm == "blind":
            sec = ax.secondary_xaxis("top")
            sel = np.linspace(0, len(ts) - 1, min(6, len(ts))).astype(int)
            sec.set_xticks([ts[i] for i in sel])
            sec.set_xticklabels([f"{val_by_t.get(ts[i], np.nan):g}" for i in sel],
                                fontsize=8)
            sec.set_xlabel(f"attack strength ({unit})", fontsize=9)

    b, i = crossings.get("blind", np.nan), crossings.get("informed", np.nan)
    if np.isfinite(b) and np.isfinite(i) and i > b:
        for ax in axes:
            ax.axvspan(b, i, color=C_BAND, alpha=0.13, zorder=0)
        axes[0].annotate(f"informed survives\n{i - b:.2f} further",
                         xy=((b + i) / 2, 0.5), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=9, color="#15803d")

    axes[1].set_xlabel("attack strength  (0 = weakest, 1 = strongest)")
    axes[0].set_title(f"Where each detector fails  [{attack}]",
                      fontsize=12, loc="left")
    fig.text(0.01, -0.01,
             "Both arms see the identical attacked file; only the detector "
             "differs. Thresholds are matched at 1% false positives, so neither "
             "is more willing to say yes than the other.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  5. null separation
# --------------------------------------------------------------------------- #
def fig_null_separation(attack):
    raw = read_csvs(os.path.join(NULL_DIR, f"nullraw_{attack}.csv"))
    curve = [r for r in read_csvs(os.path.join(SWEEP_DIR, "curve_clip*.csv"))
             if r["attack"] == attack]
    if not raw:
        print(f"  null_separation: no nullraw_{attack}.csv "
              f"(older runs did not save raw scores -- re-run Stage 1)")
        return None

    strengths = sorted({fnum(r["strength"]) for r in raw})
    target = strengths[len(strengths) // 2] if strengths else 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
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

        allv = neg + (pos or neg)
        bins = np.linspace(min(allv), max(allv), 40)
        ax.hist(neg, bins=bins, color="#9ca3af", alpha=0.8,
                label=f"no watermark (n={len(neg)})")
        if pos:
            ax.hist(pos, bins=bins, color=colour, alpha=0.6,
                    label=f"watermarked (n={len(pos)})")
        if len(neg) > 2:
            thr = np.sort(neg)[min(len(neg) - 1, int(np.ceil(0.99 * len(neg))))]
            ax.axvline(thr, color="#111827", ls="--", lw=1.2)
            ax.annotate("1% FPR", xy=(thr, 0.92),
                        xycoords=("data", "axes fraction"), rotation=90,
                        fontsize=8, ha="right", va="top")
        ax.set_title(arm, fontsize=11, loc="left")
        ax.set_xlabel("score")
        ax.grid(alpha=0.2, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("count")
    fig.suptitle(f"Are both detectors equally willing to say yes?  "
                 f"[{attack}, strength {target:.2f}]",
                 fontsize=12, x=0.008, ha="left")
    fig.text(0.008, -0.02,
             "Each threshold sits where 1% of UNWATERMARKED audio would be "
             "accepted. That is what makes two different score scales "
             "comparable.", fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def main(argv):
    os.makedirs(FIG_DIR, exist_ok=True)
    ca = get_arg(argv, "--curve-attack", "music_bed")

    figures = (
        ("overview", fig_overview),
        ("status_map", fig_status_map),
        ("paired_scatter", fig_paired_scatter),
        ("gain_curve", lambda: fig_gain_curve(ca)),
        ("null_separation", lambda: fig_null_separation(ca)),
    )
    made = 0
    for name, fn in figures:
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
        made += 1
    print(f"\n{made}/{len(figures)} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
