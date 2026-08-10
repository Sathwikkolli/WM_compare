"""
ab_aware/plot_confusion.py -- the confusion matrix as a figure.

analyze.py already prints the confusion matrix as a markdown table. This draws
it, because a 2x2 with the four outcomes named is the fastest way to explain to
someone what the detector actually does -- and in particular that the right-hand
column (the unwatermarked clips) exists at all, which is the whole point of this
experiment.

It imports analyze.py rather than recomputing anything, so the figure and
summary.md can never disagree.

TWO PANELS, always:
  left   the codebase default, conf >= 0.5
  right  the calibrated threshold -- just above the highest-scoring clean clip,
         i.e. the strictest cut that still produces zero false positives
Showing one alone invites the question "why that threshold?"; showing both makes
the cost of the default visible.

WHAT THE FOUR CELLS MEAN
  TP  watermarked, and we said so                  -- correct
  FN  watermarked, and we missed it                -- a pirate walks
  FP  NOT watermarked, and we flagged it anyway    -- a wrongful accusation
  TN  not watermarked, correctly ignored           -- correct

FP is the expensive error, and it is the one no earlier benchmark in this repo
could measure, because they scored watermarked audio only.

Usage:
    python plot_confusion.py
    python plot_confusion.py --condition highpass_0.2     # one condition
    python plot_confusion.py --out /path/to/fig.png
    AB_RUN=2026-08-10_aware-detection-ab python plot_confusion.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from analyze import (                     # noqa: E402
    load_rows, usable, calibrated_threshold, confusion,
    DEFAULT_THRESH, FIG_DIR, RUN,
)

OK = "#2e7d32"      # correct   -- green
BAD = "#c62828"     # incorrect -- red


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def panel(ax, tp, fn, fp, tn, title, thresh, note=""):
    counts = np.array([[tp, fn], [fp, tn]])
    correct = np.array([[True, False], [False, True]])
    row_tot = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, np.maximum(row_tot, 1))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()

    for i in range(2):
        for j in range(2):
            colour = OK if correct[i, j] else BAD
            # intensity tracks the share of that TRUE row, so an empty error
            # cell stays visibly empty instead of reading as a light colour
            alpha = 0.12 + 0.75 * float(frac[i, j])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colour,
                                       alpha=alpha, edgecolor="white", lw=3))
            ax.text(j + .5, i + .38, f"{counts[i, j]}",
                    ha="center", va="center", fontsize=30, fontweight="bold",
                    color="#111")
            ax.text(j + .5, i + .66, f"{100 * frac[i, j]:.1f}% of row",
                    ha="center", va="center", fontsize=9, color="#333")

    labels = [["TP  correct", "FN  missed it"],
              ["FP  false alarm", "TN  correct"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + .5, i + .12, labels[i][j], ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#222")

    ax.set_xticks([.5, 1.5])
    ax.set_xticklabels(['detector said\n"watermarked"', 'detector said\n"clean"'],
                       fontsize=10)
    ax.set_yticks([.5, 1.5])
    ax.set_yticklabels(["actually\nwatermarked", "actually\nclean"], fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)

    ax.set_title(f"{title}\nconf >= {thresh:.4f}", fontsize=12,
                 fontweight="bold", pad=38)
    ax.text(1.0, 2.30,
            f"recall (TPR) {100*tpr:.1f}%     "
            f"false alarms (FPR) {100*fpr:.1f}%\n"
            f"precision {100*prec:.1f}%     accuracy {100*acc:.1f}%"
            + (f"\n{note}" if note else ""),
            ha="center", va="top", fontsize=10.5)


def main(argv):
    cond = get_arg(argv, "--condition", None)
    out = get_arg(argv, "--out", os.path.join(FIG_DIR, "confusion.png"))

    rows = load_rows()
    pos = [r["conf"] for r in usable(rows, cond=cond, arm="wm")]
    neg = [r["conf"] for r in usable(rows, cond=cond, arm="clean")]
    if not pos or not neg:
        raise SystemExit(f"need both arms; got {len(pos)} wm / {len(neg)} clean"
                         + (f" for condition {cond!r}" if cond else ""))

    t_cal = calibrated_threshold(neg)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    panel(axes[0], *confusion(pos, neg, DEFAULT_THRESH),
          title="Default threshold", thresh=DEFAULT_THRESH)
    panel(axes[1], *confusion(pos, neg, t_cal),
          title="Calibrated threshold", thresh=t_cal,
          note="strictest cut with zero false positives")

    scope = f"condition: {cond}" if cond else "pooled over all 11 conditions"
    fig.suptitle(f"AWARE detection -- confusion matrix  ({scope}, "
                 f"n={len(pos)} watermarked / {len(neg)} unwatermarked)",
                 fontsize=13.5, y=1.0)
    fig.text(0.5, 0.015,
             "The right-hand column of each matrix is the contribution of this "
             "experiment: earlier benchmarks scored watermarked audio only, so a "
             "detector stuck on \"yes\" would have looked perfect.",
             ha="center", fontsize=9.5, style="italic", color="#444")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  run={RUN}  scope={scope}")
    print(f"  default   {DEFAULT_THRESH:.4f} -> TP/FN/FP/TN = "
          f"{confusion(pos, neg, DEFAULT_THRESH)}")
    print(f"  calibrated {t_cal:.4f} -> TP/FN/FP/TN = "
          f"{confusion(pos, neg, t_cal)}")


if __name__ == "__main__":
    main(sys.argv[1:])
