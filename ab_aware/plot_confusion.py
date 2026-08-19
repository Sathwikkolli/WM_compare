"""
ab_aware/plot_confusion.py -- confusion matrices as figures.

analyze.py prints the pooled confusion matrix as a markdown table. This draws
it, because a 2x2 with the four outcomes named is the fastest way to show what
the detector does -- and in particular that the bottom row (the unwatermarked
clips) exists at all, which is the point of this experiment.

Imports analyze.py rather than recomputing, so the numbers cannot drift from
summary.md.

TWO MODES
  default   one figure, two big panels: default threshold vs calibrated.
            Pooled over all conditions unless --condition is given.
  --grid    one small matrix per condition (11) plus POOLED, in one figure.
            Written at BOTH thresholds -> two files. This is the "everything"
            view: every attack and the clean control, side by side.

WHAT THE FOUR CELLS MEAN
  TP  watermarked, and we said so                  -- correct
  FN  watermarked, and we missed it                -- a pirate walks
  FP  NOT watermarked, and we flagged it anyway    -- a wrongful accusation
  TN  not watermarked, correctly ignored           -- correct

FP is the expensive error, and it is the one no earlier benchmark in this repo
could measure, because they scored watermarked audio only.

PESQ ON EVERY PANEL

Each matrix carries the mean wideband PESQ of its watermarked arm, measured
against the unwatermarked source. It is there so a failing matrix is never read
without its audio quality: a condition that misses most positives AND scores
poorly on PESQ has damaged the audio along with the watermark, while one that
misses positives at HIGH PESQ is a genuine vulnerability -- good audio that
evades detection. Those are opposite findings and the counts alone cannot tell
them apart.

Two cautions on reading it:
  * PESQ is blank for time_stretch_1.1 and crop_50 -- both change length, so the
    score would measure misalignment rather than damage.
  * PESQ measures perceptual QUALITY, not intelligibility. Speech can score
    poorly and remain perfectly usable to an adversary (high-pass filtering is
    the classic case). Low PESQ is therefore suggestive, not proof, that an
    attack is a non-threat -- STOI is what would settle it.

WHICH NEGATIVES CALIBRATE THE THRESHOLD (this bit matters)

  default            ALL negatives, pooled over every condition -- what
                     analyze.py does, so figures match summary.md's `TPR@cal`.
  --local-threshold  only the chosen condition's negatives. Lower, so it
                     recovers more true positives -- but it assumes you KNOW
                     which attack happened, which you do not in deployment.
                     An upper bound on what is recoverable, never a deployable
                     number. Only applies with --condition.

  For highpass_0.2 the two differ sharply: pooled 0.2463 -> 4/20,
  condition-local 0.1623 -> 11/20. Same data, same detector; only the
  calibration set changed. Never show both without saying which is which.

Usage:
    python plot_confusion.py                                             # pooled, 2 panels
    python plot_confusion.py --grid                                      # ALL conditions
    python plot_confusion.py --condition highpass_0.2                    # matches summary.md
    python plot_confusion.py --condition highpass_0.2 --local-threshold  # upper bound
    AB_RUN=2026-08-10_aware-detection-ab python plot_confusion.py --grid
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
from attacks_ab import ORDER              # noqa: E402

OK = "#2e7d32"      # correct   -- green
BAD = "#c62828"     # incorrect -- red


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def _cells(ax, tp, fn, fp, tn, big):
    """Shared 2x2 drawing. `big` toggles the verbose labelling."""
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
            # intensity tracks share of the TRUE row, so an empty error cell
            # stays visibly empty rather than reading as a pale colour
            alpha = 0.12 + 0.75 * float(frac[i, j])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colour,
                                       alpha=alpha, edgecolor="white",
                                       lw=3 if big else 2))
            ax.text(j + .5, i + (.38 if big else .5), f"{counts[i, j]}",
                    ha="center", va="center",
                    fontsize=30 if big else 17,
                    fontweight="bold", color="#111")
            if big:
                ax.text(j + .5, i + .66, f"{100 * frac[i, j]:.1f}% of row",
                        ha="center", va="center", fontsize=9, color="#333")

    if big:
        labels = [["TP  correct", "FN  missed it"],
                  ["FP  false alarm", "TN  correct"]]
        for i in range(2):
            for j in range(2):
                ax.text(j + .5, i + .12, labels[i][j], ha="center",
                        va="center", fontsize=9.5, fontweight="bold",
                        color="#222")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return frac


def cond_pesq(rows, cond=None):
    """Mean PESQ over the wm arm, for one condition or pooled.

    Returns None when there is nothing to average. That is expected, not an
    error: run_ab.py leaves PESQ blank for time_stretch_1.1 and crop_50 because
    both break sample alignment, so a number there would measure the
    misalignment rather than the distortion.

    Read per row: only the `clean` condition is watermark-only cost. Every
    attacked condition mixes in the attack's own damage -- which is precisely
    what makes it the right number to show beside a failing matrix.
    """
    vals = [r["pesq"] for r in usable(rows, cond=cond, arm="wm")
            if r["pesq"] is not None]
    return float(np.mean(vals)) if vals else None


def fmt_pesq(p):
    return "PESQ n/a (alignment)" if p is None else f"PESQ {p:.2f}"


def rates(tp, fn, fp, tn):
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)
    return tpr, fpr, prec, acc


def panel(ax, tp, fn, fp, tn, title, thresh, note=""):
    _cells(ax, tp, fn, fp, tn, big=True)
    ax.set_xticks([.5, 1.5])
    ax.set_xticklabels(['detector said\n"watermarked"', 'detector said\n"clean"'],
                       fontsize=10)
    ax.set_yticks([.5, 1.5])
    ax.set_yticklabels(["actually\nwatermarked", "actually\nclean"], fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    tpr, fpr, prec, acc = rates(tp, fn, fp, tn)
    ax.set_title(f"{title}\nconf >= {thresh:.4f}", fontsize=12,
                 fontweight="bold", pad=38)
    ax.text(1.0, 2.30,
            f"recall (TPR) {100*tpr:.1f}%     false alarms (FPR) {100*fpr:.1f}%\n"
            f"precision {100*prec:.1f}%     accuracy {100*acc:.1f}%"
            + (f"\n{note}" if note else ""),
            ha="center", va="top", fontsize=10.5)


def mini(ax, tp, fn, fp, tn, title, highlight=False, pesq=None):
    _cells(ax, tp, fn, fp, tn, big=False)
    ax.set_xticks([])
    ax.set_yticks([])
    tpr, fpr, _, _ = rates(tp, fn, fp, tn)
    ax.set_title(title, fontsize=11,
                 fontweight="bold" if highlight else "normal",
                 color="#c62828" if highlight else "#111", pad=6)
    # PESQ sits directly under the counts so quality is never read separately
    # from the detection result -- a failing matrix and its audio quality belong
    # in the same glance.
    ax.set_xlabel(f"caught {tp}/{tp+fn}   false alarms {fp}/{fp+tn}\n"
                  f"{fmt_pesq(pesq)}",
                  fontsize=9, labelpad=6)


def grid_figure(rows, thresh, thresh_label, out):
    """One mini matrix per condition + POOLED, all at the same threshold."""
    conds = [c for c in ORDER if usable(rows, cond=c, arm="wm")]
    panels = [(c, [r["conf"] for r in usable(rows, cond=c, arm="wm")],
                  [r["conf"] for r in usable(rows, cond=c, arm="clean")],
                  cond_pesq(rows, c))
              for c in conds]
    panels.append(("POOLED (all conditions)",
                   [r["conf"] for r in usable(rows, arm="wm")],
                   [r["conf"] for r in usable(rows, arm="clean")],
                   cond_pesq(rows)))

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.5 * nrow),
                             squeeze=False)
    for k, (name, pos, neg, pq) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        tp, fn, fp, tn = confusion(pos, neg, thresh)
        # flag any condition that loses more than half the positives
        bad = (tp / max(tp + fn, 1)) < 0.5
        mini(ax, tp, fn, fp, tn, name, highlight=bad, pesq=pq)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"AWARE detection -- confusion matrix per condition "
                 f"({thresh_label}, conf >= {thresh:.4f})",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Each panel:  top-left TP (caught)  |  top-right FN (missed)  |  "
             "bottom-left FP (false alarm)  |  bottom-right TN (correctly ignored).   "
             "Red titles = more than half the watermarked clips missed.\n"
             "PESQ is mean wideband PESQ of the watermarked arm vs. the "
             "unwatermarked source (higher = better audio; roughly <2.0 = poor). "
             "Read it next to a red panel: low PESQ means the attack also wrecked "
             "the audio.\n"
             "PESQ is unavailable for time_stretch_1.1 and crop_50 -- both change "
             "length, so the score would measure misalignment, not damage.",
             ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv):
    cond = get_arg(argv, "--condition", None)
    rows = load_rows()

    # analyze.py calibrates on ALL negatives; match it so figures agree with
    # summary.md. --local-threshold opts into the condition-only calibration.
    neg_all = [r["conf"] for r in usable(rows, arm="clean")]
    t_pooled = calibrated_threshold(neg_all)

    if "--grid" in argv:
        os.makedirs(FIG_DIR, exist_ok=True)
        grid_figure(rows, DEFAULT_THRESH, "default threshold",
                    os.path.join(FIG_DIR, "confusion_grid_default.png"))
        grid_figure(rows, t_pooled, "calibrated threshold",
                    os.path.join(FIG_DIR, "confusion_grid_calibrated.png"))
        print(f"  run={RUN}  conditions={len(ORDER)}  "
              f"pooled calibrated threshold={t_pooled:.4f}")
        return

    out = get_arg(argv, "--out", os.path.join(FIG_DIR, "confusion.png"))
    pos = [r["conf"] for r in usable(rows, cond=cond, arm="wm")]
    neg = [r["conf"] for r in usable(rows, cond=cond, arm="clean")]
    if not pos or not neg:
        raise SystemExit(f"need both arms; got {len(pos)} wm / {len(neg)} clean"
                         + (f" for condition {cond!r}" if cond else ""))

    local = "--local-threshold" in argv and cond
    t_cal = calibrated_threshold(neg) if local else t_pooled
    note = ("calibrated on THIS condition only -- upper bound, assumes the "
            "attack is known" if local else
            "strictest cut with zero false positives, all conditions")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    panel(axes[0], *confusion(pos, neg, DEFAULT_THRESH),
          title="Default threshold", thresh=DEFAULT_THRESH)
    panel(axes[1], *confusion(pos, neg, t_cal),
          title="Calibrated threshold", thresh=t_cal, note=note)

    scope = f"condition: {cond}" if cond else "pooled over all 11 conditions"
    pq = cond_pesq(rows, cond)
    fig.suptitle(f"AWARE detection -- confusion matrix  ({scope}, "
                 f"n={len(pos)} watermarked / {len(neg)} unwatermarked, "
                 f"{fmt_pesq(pq)})",
                 fontsize=13.5, y=1.0)
    fig.text(0.5, 0.015,
             "The bottom row of each matrix is the contribution of this "
             "experiment: earlier benchmarks scored watermarked audio only, so a "
             "detector stuck on \"yes\" would have looked perfect.",
             ha="center", fontsize=9.5, style="italic", color="#444")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  run={RUN}  scope={scope}  calibration="
          f"{'condition-local' if local else 'pooled'}")
    print(f"  default    {DEFAULT_THRESH:.4f} -> TP/FN/FP/TN = "
          f"{confusion(pos, neg, DEFAULT_THRESH)}")
    print(f"  calibrated {t_cal:.4f} -> TP/FN/FP/TN = {confusion(pos, neg, t_cal)}")
    print(f"  {fmt_pesq(pq)}")


if __name__ == "__main__":
    main(sys.argv[1:])
