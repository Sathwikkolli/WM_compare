"""
ab_aware/analyze.py -- turn 440 raw rows into the numbers and the caveats.

Reads results/<RUN>/data/raw_*.csv, writes:

    data/metrics.csv     every row, merged
    summary.md           the tables
    figures/*.png        ROC, score distributions, per-condition breakdown

Dependencies are numpy + matplotlib only. Every statistic below is implemented
here rather than imported so the assumptions are visible and auditable.

============================ WHAT EACH METRIC MEANS ==========================

CONFUSION MATRIX -- reported at TWO thresholds, never one:
  * conf >= 0.5      the codebase default (detect_aware.py). Arbitrary.
  * calibrated       the smallest threshold that produces ZERO false positives
                     on the clean arm. This is the operating point you would
                     actually ship, and it is derived from the negatives rather
                     than assumed.
  Quoting only the 0.5 matrix would report an operating point nobody chose.

ROC / AUC -- threshold-free. AUC is computed as the Mann-Whitney statistic:
  the probability that a random positive scores above a random negative. 0.5 is
  chance, 1.0 is perfect separation. Immune to the threshold argument entirely.

TPR @ FPR=0 (observed) -- NOT "TPR @ FPR=1e-3". With 20 negatives the finest
  resolvable false-positive rate is 1/20 = 5%. Any claim below that is beyond
  what this data can support, so the honest statistic is the true-positive rate
  at the strictest threshold that still gives zero observed false positives --
  reported together with the Wilson interval on the FPR, which will be wide.

EER -- the threshold where FPR == FNR. Coarse here for the same reason.

WILSON INTERVAL -- 95% CI for a proportion. Used instead of the normal
  approximation because at n=20 with counts near 0 or n the normal interval is
  badly wrong (it can extend below zero). 0/20 gives roughly [0, 16%].

BOOTSTRAP CI -- resamples CLIPS, not rows. Rows from one clip share that clip's
  content and are not independent; resampling rows would understate the interval.

BIT ACCURACY vs. BINOMIAL NULL -- a blind guesser gets 50% of 20 bits right, so
  raw bit accuracy is not evidence on its own. p = P(X >= k | n=20, p=0.5) is
  the per-clip probability of doing that well by chance. Reported alongside.

McNEMAR -- clean vs. each distortion WITHIN the wm arm. Valid because it is the
  same 20 clips in both conditions (paired). Exact binomial on the discordant
  pairs, since the chi-square approximation is unreliable at these counts.
  NOT run between arms: the arms are disjoint clips, not pairs (see make_clips).

HOLM-BONFERRONI -- 10 distortions tested at once means ~40% chance of one
  spurious "significant" result at alpha=0.05 uncorrected. Holm controls that
  without being as conservative as plain Bonferroni.

=============================================================================

Usage:
    python analyze.py
    AB_RUN=2026-08-10_aware-detection-ab python analyze.py
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

from attacks_ab import ORDER, family  # noqa: E402

RUN = os.environ.get("AB_RUN", "2026-08-10_aware-detection-ab")
RUN_DIR = os.path.join(BASE, "results", RUN)
DATA_DIR = os.path.join(RUN_DIR, "data")
FIG_DIR = os.path.join(RUN_DIR, "figures")
SUMMARY = os.path.join(RUN_DIR, "summary.md")

N_BITS = 20
DEFAULT_THRESH = 0.5
N_BOOT = 10000
BOOT_SEED = 0


# --------------------------------------------------------------------------- #
#  Statistics -- implemented locally so the assumptions stay visible
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_mw(pos, neg):
    """AUC as the Mann-Whitney U statistic. Ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_points(pos, neg):
    """(fpr, tpr, threshold) at every distinct score. Threshold rule: conf >= t."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ts = np.unique(np.concatenate([pos, neg, [np.inf]]))[::-1]
    out = []
    for t in ts:
        tpr = float((pos >= t).mean()) if len(pos) else float("nan")
        fpr = float((neg >= t).mean()) if len(neg) else float("nan")
        out.append((fpr, tpr, float(t)))
    return out


def calibrated_threshold(neg):
    """Smallest threshold with ZERO false positives on the observed negatives.

    Just above the highest negative score. If a negative ever outscores every
    positive this becomes unreachable, which is itself the finding.
    """
    neg = np.asarray(neg, float)
    if len(neg) == 0:
        return DEFAULT_THRESH
    return float(np.nextafter(neg.max(), np.inf))


def eer(pos, neg):
    best, gap = None, float("inf")
    for fpr, tpr, t in roc_points(pos, neg):
        fnr = 1.0 - tpr
        if abs(fpr - fnr) < gap:
            gap, best = abs(fpr - fnr), ((fpr + fnr) / 2, t)
    return best if best else (float("nan"), float("nan"))


def boot_ci(fn, pos_by_clip, neg_by_clip, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap CI resampling CLIPS. fn(pos_scores, neg_scores) -> statistic."""
    rng = np.random.RandomState(seed)
    pk, nk = list(pos_by_clip), list(neg_by_clip)
    if not pk or not nk:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        p = np.concatenate([pos_by_clip[pk[i]] for i in rng.randint(0, len(pk), len(pk))])
        n = np.concatenate([neg_by_clip[nk[i]] for i in rng.randint(0, len(nk), len(nk))])
        v = fn(p, n)
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p). Exact."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(int(math.ceil(k)), n + 1)))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    m, adj, prev = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


# --------------------------------------------------------------------------- #
#  Load
# --------------------------------------------------------------------------- #
def load_rows():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "raw_*.csv")))
    if not files:
        raise SystemExit(f"no raw_*.csv in {DATA_DIR} -- has the array finished?")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                for key in ("conf", "bit_acc", "pesq", "snr_db", "seconds"):
                    r[key] = float(r[key]) if r[key] not in ("", None) else None
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} files")
    return rows


def usable(rows, cond=None, arm=None):
    out = [r for r in rows if r["status"] == "ok" and r["conf"] is not None]
    if cond:
        out = [r for r in out if r["condition"] == cond]
    if arm:
        out = [r for r in out if r["arm"] == arm]
    return out


def by_clip(rows):
    d = {}
    for r in rows:
        d.setdefault(r["clip_id"], []).append(r["conf"])
    return {k: np.asarray(v, float) for k, v in d.items()}


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def confusion(pos, neg, t):
    tp = int((np.asarray(pos) >= t).sum()); fn = len(pos) - tp
    fp = int((np.asarray(neg) >= t).sum()); tn = len(neg) - fp
    return tp, fn, fp, tn


def fmt_pct(k, n):
    """Rate WITHOUT a confidence interval.

    Deliberate. These counts pool 11 conditions per clip, so a Wilson interval on
    n=440 would claim a precision the design cannot deliver -- 0/220 would print
    as [0, 1.7%] when the honest clip-level interval is [0, 16.1%]. Intervals are
    reported at clip level, next to the matrix, where the unit is independent.
    """
    if n == 0:
        return "n/a"
    return f"{k}/{n} = {100*k/n:.1f}%"


def main():
    rows = load_rows()
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L = []
    A = L.append
    A(f"# AWARE detection A/B -- {RUN}\n")
    A("20 watermarked + 20 unwatermarked Emilia clips, 10 distortions + clean "
      "control.\n")
    A("**Read the caveat in `Limits` before quoting any false-positive number.**\n")

    n_skip = sum(1 for r in rows if r["status"] == "skip")
    n_err = sum(1 for r in rows if r["status"] == "error")
    if n_skip or n_err:
        A(f"\n`{n_skip}` skipped (attack unavailable), `{n_err}` errored. "
          "These are excluded from every statistic below.\n")

    # ---------------- headline: pooled over all conditions ----------------- #
    pos_all = [r["conf"] for r in usable(rows, arm="wm")]
    neg_all = [r["conf"] for r in usable(rows, arm="clean")]
    t_cal = calibrated_threshold(neg_all)
    auc_all = auc_mw(pos_all, neg_all)
    lo, hi = boot_ci(auc_mw, by_clip(usable(rows, arm="wm")),
                     by_clip(usable(rows, arm="clean")))
    ee, ee_t = eer(pos_all, neg_all)

    A("\n## Headline (pooled over all 11 conditions)\n")
    A(f"- **AUC = {auc_all:.4f}**  95% CI [{lo:.4f}, {hi:.4f}] (clip bootstrap, "
      f"{N_BOOT} draws)")
    A(f"- **EER = {100*ee:.1f}%** at conf {ee_t:.4f}")
    A(f"- Calibrated threshold (zero FP on the clean arm): **conf >= {t_cal:.4f}**")
    A(f"- Default threshold in `detect_aware.py`: conf >= {DEFAULT_THRESH}\n")
    A("Pooling mixes an undistorted control with attacks that are expected to "
      "destroy the signal, so the per-condition table below is the one to read. "
      "The pooled number is here only to be cited as a single figure.\n")

    # ---------------- confusion matrices ----------------------------------- #
    for label, t in (("default `conf >= 0.5`", DEFAULT_THRESH),
                     (f"calibrated `conf >= {t_cal:.4f}`", t_cal)):
        tp, fn, fp, tn = confusion(pos_all, neg_all, t)
        A(f"\n### Confusion matrix -- {label}\n")
        A("| | predicted wm | predicted clean |")
        A("|---|---|---|")
        A(f"| **actual wm** | TP {tp} | FN {fn} |")
        A(f"| **actual clean** | FP {fp} | TN {tn} |")
        A("")
        A(f"- TPR (recall): {fmt_pct(tp, tp+fn)}")
        A(f"- FPR: {fmt_pct(fp, fp+tn)}")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        A(f"- Precision: {100*prec:.1f}%" if prec == prec else "- Precision: n/a")
        A(f"- Accuracy: {fmt_pct(tp+tn, tp+fn+fp+tn)}")

        # Clip-level FPR: a clean clip counts as a false positive if ANY of its
        # 11 conditions fires. The clip IS the independent unit, so this is the
        # only false-positive number here that carries an honest interval.
        neg_clips = by_clip(usable(rows, arm="clean"))
        fired = sum(1 for v in neg_clips.values() if (np.asarray(v) >= t).any())
        flo, fhi = wilson(fired, len(neg_clips))
        A(f"- **Clip-level FPR** (clip counts as FP if any condition fires): "
          f"{fired}/{len(neg_clips)} = {100*fired/max(1,len(neg_clips)):.1f}%  "
          f"95% CI [{100*flo:.1f}%, {100*fhi:.1f}%]")
        A("\nThe row counts above pool 11 conditions per clip and are **descriptive "
          "only** -- they are not 440 independent trials, so no interval is quoted "
          "on them. The clip-level FPR is the one to cite. See Limits.")

    # ---------------- per-condition ---------------------------------------- #
    A("\n## Per condition\n")
    A("`TPR@cal` uses the calibrated threshold. `bit_acc` is over the wm arm "
      "only; `p_bits` is P(X >= k | n=20, p=0.5) for the median clip -- the "
      "chance of doing that well by guessing.\n")
    A("| condition | family | AUC | TPR@0.5 | TPR@cal | FPR@cal | bit_acc | p_bits | PESQ (wm) |")
    A("|---|---|---|---|---|---|---|---|---|")

    per_cond = {}
    for cond in ORDER:
        p = [r["conf"] for r in usable(rows, cond=cond, arm="wm")]
        n = [r["conf"] for r in usable(rows, cond=cond, arm="clean")]
        if not p and not n:
            A(f"| {cond} | {family(cond)} | -- all skipped/errored -- |||||||")
            continue
        a = auc_mw(p, n)
        tp5 = int((np.asarray(p) >= DEFAULT_THRESH).sum())
        tpc = int((np.asarray(p) >= t_cal).sum())
        fpc = int((np.asarray(n) >= t_cal).sum())
        accs = [r["bit_acc"] for r in usable(rows, cond=cond, arm="wm")
                if r["bit_acc"] is not None]
        med = float(np.median(accs)) if accs else float("nan")
        pb = binom_sf(round(med * N_BITS), N_BITS) if med == med else float("nan")
        pes = [r["pesq"] for r in usable(rows, cond=cond, arm="wm") if r["pesq"] is not None]
        pes_s = f"{np.mean(pes):.2f}" if pes else "n/a"
        per_cond[cond] = dict(auc=a, tpr5=tp5, npos=len(p), tprc=tpc,
                              fpc=fpc, nneg=len(n), bit=med, p=pb)
        A(f"| {cond} | {family(cond)} | {a:.3f} | {tp5}/{len(p)} | {tpc}/{len(p)} "
          f"| {fpc}/{len(n)} | {med:.3f} | {pb:.2g} | {pes_s} |")

    A("\nPESQ is blank for `time_stretch_1.1` and `crop_50`: both break sample "
      "alignment, so a PESQ number there would be measuring the misalignment. "
      "Only the `clean` row's PESQ is watermark-only cost; every other row "
      "includes the attack's own damage.\n")

    # ---------------- McNemar: clean vs each attack, wm arm ---------------- #
    A("\n## Does each distortion significantly hurt detection?\n")
    A("McNemar exact, paired on the same 20 watermarked clips, detection at the "
      "calibrated threshold. `b` = detected clean but lost after the attack; "
      "`c` = the reverse.\n")
    A("| distortion | b (lost) | c (gained) | p | Holm-adj p | significant |")
    A("|---|---|---|---|---|---|")

    base = {r["clip_id"]: r["conf"] for r in usable(rows, cond="clean", arm="wm")}
    names, ps, bcs = [], [], []
    for cond in ORDER:
        if cond == "clean":
            continue
        cur = {r["clip_id"]: r["conf"] for r in usable(rows, cond=cond, arm="wm")}
        shared = sorted(set(base) & set(cur))
        b = sum(1 for k in shared if base[k] >= t_cal and cur[k] < t_cal)
        c = sum(1 for k in shared if base[k] < t_cal and cur[k] >= t_cal)
        names.append(cond); bcs.append((b, c)); ps.append(mcnemar_exact(b, c))
    adj = holm(ps) if ps else []
    for nm, (b, c), p, q in zip(names, bcs, ps, adj):
        A(f"| {nm} | {b} | {c} | {p:.3g} | {q:.3g} | {'yes' if q < 0.05 else 'no'} |")

    # ---------------- limits ------------------------------------------------ #
    n_neg_clips = len({r["clip_id"] for r in rows if r["arm"] == "clean"})
    lo0, hi0 = wilson(0, n_neg_clips)
    A("\n## Limits\n")
    A(f"1. **{n_neg_clips} negative clips cannot certify a low FPR.** Even a "
      f"perfect 0/{n_neg_clips} leaves a 95% Wilson interval of "
      f"[{100*lo0:.1f}%, {100*hi0:.1f}%]. This run can detect a *gross* "
      "false-positive problem; it cannot support a claim like \"FPR < 1e-3\". "
      "Scaling the clean arm to 200+ clips is the only fix -- negatives cost "
      "nothing to make, since they skip the embedding step.")
    A("2. **The finest resolvable FPR is 1/%d = %.0f%%.** Any TPR@FPR figure "
      "below that is unsupported, which is why none is reported."
      % (n_neg_clips, 100.0 / n_neg_clips))
    A("3. **The two arms are disjoint clips, not matched pairs.** Differences "
      "between arms carry clip-level variance as well as the watermark. No "
      "paired test is run across arms. The McNemar tests above are within the "
      "wm arm, where the pairing is real.")
    A("4. **Rows are not independent.** 11 conditions share one clip each, so "
      "the 440 rows are ~40 independent units. Every CI here resamples clips "
      "for that reason; the raw counts in the confusion matrices do not, and "
      "should be read as descriptive.")
    A("5. **One strength per attack.** Where a cell fails, this run does not "
      "say how far it was from passing -- that needs a strength sweep.")

    open(SUMMARY, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {SUMMARY}")

    make_figures(rows, pos_all, neg_all, t_cal, per_cond)


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def make_figures(rows, pos_all, neg_all, t_cal, per_cond):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipping figures: {e}")
        return

    # 1. ROC
    pts = roc_points(pos_all, neg_all)
    fpr = [p[0] for p in pts]; tpr = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_mw(pos_all, neg_all):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("AWARE detection ROC (pooled over all conditions)")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "roc.png"), dpi=150)
    plt.close(fig)

    # 2. Score distributions -- the plot that explains every other number
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(neg_all, bins=bins, alpha=.6, label=f"unwatermarked (n={len(neg_all)})")
    ax.hist(pos_all, bins=bins, alpha=.6, label=f"watermarked (n={len(pos_all)})")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="default 0.5")
    ax.axvline(t_cal, color="r", ls=":", lw=1.5, label=f"calibrated {t_cal:.3f}")
    ax.set_xlabel("detector confidence"); ax.set_ylabel("count")
    ax.set_title("Confidence distribution by arm")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "score_dist.png"), dpi=150)
    plt.close(fig)

    # 3. Per-condition TPR and AUC
    conds = [c for c in ORDER if c in per_cond]
    if conds:
        tprs = [per_cond[c]["tprc"] / max(1, per_cond[c]["npos"]) for c in conds]
        aucs = [per_cond[c]["auc"] for c in conds]
        x = np.arange(len(conds))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - .2, tprs, .4, label="TPR @ calibrated")
        ax.bar(x + .2, aucs, .4, label="AUC")
        ax.axhline(0.5, color="k", ls="--", lw=1, label="chance AUC")
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=45, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("rate")
        ax.set_title("Detection by condition")
        ax.legend(); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "per_condition.png"), dpi=150)
        plt.close(fig)

        # 4. Bit accuracy against the 50% chance floor
        bits = [per_cond[c]["bit"] for c in conds]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x, bits, .6, color="tab:green")
        ax.axhline(0.5, color="r", ls="--", lw=1.5, label="chance (50%)")
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=45, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("median bit accuracy")
        ax.set_title(f"Payload recovery ({N_BITS} bits) -- bars near the red line carry no information")
        ax.legend(); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "bit_acc.png"), dpi=150)
        plt.close(fig)

    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
