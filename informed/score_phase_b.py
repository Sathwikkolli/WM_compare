"""
informed/score_phase_b.py -- does informed detection extend robustness?

Reads results/<run>/data/sweep/sweep_clip*.csv, writes summary_phase_b.md and
data/phase_b_metrics.csv.

THE RESULT

Per clip and attack, two crossings: the strength at which blind detection stops
clearing its FPR-matched threshold, and the same for informed. The gap is the
benefit.

    gain_t     = t_informed - t_blind        always positive when informed wins
    gain_native = the same, in the attack's own units (dB, kbps, ...)

`t` runs 0 (weakest) to 1 (strongest), so gain_t is directly comparable across
attacks. Native units are not comparable across attacks -- "5 dB of music" and
"20 kbps of MP3" share nothing -- so they are reported per attack and never
averaged together.

WHY A PAIRED TEST

Clip variance is enormous: the Phase A music sweep found a 22 dB range across 50
clips. Comparing two means would drown the effect. Each clip carries BOTH arms,
so each is its own control, and a Wilcoxon signed-rank test on the 50 paired
differences is both the right test and a far stronger one.

STATUS HANDLING -- the part that decides whether the numbers mean anything

  CROSSED                both arms found a crossing; the gain is a number
  NO_CROSSING_SURVIVED   still detected at maximum strength. The gain is a LOWER
                         BOUND. Counted and reported separately, never folded
                         into a mean, and never treated as "crossed at t=1".
  NO_CROSSING_FAILED     already failed at the weakest setting -- the axis does
                         not start weak enough for this clip
  UNAVAILABLE            the attack could not run

Only clips where BOTH arms report CROSSED contribute a paired difference. If a
large share are censored the mean is not the headline and the summary says so.

Usage:
    python score_phase_b.py
    python score_phase_b.py --fpr 0.05
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

import strength_axis as SA                      # noqa: E402

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
SWEEP_DIR = os.path.join(DATA_DIR, "sweep")

# Registered in PHASE_B_PLAN.md prediction 5.
WIN_FRACTION_TARGET = 0.80


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load():
    files = sorted(glob.glob(os.path.join(SWEEP_DIR, "sweep_clip*.csv")))
    if not files:
        raise SystemExit(f"no sweep_clip*.csv in {SWEEP_DIR} -- run bisect_sweep.py")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("t_cross", "value_cross", "score_at_t0", "thr_at_t0",
                          "score_at_t1", "thr_at_t1", "fpr"):
                    r[k] = fnum(r.get(k))
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} file(s)")
    return rows


def native_gain(attack, v_blind, v_informed):
    """Gain in the attack's own units, positive when informed tolerates more.

    Direction is implied by the axis: `hi` may be numerically smaller (SNR,
    bitrate) or larger (cutoff, reverb), so the sign has to be taken from the
    axis rather than assumed.
    """
    ax = SA.AXIS.get(attack)
    if ax is None or not (np.isfinite(v_blind) and np.isfinite(v_informed)):
        return float("nan")
    direction = 1.0 if float(ax["hi"]) > float(ax["lo"]) else -1.0
    return (v_informed - v_blind) * direction


def wilcoxon(diffs):
    """(statistic, p) for a two-sided signed-rank test. (nan, nan) if too few."""
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0.0], dtype=float)
    if len(d) < 6:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        s, p = _w(d, alternative="two-sided")
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


def stat(vals):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return dict(n=0, mean=float("nan"), sd=float("nan"),
                    median=float("nan"), lo=float("nan"), hi=float("nan"))
    a = np.array(v, dtype=float)
    return dict(n=len(v), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                median=float(np.median(a)), lo=float(a.min()), hi=float(a.max()))


def fmt(v, nd=3, dash="-"):
    return f"{v:.{nd}f}" if np.isfinite(v) else dash


def main(argv):
    want_fpr = get_arg(argv, "--fpr", None, float)
    rows = load()
    if want_fpr is not None:
        rows = [r for r in rows if abs(r["fpr"] - want_fpr) < 1e-9]
    fprs = sorted({r["fpr"] for r in rows if np.isfinite(r["fpr"])})

    # index: (attack, clip) -> {arm: row}
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["attack"], r["clip_id"])][r["arm"]] = r

    attacks = sorted({r["attack"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    print(f"  {len(attacks)} attacks, {len(clips)} clips, FPR {fprs}")

    per_attack, per_pair = [], []
    for atk in attacks:
        gains_t, gains_native, blind_v, inf_v = [], [], [], []
        # Every crossing found, regardless of whether the partner arm crossed.
        # The old table blanked blind's number whenever informed was censored,
        # which hid a perfectly good measurement behind a dash.
        blind_all, inf_all = [], []
        status_count = defaultdict(int)
        censored_informed = 0
        # "informed survived" only means informed WON if blind did not also
        # survive. Counting them together made mp3 unreadable: 23 censored
        # against 24 paired losses, with no way to tell which way it went.
        inf_won_censored = 0      # informed survived, blind crossed
        both_survived = 0         # neither crossed: the attack was too weak

        for cid in clips:
            pair = idx.get((atk, cid), {})
            b, i = pair.get("blind"), pair.get("informed")
            if not b or not i:
                continue
            status_count[f"{b['status']}/{i['status']}"] += 1
            if b["status"] == "CROSSED" and np.isfinite(b["value_cross"]):
                blind_all.append(b["value_cross"])
            if i["status"] == "CROSSED" and np.isfinite(i["value_cross"]):
                inf_all.append(i["value_cross"])
            if i["status"] == "NO_CROSSING_SURVIVED":
                if b["status"] == "CROSSED":
                    inf_won_censored += 1
                elif b["status"] == "NO_CROSSING_SURVIVED":
                    both_survived += 1

            # A survived-to-maximum informed arm is a LOWER BOUND on the gain,
            # not a data point. Counted, never averaged in.
            if i["status"] == "NO_CROSSING_SURVIVED":
                censored_informed += 1
            if b["status"] != "CROSSED" or i["status"] != "CROSSED":
                continue

            gt = i["t_cross"] - b["t_cross"]
            gn = native_gain(atk, b["value_cross"], i["value_cross"])
            gains_t.append(gt)
            gains_native.append(gn)
            blind_v.append(b["value_cross"])
            inf_v.append(i["value_cross"])
            per_pair.append({
                "attack": atk, "clip_id": cid,
                "blind_value": b["value_cross"], "informed_value": i["value_cross"],
                "gain_t": gt, "gain_native": gn,
                "unit": b.get("unit", ""),
            })

        st, sn = stat(gains_t), stat(gains_native)
        w_stat, w_p = wilcoxon(gains_t)
        wins = sum(1 for g in gains_t if g > 0)
        n_paired = len(gains_t)
        per_attack.append({
            "attack": atk,
            "category": {"music_bed": "additive", "gaussian_noise": "additive",
                         "noise_babble": "additive", "noise_factory": "additive",
                         "noise_machinegun": "additive"}.get(
                             atk, _category(atk)),
            "unit": SA.AXIS.get(atk, {}).get("unit", ""),
            "n_paired": n_paired, "n_clips": len(clips),
            "censored_informed": censored_informed,
            "informed_won_censored": inf_won_censored,
            "both_survived": both_survived,
            # medians over ALL crossings, not only paired ones
            "blind_median": stat(blind_all)["median"],
            "informed_median": stat(inf_all)["median"],
            "n_blind_crossed": len(blind_all), "n_informed_crossed": len(inf_all),
            # Decisive wins = paired wins + clips where informed survived and
            # blind did not. This is the number that answers "who won".
            "decisive_wins": sum(1 for g in gains_t if g > 0) + inf_won_censored,
            "gain_t_mean": st["mean"], "gain_t_sd": st["sd"],
            "gain_native_mean": sn["mean"], "gain_native_sd": sn["sd"],
            "gain_native_median": sn["median"],
            "win_fraction": (wins / n_paired) if n_paired else float("nan"),
            "wilcoxon_p": w_p,
        })

    L = []
    w = L.append
    w("# Phase B — does informed detection extend robustness?\n")
    w(f"Run `{RUN_SLUG}`. {len(rows)} rows, {len(attacks)} attacks, "
      f"{len(clips)} clips, FPR {fprs}.\n")
    w("Both arms see the identical attacked file and both thresholds are set to "
      "the same false-positive rate on unwatermarked audio, so the comparison is "
      "paired and scale-fair. **No audio-quality threshold enters this claim** — "
      "quality is constant across the two arms by construction.\n")
    w("Metric definitions and status handling are at the top of "
      "`score_phase_b.py`. Read them before quoting anything here.\n")

    # ---- who won ----------------------------------------------------------
    w("\n## 0. Who won, per attack  <- READ THIS FIRST\n")
    w("A clip is an **informed win** if informed crossed later than blind, OR if "
      "informed never broke while blind did. A **blind win** is a paired clip "
      "where blind crossed later.\n")
    w("This table exists because a dash elsewhere means two opposite things: "
      "informed winning so completely there was no crossing to pair with, or "
      "nothing measured at all. Those must not look the same.\n")
    w("| attack | informed wins | blind wins | both survived | no data | verdict |")
    w("|---|---|---|---|---|---|")
    for r in sorted(per_attack, key=lambda x: -(x.get("decisive_wins") or 0)):
        n_paired = r["n_paired"]
        wf = r["win_fraction"] if np.isfinite(r["win_fraction"]) else 0.0
        paired_wins = int(round(wf * n_paired))
        iw = r.get("decisive_wins", 0)
        bw = n_paired - paired_wins                 # paired clips blind won
        bs = r.get("both_survived", 0)
        nd = max(0, r["n_clips"] - iw - bw - bs)

        if iw + bw == 0:
            verdict = "**no data**"
        elif iw >= 2 * max(bw, 1):
            verdict = "**INFORMED**"
        elif bw >= 2 * max(iw, 1):
            verdict = "**BLIND**"
        else:
            verdict = "mixed"
        w(f"| `{r['attack']}` | {iw} | {bw} | {bs} | {nd} | {verdict} |")
    w("\n`both survived` = neither detector broke anywhere on the axis, so the "
      "attack never got strong enough to decide anything and the axis needs "
      "widening. `no data` = the attack or its calibration failed.")

    # ---- headline ---------------------------------------------------------
    per_attack.sort(key=lambda r: (-(r["gain_t_mean"] if np.isfinite(r["gain_t_mean"])
                                     else -9e9), r["attack"]))
    w("\n## 1. Gain per attack  <- THE RESULT\n")
    w("`gain` is how much further informed detection survives along the strength "
      "axis. Positive = informed helps. Native units are per attack and must not "
      "be averaged across them.\n")
    w("| attack | category | n | blind fails at | informed fails at | **gain** | unit | win frac | Wilcoxon p |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in per_attack:
        star = " ⚠" if r["censored_informed"] else ""
        w(f"| `{r['attack']}`{star} | {r['category']} | {r['n_paired']}/{r['n_clips']} "
          f"| {fmt(r['blind_median'], 4)} | {fmt(r['informed_median'], 4)} "
          f"| **{fmt(r['gain_native_mean'], 3)}** ± {fmt(r['gain_native_sd'], 2)} "
          f"| {r['unit']} | {fmt(r['win_fraction'], 2)} "
          f"| {fmt(r['wilcoxon_p'], 4)} |")
    w("\n⚠ = some clips had informed detection survive the whole axis. For those "
      "the gain is a **lower bound**; they are excluded from the mean and counted "
      "in section 4.")

    # ---- prediction 5 -----------------------------------------------------
    w(f"\n## 2. Paired test  <- PREDICTION 5\n")
    w(f"Prediction 5 expects informed to win on **≥{WIN_FRACTION_TARGET:.0%} of "
      f"individual clips**, not merely on the mean. Each clip carries both arms, "
      f"so each is its own control.\n")
    w("| attack | win fraction | meets ≥80%? | Wilcoxon p | significant at 0.05? |")
    w("|---|---|---|---|---|")
    for r in per_attack:
        if not r["n_paired"]:
            continue
        meets = ("yes" if np.isfinite(r["win_fraction"])
                 and r["win_fraction"] >= WIN_FRACTION_TARGET else "no")
        sig = ("yes" if np.isfinite(r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05
               else "no" if np.isfinite(r["wilcoxon_p"]) else "-")
        w(f"| `{r['attack']}` | {fmt(r['win_fraction'], 2)} | {meets} "
          f"| {fmt(r['wilcoxon_p'], 4)} | {sig} |")

    # ---- by category ------------------------------------------------------
    w("\n## 3. By category  <- PREDICTIONS 1, 3, 4\n")
    w("Prediction 1: additive attacks show a positive gain (the central claim). "
      "Prediction 3: codec attacks show ~zero. Prediction 4: filtering shows zero "
      "or negative with scalar host removal.\n")
    w("| category | attacks | mean gain_t | attacks with positive gain |")
    w("|---|---|---|---|")
    bycat = defaultdict(list)
    for r in per_attack:
        bycat[r["category"]].append(r)
    for cat in sorted(bycat):
        rs = bycat[cat]
        gt = stat([r["gain_t_mean"] for r in rs])
        pos = sum(1 for r in rs if np.isfinite(r["gain_t_mean"]) and r["gain_t_mean"] > 0)
        w(f"| {cat} | {len(rs)} | {fmt(gt['mean'], 4)} | {pos}/{len(rs)} |")

    # ---- censoring / integrity -------------------------------------------
    w("\n## 4. Status accounting\n")
    w("A crossing that was never found is not a data point. These counts decide "
      "whether the means above are the headline or a footnote.\n")
    w("| attack | paired | informed survived, blind broke | both survived "
      "| blind crossed (any) | informed crossed (any) |")
    w("|---|---|---|---|---|---|")
    for r in per_attack:
        w(f"| `{r['attack']}` | {r['n_paired']} "
          f"| **{r.get('informed_won_censored', 0)}** "
          f"| {r.get('both_survived', 0)} "
          f"| {r.get('n_blind_crossed', 0)} | {r.get('n_informed_crossed', 0)} |")

    total_pairs = sum(r["n_paired"] for r in per_attack)
    total_cens = sum(r["censored_informed"] for r in per_attack)
    w(f"\n**{total_pairs} usable pairs, {total_cens} censored** "
      f"(informed survived to maximum strength).")
    if total_cens > 0.2 * max(1, total_pairs + total_cens):
        w("\n**More than a fifth of cases are censored.** The means above "
          "UNDERSTATE the gain, because the cases where informed did best are "
          "exactly the ones excluded. Widen the strength axes and re-run before "
          "quoting a number.")

    w("\n## Conclusion\n")
    w("*(write this by hand after reading the tables — results/README.md rule 2)*")

    os.makedirs(DATA_DIR, exist_ok=True)
    if per_pair:
        cols = sorted({k for d in per_pair for k in d})
        with open(os.path.join(DATA_DIR, "phase_b_pairs.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per_pair)
    if per_attack:
        cols = sorted({k for d in per_attack for k in d})
        with open(os.path.join(DATA_DIR, "phase_b_metrics.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per_attack)

    txt = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary_phase_b.md"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {os.path.join(RESULTS_DIR, 'summary_phase_b.md')}")


def _category(attack):
    try:
        import attacks_screen as A
        return A.CATEGORY.get(SA.base_attack(attack), "uncategorised")
    except Exception:
        return "uncategorised"


if __name__ == "__main__":
    main(sys.argv[1:])
