"""
informed/score_phase_b.py -- does informed detection extend robustness?

Reads results/<run>/data/sweep/sweep_clip*.csv, writes summary_phase_b<suffix>.md
and data/phase_b_metrics<suffix>.csv.

    python score_phase_b.py                          # blind vs registered informed
    python score_phase_b.py --informed-arm informed16  # blind vs corrected informed
    python score_phase_b.py --fpr 0.05

THE TWO INFORMED ARMS

  informed     registered primary: windowed correlation at 22.05 kHz
  informed16   post-hoc correction: whole-clip correlation at 16 kHz. See
               informed_detector.score_16k for the two defects it removes.
               Written to files with an `_informed16` suffix and always reported
               NEXT TO the registered result, never instead of it.

THE RESULT

Per clip and attack, two crossings: the strength at which blind detection stops
clearing its FPR-matched threshold, and the same for informed. The gap is the
benefit.

    gain_t      = t_informed - t_blind       positive when informed wins
    gain_native = the same, in the attack's own units (dB, kbps, ...)

Native units are not comparable across attacks and are never averaged together.

WHY A PAIRED TEST

Clip variance is enormous (22 dB across 50 clips for music). Each clip carries
BOTH arms, so each is its own control; Wilcoxon signed-rank on the paired
differences.

STATUS HANDLING -- the part that decides whether the numbers mean anything

  CROSSED                a crossing was found; a number
  NO_CROSSING_SURVIVED   still detected at maximum strength (a lower bound)
  NO_CROSSING_FAILED     already failed at the weakest setting
  NON_MONOTONE           detection failed and then RECOVERED as the attack got
                         stronger, so "the" crossing does not exist
  UNAVAILABLE            the attack could not run

WHO WON A CLIP -- `outcome()`

Every combination is decided, not only both-crossed. Detectors are ranked by how
far along the axis they lasted: FAILED < CROSSED < SURVIVED. Two CROSSED are
compared by t. The earlier version only credited informed for censored wins, so
"blind survived while informed broke" and "one failed at t=0 while the other
worked" were both shown as no data.

Only clips where BOTH arms report CROSSED contribute a paired difference.
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

RUN_SLUG = os.environ.get("PHASEB_RUN", "2026-08-28_informed-detection")
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
SWEEP_DIR = os.path.join(DATA_DIR, "sweep")

# Registered in PHASE_B_PLAN.md prediction 5.
WIN_FRACTION_TARGET = 0.80

INFORMED_ARMS = ("informed", "informed16")
OUTCOMES = ("informed", "blind", "tie", "both_survived", "both_failed",
            "non_monotone", "no_data")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def suffix_for(arm):
    return "" if arm == "informed" else f"_{arm}"


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


_RANK = {"NO_CROSSING_FAILED": 0, "CROSSED": 1, "NO_CROSSING_SURVIVED": 2}


def outcome(b, i):
    """Who won one clip. `b`, `i` are sweep rows for the blind and informed arm."""
    bs, ist = b["status"], i["status"]
    if "NON_MONOTONE" in (bs, ist):
        return "non_monotone"
    rb, ri = _RANK.get(bs), _RANK.get(ist)
    if rb is None or ri is None:
        return "no_data"
    if bs == "CROSSED" and ist == "CROSSED":
        g = fnum(i["t_cross"]) - fnum(b["t_cross"])
        if not np.isfinite(g):
            return "no_data"
        return "informed" if g > 0 else "blind" if g < 0 else "tie"
    if rb == ri:
        return "both_survived" if bs == "NO_CROSSING_SURVIVED" else "both_failed"
    return "informed" if ri > rb else "blind"


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

    `hi` may be numerically smaller (SNR, bitrate, lowpass cutoff) or larger
    (reverb, highpass cutoff), so the sign comes from the axis.
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
    inf_arm = get_arg(argv, "--informed-arm", "informed")
    if inf_arm not in INFORMED_ARMS:
        raise SystemExit(f"--informed-arm must be one of {INFORMED_ARMS}")
    sfx = suffix_for(inf_arm)
    want_fpr = get_arg(argv, "--fpr", None, float)

    rows = load()
    if want_fpr is not None:
        rows = [r for r in rows if abs(r["fpr"] - want_fpr) < 1e-9]
    fprs = sorted({r["fpr"] for r in rows if np.isfinite(r["fpr"])})

    idx = defaultdict(dict)                     # (attack, clip) -> {arm: row}
    for r in rows:
        idx[(r["attack"], r["clip_id"])][r["arm"]] = r

    attacks = sorted({r["attack"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    print(f"  {len(attacks)} attacks, {len(clips)} clips, FPR {fprs}, "
          f"blind vs {inf_arm}")

    per_attack, per_pair = [], []
    for atk in attacks:
        gains_t, gains_native = [], []
        blind_all, inf_all = [], []             # every crossing, paired or not
        counts = defaultdict(int)
        censored_informed = 0

        for cid in clips:
            pair = idx.get((atk, cid), {})
            b, i = pair.get("blind"), pair.get(inf_arm)
            if not b or not i:
                continue
            counts[outcome(b, i)] += 1
            if b["status"] == "CROSSED" and np.isfinite(b["value_cross"]):
                blind_all.append(b["value_cross"])
            if i["status"] == "CROSSED" and np.isfinite(i["value_cross"]):
                inf_all.append(i["value_cross"])
            if i["status"] == "NO_CROSSING_SURVIVED":
                censored_informed += 1
            if b["status"] != "CROSSED" or i["status"] != "CROSSED":
                continue

            gt = i["t_cross"] - b["t_cross"]
            gn = native_gain(atk, b["value_cross"], i["value_cross"])
            gains_t.append(gt)
            gains_native.append(gn)
            per_pair.append({
                "attack": atk, "clip_id": cid,
                "blind_value": b["value_cross"], "informed_value": i["value_cross"],
                "gain_t": gt, "gain_native": gn, "unit": b.get("unit", ""),
            })

        st, sn = stat(gains_t), stat(gains_native)
        _w_stat, w_p = wilcoxon(gains_t)
        n_paired = len(gains_t)
        paired_inf_wins = sum(1 for g in gains_t if g > 0)
        per_attack.append({
            "attack": atk,
            "category": {"music_bed": "additive", "gaussian_noise": "additive",
                         "noise_babble": "additive", "noise_factory": "additive",
                         "noise_machinegun": "additive"}.get(atk, _category(atk)),
            "unit": SA.AXIS.get(atk, {}).get("unit", ""),
            "informed_arm": inf_arm,
            "n_paired": n_paired, "n_clips": len(clips),
            "censored_informed": censored_informed,
            "decisive_wins": counts["informed"],          # kept for old readers
            "informed_decisive_wins": counts["informed"],
            "blind_decisive_wins": counts["blind"],
            "informed_won_censored": counts["informed"] - paired_inf_wins,
            "ties": counts["tie"],
            "both_survived": counts["both_survived"],
            "both_failed": counts["both_failed"],
            "non_monotone": counts["non_monotone"],
            "no_data": counts["no_data"] + max(
                0, len(clips) - sum(counts[o] for o in OUTCOMES)),
            "blind_median": stat(blind_all)["median"],
            "informed_median": stat(inf_all)["median"],
            "n_blind_crossed": len(blind_all), "n_informed_crossed": len(inf_all),
            "gain_t_mean": st["mean"], "gain_t_sd": st["sd"], "gain_t_median": st["median"],
            "gain_native_mean": sn["mean"], "gain_native_sd": sn["sd"],
            "gain_native_median": sn["median"],
            "win_fraction": (paired_inf_wins / n_paired) if n_paired else float("nan"),
            "wilcoxon_p": w_p,
        })

    L = []
    w = L.append
    label = ("REGISTERED informed (windowed, 22 kHz)" if inf_arm == "informed" else
             "CORRECTED informed (whole clip, 16 kHz) -- post-hoc, see score_16k")
    w(f"# Phase B — blind vs {label}\n")
    w(f"Run `{RUN_SLUG}`. {len(rows)} rows, {len(attacks)} attacks, "
      f"{len(clips)} clips, FPR {fprs}.\n")
    w("Both arms see the identical attacked file and both thresholds are set to "
      "the same false-positive rate on unwatermarked audio, so the comparison is "
      "paired and scale-fair. Metric definitions and status handling are at the "
      "top of `score_phase_b.py`.\n")

    # ---- who won ----------------------------------------------------------
    w("\n## 0. Who won, per attack  <- READ THIS FIRST\n")
    w("Every clip is decided by how far along the axis each detector lasted "
      "(failed at weakest < crossed < survived); two crossings are compared by "
      "strength. `non-monotone` = a detector failed and then recovered as the "
      "attack got stronger, so there is no single crossing to compare.\n")
    w("| attack | informed wins | blind wins | tie | both survived | both failed "
      "| non-monotone | no data | verdict |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(per_attack, key=lambda x: -(x["informed_decisive_wins"]
                                               - x["blind_decisive_wins"])):
        iw, bw = r["informed_decisive_wins"], r["blind_decisive_wins"]
        if iw + bw == 0:
            verdict = "**no decision**"
        elif iw >= 2 * max(bw, 1):
            verdict = "**INFORMED**"
        elif bw >= 2 * max(iw, 1):
            verdict = "**BLIND**"
        else:
            verdict = "mixed"
        w(f"| `{r['attack']}` | {iw} | {bw} | {r['ties']} | {r['both_survived']} "
          f"| {r['both_failed']} | {r['non_monotone']} | {r['no_data']} | {verdict} |")
    w("\n`both survived` = the axis never got strong enough. `both failed` = the "
      "axis never started weak enough. `no data` = the attack or its calibration "
      "failed, which is a bug, not a result.")

    # ---- headline ---------------------------------------------------------
    per_attack.sort(key=lambda r: (-(r["gain_t_median"] if np.isfinite(r["gain_t_median"])
                                     else -9e9), r["attack"]))
    w("\n## 1. Gain per attack, on clips where BOTH detectors crossed\n")
    w("Positive = informed helps. **median** is the headline: on log-spaced axes "
      "(bitrate, sample rate) a mean in native units can take the opposite sign "
      "from the per-clip majority. Mean ± sd kept for reference.\n")
    w("| attack | category | paired | blind fails at | informed fails at "
      "| **median gain** | mean ± sd | unit | win frac | Wilcoxon p |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in per_attack:
        star = " ⚠" if r["censored_informed"] else ""
        w(f"| `{r['attack']}`{star} | {r['category']} | {r['n_paired']}/{r['n_clips']} "
          f"| {fmt(r['blind_median'], 4)} | {fmt(r['informed_median'], 4)} "
          f"| **{fmt(r['gain_native_median'], 3)}** "
          f"| {fmt(r['gain_native_mean'], 3)} ± {fmt(r['gain_native_sd'], 2)} "
          f"| {r['unit']} | {fmt(r['win_fraction'], 2)} | {fmt(r['wilcoxon_p'], 4)} |")
    w("\n⚠ = some clips had informed survive the whole axis; for those the gain is "
      "a lower bound and they are counted in section 0, not averaged here.")

    # ---- prediction 5 -----------------------------------------------------
    w("\n## 2. Paired test  <- PREDICTION 5\n")
    w(f"Prediction 5 expects informed to win on **≥{WIN_FRACTION_TARGET:.0%} of "
      f"individual clips**.\n")
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
    w("| category | attacks | median gain_t (median of attacks) | informed-won attacks "
      "| blind-won attacks |")
    w("|---|---|---|---|---|")
    bycat = defaultdict(list)
    for r in per_attack:
        bycat[r["category"]].append(r)
    for cat in sorted(bycat):
        rs = bycat[cat]
        gt = stat([r["gain_t_median"] for r in rs])
        iwin = sum(1 for r in rs if r["informed_decisive_wins"] > r["blind_decisive_wins"])
        bwin = sum(1 for r in rs if r["blind_decisive_wins"] > r["informed_decisive_wins"])
        w(f"| {cat} | {len(rs)} | {fmt(gt['median'], 4)} | {iwin}/{len(rs)} "
          f"| {bwin}/{len(rs)} |")

    # ---- integrity --------------------------------------------------------
    w("\n## 4. Status accounting\n")
    w("| attack | paired | informed won, censored | blind crossed (any) "
      "| informed crossed (any) | non-monotone | no data |")
    w("|---|---|---|---|---|---|---|")
    for r in sorted(per_attack, key=lambda x: x["attack"]):
        w(f"| `{r['attack']}` | {r['n_paired']} | {r['informed_won_censored']} "
          f"| {r['n_blind_crossed']} | {r['n_informed_crossed']} "
          f"| {r['non_monotone']} | {r['no_data']} |")
    total_nd = sum(r["no_data"] for r in per_attack)
    total_nm = sum(r["non_monotone"] for r in per_attack)
    w(f"\n**{sum(r['n_paired'] for r in per_attack)} paired, {total_nm} non-monotone, "
      f"{total_nd} no data.**")
    if total_nd:
        w("\n**Any `no data` is a pipeline failure.** Find it before quoting results.")

    w("\n## Conclusion\n")
    w("*(write this by hand after reading the tables — results/README.md rule 2)*")

    os.makedirs(DATA_DIR, exist_ok=True)
    if per_pair:
        cols = sorted({k for d in per_pair for k in d})
        with open(os.path.join(DATA_DIR, f"phase_b_pairs{sfx}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per_pair)
    if per_attack:
        cols = sorted({k for d in per_attack for k in d})
        with open(os.path.join(DATA_DIR, f"phase_b_metrics{sfx}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per_attack)

    out_md = os.path.join(RESULTS_DIR, f"summary_phase_b{sfx}.md")
    txt = "\n".join(L) + "\n"
    with open(out_md, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {out_md}")


def _category(attack):
    try:
        import attacks_screen as A
        return A.CATEGORY.get(SA.base_attack(attack), "uncategorised")
    except Exception:
        return "uncategorised"


if __name__ == "__main__":
    main(sys.argv[1:])
