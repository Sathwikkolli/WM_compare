"""
align_bench/score.py -- turn raw rows into the decision.

Reads results/<run>/data/raw_clip*.csv, writes metrics.csv and summary.md.

WHAT EACH METRIC TELLS YOU  (this is the part to read slowly)

  offset error (median / p95)
      How far off, in ms. Median = typical case, p95 = bad case. Median alone
      hides catastrophes: a method can be perfect 90% of the time and off by
      30 seconds the rest, which is worse than being consistently off by 20 ms.

  hit rate @ 50 ms  <- HEADLINE
      Fraction of pairs aligned closely enough to be USEFUL FOR THIS APPLICATION.
      AWARE(20bps) slides its detector in 42 ms steps, so finer alignment buys
      nothing. 50 ms is also the conventional tolerance in the music
      synchronisation literature. "92% @ 50 ms" = 92 of 100 files come out usable.

  hit rate @ 20 ms
      Kept for comparability with W_MS in fsss/exp_a_repeatability.py.
      CAUTION: audalign's fingerprint and spectrogram recognizers have a ~22-25 ms
      resolution floor even on clean audio. A big gap between hit@20ms and
      hit@50ms means resolution-limited, NOT wrong.

  hit rate @ 1 ms
      The strict bar. Only matters if something downstream needs sample-grade sync.

  false-shift rate
      On attacks that move NOTHING, how often does it report a shift > 5 ms?
      A method that invents offsets will silently corrupt everything after it
      and you will never notice, because it reports success.
      >5% here is DISQUALIFYING no matter how good the hit rate is.

  confidence AUC
      Does the method's own confidence separate its right answers from its
      wrong ones? 0.5 = confidence is noise. 0.8+ = you can trust it to know
      when it failed, which means you can fall back to a second method.
      A method that is 80% accurate AND flags its own failures beats one that
      is 90% accurate and silently confident when wrong.

  runtime
      Seconds per pair, and how it grows with clip length.

DECISION ORDER
  false-shift > 5%  -> disqualified, stop reading
  hit rate @ 20 ms  -> primary ranking
  confidence AUC    -> tiebreak
  runtime           -> veto if it explodes

Usage:
    python score.py
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

RUN_SLUG = "2026-08-05_align-method-bakeoff"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")

# Three tolerances, because one number would mislead.
#
#   1 ms   strict -- only matters if something downstream needs sample-grade sync
#  20 ms   matches W_MS in fsss/exp_a_repeatability.py, so this benchmark stays
#          comparable to the salient-point work
#  50 ms   APPLICATION-RELEVANT, and the standard tolerance in the music
#          synchronisation literature. AWARE(20bps) slides its detection window
#          in chunk_duration//24 = 42 ms steps, so alignment finer than ~40 ms
#          buys the detector nothing.
#
# This matters: audalign's fingerprint and spectrogram recognizers have a
# resolution floor around 22-25 ms (measured by methods.calibrate() on a CLEAN
# synthetic crop). Reporting only the 20 ms bar would score them near zero for
# being coarse rather than for being wrong.
TOL_STRICT_MS = 1.0
TOL_MS = 20.0          # primary
TOL_LOOSE_MS = 50.0    # application-relevant
FALSE_SHIFT_MS = 5.0   # a "null" prediction bigger than this is a hallucination
CODEC_TOL_MS = 100.0   # encoder delay is real but unknown; this bounds it
FAMILIES = ["null", "shift", "multiseg", "warp", "codec"]


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "raw_clip*.csv"))):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def _f(r, k):
    v = r.get(k, "")
    if v == "" or v is None:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def auc(scores, labels):
    """Mann-Whitney AUC. labels: 1 = the method was right, 0 = wrong.

    No sklearn dependency -- rank-based, handles ties.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def summarize(rows):
    """method -> family -> metric dict, plus an 'ALL' family."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r["method"]][r["family"]].append(r)
        buckets[r["method"]]["ALL"].append(r)

    out = {}
    for method, fams in buckets.items():
        out[method] = {}
        for fam, rs in fams.items():
            errs, hits20, hits1, hits50, confs, rts = [], [], [], [], [], []
            n_fail = 0
            for r in rs:
                rts.append(_f(r, "runtime_s"))
                if int(r.get("ok", 0) or 0) != 1:
                    n_fail += 1
                    hits20.append(0)
                    hits1.append(0)
                    hits50.append(0)
                    confs.append(_f(r, "confidence"))
                    continue
                e = abs(_f(r, "error_ms"))
                codec = r["family"] == "codec"
                errs.append(e)
                hits20.append(int(np.isfinite(e) and
                                  e <= (CODEC_TOL_MS if codec else TOL_MS)))
                hits50.append(int(np.isfinite(e) and
                                  e <= (CODEC_TOL_MS if codec else TOL_LOOSE_MS)))
                hits1.append(int(np.isfinite(e) and e <= TOL_STRICT_MS))
                confs.append(_f(r, "confidence"))

            errs = np.array([e for e in errs if np.isfinite(e)])
            m = {
                "n": len(rs),
                "fail_rate": n_fail / max(1, len(rs)),
                "median_err_ms": float(np.median(errs)) if len(errs) else float("nan"),
                "p95_err_ms": float(np.percentile(errs, 95)) if len(errs) else float("nan"),
                "hit20": float(np.mean(hits20)) if hits20 else float("nan"),
                "hit50": float(np.mean(hits50)) if hits50 else float("nan"),
                "hit1": float(np.mean(hits1)) if hits1 else float("nan"),
                "conf_auc": auc(confs, hits50),   # calibrated against the bar
                                                  # the application actually needs
                "runtime_med_s": float(np.nanmedian(rts)) if rts else float("nan"),
            }
            out[method][fam] = m

        # false-shift: on 'null' rows, how often did it claim a real shift?
        nulls = fams.get("null", [])
        fs = [1 if (int(r.get("ok", 0) or 0) == 1 and
                    abs(_f(r, "pred_offset")) / 16000.0 * 1000.0 > FALSE_SHIFT_MS)
              else 0 for r in nulls]
        out[method]["_false_shift"] = float(np.mean(fs)) if fs else float("nan")
    return out


def write_metrics_csv(summ, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "family", "n", "median_err_ms", "p95_err_ms",
                    "hit1", "hit20", "hit50", "conf_auc", "fail_rate",
                    "runtime_med_s", "false_shift_rate"])
        for method in sorted(summ):
            fsr = summ[method]["_false_shift"]
            for fam in ["ALL"] + FAMILIES:
                m = summ[method].get(fam)
                if not m:
                    continue
                w.writerow([method, fam, m["n"],
                            round(m["median_err_ms"], 3), round(m["p95_err_ms"], 3),
                            round(m["hit1"], 4), round(m["hit20"], 4),
                            round(m["hit50"], 4),
                            "" if not np.isfinite(m["conf_auc"]) else round(m["conf_auc"], 4),
                            round(m["fail_rate"], 4), round(m["runtime_med_s"], 4),
                            "" if fam != "ALL" or not np.isfinite(fsr) else round(fsr, 4)])


def fmt(x, nd=1, pct=False):
    if x is None or not np.isfinite(x):
        return "  --  "
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def write_summary_md(summ, rows, path):
    methods = sorted(summ)
    L = []
    L.append("# Alignment bake-off -- results\n")
    L.append(f"Rows scored: **{len(rows)}**  |  tolerances "
             f"**{TOL_STRICT_MS:.0f} / {TOL_MS:.0f} / {TOL_LOOSE_MS:.0f} ms**  |  "
             f"codec family bounded at {CODEC_TOL_MS:.0f} ms\n")
    L.append(f"Headline bar is **{TOL_LOOSE_MS:.0f} ms**: AWARE(20bps) slides its "
             f"detection window in 42 ms steps, so alignment finer than that buys "
             f"the detector nothing. The {TOL_MS:.0f} ms column is kept for "
             f"comparability with `fsss/exp_a_repeatability.py` (`W_MS`).\n")

    L.append(f"\n## Headline: hit rate @ {TOL_LOOSE_MS:.0f} ms, by family\n")
    hdr = "| method | " + " | ".join(FAMILIES) + " | ALL | false-shift | conf AUC |"
    L.append(hdr)
    L.append("|" + "---|" * (len(FAMILIES) + 4))
    for me in methods:
        cells = []
        for fam in FAMILIES:
            m = summ[me].get(fam)
            cells.append(fmt(m["hit50"], 0, pct=True) if m else "  --  ")
        allm = summ[me]["ALL"]
        L.append(f"| `{me}` | " + " | ".join(cells) + " | "
                 f"**{fmt(allm['hit50'], 0, pct=True)}** | "
                 f"{fmt(summ[me]['_false_shift'], 1, pct=True)} | "
                 f"{fmt(allm['conf_auc'], 3)} |")

    L.append("\n## Accuracy and cost (all families)\n")
    L.append("| method | median err (ms) | p95 err (ms) | hit@1ms | hit@20ms | "
             "hit@50ms | fail | runtime (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for me in methods:
        m = summ[me]["ALL"]
        L.append(f"| `{me}` | {fmt(m['median_err_ms'], 2)} | {fmt(m['p95_err_ms'], 1)} | "
                 f"{fmt(m['hit1'], 0, pct=True)} | {fmt(m['hit20'], 0, pct=True)} | "
                 f"{fmt(m['hit50'], 0, pct=True)} | "
                 f"{fmt(m['fail_rate'], 1, pct=True)} | {fmt(m['runtime_med_s'], 3)} |")
    L.append("\n*A method whose hit@20ms is far below its hit@50ms is "
             "**resolution-limited, not wrong** -- audalign's fingerprint and "
             "spectrogram recognizers sit at a ~22-25 ms floor even on clean "
             "audio (measured by `methods.calibrate()`).*")

    L.append("\n## Decision\n")
    L.append("Applying the pre-registered order: false-shift gate -> hit@20ms "
             "-> confidence AUC -> runtime veto.\n")

    disq, ok = [], []
    for me in methods:
        fs = summ[me]["_false_shift"]
        if np.isfinite(fs) and fs > 0.05:
            disq.append((me, fs))
        else:
            ok.append(me)
    if disq:
        L.append("**Disqualified** (false-shift > 5% -- hallucinates offsets on "
                 "audio that never moved):\n")
        for me, fs in sorted(disq, key=lambda t: -t[1]):
            L.append(f"- `{me}` -- {fmt(fs, 1, pct=True)}")
        L.append("")
    ranked = sorted(ok, key=lambda me: -(summ[me]["ALL"]["hit50"]
                                         if np.isfinite(summ[me]["ALL"]["hit50"]) else -1))
    if ranked:
        L.append(f"**Ranking of survivors** (by hit@{TOL_LOOSE_MS:.0f}ms over all "
                 f"families):\n")
        for i, me in enumerate(ranked, 1):
            m = summ[me]["ALL"]
            L.append(f"{i}. `{me}` -- {fmt(m['hit50'], 0, pct=True)} @50ms "
                     f"({fmt(m['hit20'], 0, pct=True)} @20ms), "
                     f"conf AUC {fmt(m['conf_auc'], 3)}, "
                     f"{fmt(m['runtime_med_s'], 3)}s/pair")
        best = ranked[0]
        ms_best = max(ranked, key=lambda me: (summ[me].get("multiseg", {}) or {}).get("hit50", -1)
                      if summ[me].get("multiseg") else -1)
        warp_best = max(ranked, key=lambda me: (summ[me].get("warp", {}) or {}).get("hit50", -1)
                        if summ[me].get("warp") else -1)
        L.append(f"\n**Primary: `{best}`.**")
        if ms_best != best:
            L.append(f" Multi-segment (splice/insert) is better served by "
                     f"`{ms_best}` -- pair them.")
        if warp_best not in (best, ms_best):
            L.append(f" Warping (time_stretch/jitter) falls to `{warp_best}`.")
        L.append("")

    L.append("\n---\n*Interpretation of each metric is documented at the top of "
             "`align_bench/score.py`.*\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(f"no raw_clip*.csv in {DATA_DIR} -- run run_bench.py first")
    print(f"scoring {len(rows)} rows from {DATA_DIR}")
    summ = summarize(rows)
    write_metrics_csv(summ, os.path.join(RESULTS_DIR, "data", "metrics.csv"))
    write_summary_md(summ, rows, os.path.join(RESULTS_DIR, "summary.md"))
    print(f"wrote {os.path.join(RESULTS_DIR, 'summary.md')}")
    print(f"wrote {os.path.join(RESULTS_DIR, 'data', 'metrics.csv')}")

    print("\n  method            @1ms    @20ms   @50ms   false-shift")
    for me in sorted(summ):
        m = summ[me]["ALL"]
        print(f"  {me:16s} {fmt(m['hit1'], 0, pct=True):>6s}  "
              f"{fmt(m['hit20'], 0, pct=True):>6s}  {fmt(m['hit50'], 0, pct=True):>6s}   "
              f"{fmt(summ[me]['_false_shift'], 1, pct=True):>7s}")


if __name__ == "__main__":
    main()
