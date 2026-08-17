"""
frame_align/score_frames.py -- turn raw frame rows into the threshold.

Reads results/<run>/data/raw_clip*.csv, writes metrics.csv and summary.md.

WHAT EACH METRIC TELLS YOU  (read this part slowly)

  hit rate @ 50 ms, by frame length   <- THE KNEE
      Experiment A only. Fraction of frames aligned closely enough to be useful:
      AWARE(20bps) hops its detector in 42 ms, so finer buys nothing. Plotted
      against frame length this gives the shortest frame that still aligns --
      which is a floor on how short a key-hop dwell can be before sync, not the
      watermark, is the limiting factor.

  hit rate @ 1 ms / 20 ms
      1 ms is the strict bar. 20 ms is kept for comparability with the bake-off
      and with W_MS in fsss/exp_a_repeatability.py.

  false-accept rate (FA)              <- EXPERIMENT B
      Frames searched in a reference that does NOT contain them. There is no
      right answer, so every accept is a false alarm. FA at a given threshold is
      the probability of confidently aligning unrelated audio.

  useful-accept rate at FA <= 1%      <- THE HEADLINE, and the number to ship
      Set the threshold where B's false-accept rate hits 1%. At that same
      threshold, how many genuine frames both clear it AND land within 50 ms?
      Accepting with a wrong offset is not a success, so correctness is required,
      not just confidence. This single number is what the detector can rely on.

  false alarms WITHOUT calibration      <- what experiment B is worth
      The 1% above is true by construction, so it cannot tell you how bad the
      problem is. Three thresholds nobody needed a null to pick:
        accept-all  -- how often a mismatched pair still returns an offset.
                       Neither method has a "not present" answer, so this is
                       usually ~100% and is the baseline confidence must fix.
        conf>=0.5   -- the naive midpoint of the squashed [0,1] score, and the
                       same 0.5 the detectors defaulted to in the null test.
        keep-95%    -- tau set on matched data ALONE to retain 95% of genuine
                       frames. Its false-accept rate is the number this whole
                       experiment exists to reveal: what a matched-data-only
                       calibration would have shipped.

  separation AUC
      Does the score tell "frame is in here" from "frame is not in here" at all,
      ignoring whether the offset is right? Low AUC with a decent hit rate means
      the method aligns well but cannot report when it has failed -- so it cannot
      be gated, only trusted blindly.

  score comparability across reference length
      gcc_phat's confidence is a peak-to-sidelobe ratio, and the sidelobe
      population changes completely between a 10 s and a 180 s reference. If the
      null score distribution shifts with reference length, ONE GLOBAL THRESHOLD
      IS INVALID and it has to be set per condition. The run README flags this as
      the main threat to validity; this is where it gets checked.

  energy stratum
      Emilia is speech with pauses. A random 50 ms frame can land in one, where
      alignment is impossible and the method is not at fault. Silent frames are
      reported separately, never dropped and never pooled.

Usage:
    python score_frames.py
    python score_frames.py --fa 0.01
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

RUN_SLUG = "2026-08-18_frame-align-null"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")

TOL_MS = (1.0, 20.0, 50.0)
HEADLINE_TOL_MS = 50.0
TARGET_FA = 0.01

# dBFS cut points. Emilia pauses sit well below -60; conversational speech well
# above -40. The band between is genuinely ambiguous, so it gets its own row
# instead of being forced into one side.
SILENT_DBFS = -60.0
LOW_DBFS = -40.0


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def stratum(db):
    if not np.isfinite(db) or db < SILENT_DBFS:
        return "silent"
    if db < LOW_DBFS:
        return "low"
    return "voiced"


def load_rows():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "raw_clip*.csv")))
    if not files:
        raise SystemExit(
            f"no raw_clip*.csv in {DATA_DIR}\nRun run_frames.py first "
            f"(or set WM_COMPARE_BASE if the results live elsewhere).")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                r["frame_len_ms"] = int(r["frame_len_ms"])
                r["error_ms"] = fnum(r["error_ms"])
                r["confidence"] = fnum(r["confidence"])
                r["raw_score"] = fnum(r["raw_score"])
                r["frame_dbfs"] = fnum(r["frame_dbfs"])
                r["ok"] = r["ok"] == "1"
                r["touches_xfade"] = r["touches_xfade"] == "1"
                r["stratum"] = stratum(r["frame_dbfs"])
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} file(s)")
    return rows


def hit(rows, tol):
    """Fraction within tol ms. A crashed or non-finite row counts as a MISS.

    Scoring failures as missing data would let a method improve its numbers by
    failing more often.
    """
    if not rows:
        return float("nan")
    good = [r for r in rows if r["ok"] and np.isfinite(r["error_ms"])]
    return sum(abs(r["error_ms"]) <= tol for r in good) / float(len(rows))


def pct(vals, q):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


def auc(pos, neg):
    """Rank AUC with ties at 0.5. nan if either side is empty."""
    pos = [x for x in pos if np.isfinite(x)]
    neg = [x for x in neg if np.isfinite(x)]
    if not pos or not neg:
        return float("nan")
    allv = np.array(pos + neg, dtype=float)
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def threshold_at_fa(null_scores, target_fa):
    """Lowest score admitting <= target_fa of the null. inf if none does."""
    s = np.sort(np.array([x for x in null_scores if np.isfinite(x)], dtype=float))
    if not len(s):
        return float("nan")
    k = int(np.ceil((1.0 - target_fa) * len(s)))
    if k >= len(s):
        return float(s[-1]) + 1e-9      # even the max cannot be admitted
    return float(s[k])


def operating_point(a_rows, b_rows, target_fa, tol_ms, key="raw_score"):
    """(tau, FA achieved, useful-accept rate) -- the headline computation.

    A genuine frame counts only if it BOTH clears tau AND lands within tol_ms.
    Confident and wrong is a failure, not a success.
    """
    null = [r[key] for r in b_rows]
    tau = threshold_at_fa(null, target_fa)
    if not np.isfinite(tau) or not a_rows:
        return tau, float("nan"), float("nan")
    n_null = sum(1 for x in null if np.isfinite(x))
    fa = (sum(1 for x in null if np.isfinite(x) and x >= tau) / float(n_null)
          if n_null else float("nan"))
    useful = sum(1 for r in a_rows
                 if r["ok"] and np.isfinite(r[key]) and r[key] >= tau
                 and np.isfinite(r["error_ms"]) and abs(r["error_ms"]) <= tol_ms)
    return tau, fa, useful / float(len(a_rows))


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def main(argv):
    target_fa = get_arg(argv, "--fa", TARGET_FA, float)
    rows = load_rows()

    A = [r for r in rows if r["experiment"] == "A"]
    B = [r for r in rows if r["experiment"] == "B"]
    methods = sorted({r["method"] for r in rows})
    lens = sorted({r["frame_len_ms"] for r in rows})
    kinds = sorted({r["ref_kind"] for r in rows})
    print(f"  A(matched)={len(A)}  B(null)={len(B)}  "
          f"methods={methods}  lens={lens}  ref_kinds={kinds}")
    if not B:
        print("  ! no experiment B rows -- the null test did not run, so no "
              "threshold can be derived")

    out, L = [], []
    w = L.append

    w("# Frame-level alignment: resolution limit and false-alarm floor\n")
    w(f"Run `{RUN_SLUG}`. {len(rows)} rows "
      f"({len(A)} matched, {len(B)} null), methods {', '.join(methods)}, "
      f"reference kinds {', '.join(kinds)}.\n")
    w("Metric definitions are at the top of `score_frames.py`. Read them before "
      "quoting any number here.\n")

    # ---- 1. the knee -------------------------------------------------------
    w("\n## 1. The knee: hit rate @ 50 ms vs frame length\n")
    w("Experiment A, voiced frames only (silent frames are section 3).\n")
    for kind in kinds:
        w(f"\n**reference = {kind}**\n")
        w("| frame | " + " | ".join(f"{m} @50ms | {m} @20ms | {m} @1ms"
                                    for m in methods) + " | n |")
        w("|---" * (1 + 3 * len(methods)) + "|---|")
        for L_ms in lens:
            cells, n_show = [], 0
            for m in methods:
                sel = [r for r in A if r["method"] == m and r["frame_len_ms"] == L_ms
                       and r["ref_kind"] == kind and r["stratum"] == "voiced"]
                n_show = max(n_show, len(sel))
                cells += [f"{100*hit(sel, t):.1f}%" if sel else "-"
                          for t in sorted(TOL_MS, reverse=True)]
                for t in TOL_MS:
                    out.append({"section": "hit_rate", "method": m,
                                "ref_kind": kind, "frame_len_ms": L_ms,
                                "stratum": "voiced", "tol_ms": t,
                                "hit_rate": hit(sel, t), "n_a": len(sel)})
            w(f"| {L_ms} ms | " + " | ".join(cells) + f" | {n_show} |")
    w("\n`@1ms` is the strict bar -- it only matters if something downstream needs "
      "sample-grade sync. A large gap between `@1ms` and `@20ms` means the method "
      "is resolution-limited, not wrong.\n")

    # ---- 2. the headline ---------------------------------------------------
    w(f"\n## 2. Useful-accept rate at FA <= {100*target_fa:.0f}%  <- HEADLINE\n")
    w("Threshold set on the null (experiment B) so false accepts sit at or below "
      f"{100*target_fa:.0f}%. At that same threshold, the fraction of genuine "
      f"frames that clear it *and* land within {HEADLINE_TOL_MS:.0f} ms.\n")
    w("Scored on the method's raw statistic (`raw_score`): PSR for `gcc_phat`, "
      "`standard_score` for `aof`.\n")
    for kind in kinds:
        w(f"\n**reference = {kind}**\n")
        w("| frame | " + " | ".join(f"{m} tau | {m} useful" for m in methods) + " |")
        w("|---" * (1 + 2 * len(methods)) + "|")
        for L_ms in lens:
            cells = []
            for m in methods:
                a = [r for r in A if r["method"] == m and r["frame_len_ms"] == L_ms
                     and r["ref_kind"] == kind and r["stratum"] == "voiced"]
                b = [r for r in B if r["method"] == m and r["frame_len_ms"] == L_ms
                     and r["ref_kind"] == kind and r["stratum"] == "voiced"]
                tau, fa, useful = operating_point(a, b, target_fa, HEADLINE_TOL_MS)
                cells += [f"{tau:.2f}" if np.isfinite(tau) else "-",
                          f"{100*useful:.1f}%" if np.isfinite(useful) else "-"]
                out.append({"section": "operating_point", "method": m,
                            "ref_kind": kind, "frame_len_ms": L_ms,
                            "stratum": "voiced", "tau": tau, "fa": fa,
                            "useful_accept": useful, "n_a": len(a), "n_b": len(b)})
            w(f"| {L_ms} ms | " + " | ".join(cells) + " |")

    # ---- 3. energy stratum -------------------------------------------------
    w("\n## 3. Energy stratum: is short-frame failure the method or the silence?\n")
    w(f"`silent` < {SILENT_DBFS:.0f} dBFS, `low` < {LOW_DBFS:.0f}, "
      f"`voiced` above. Hit @ 50 ms, experiment A, pooled over reference kinds.\n")
    w("| frame | stratum | n | " + " | ".join(methods) + " |")
    w("|---|---|---|" + "---|" * len(methods))
    for L_ms in lens:
        for st in ("voiced", "low", "silent"):
            sel_any = [r for r in A if r["frame_len_ms"] == L_ms and r["stratum"] == st]
            if not sel_any:
                continue
            cells = []
            for m in methods:
                sel = [r for r in sel_any if r["method"] == m]
                cells.append(f"{100*hit(sel, 50.0):.1f}%" if sel else "-")
            n = len({(r["clip_idx"], r["trial"], r["ref_kind"]) for r in sel_any})
            w(f"| {L_ms} ms | {st} | {n} | " + " | ".join(cells) + " |")

    # ---- 4. separation -----------------------------------------------------
    w("\n## 4. Separation AUC: can the score tell matched from unrelated?\n")
    w("Ignores whether the offset is right -- purely 'is this frame in here'. "
      "A method with a good hit rate but AUC near 0.5 cannot be gated.\n")
    w("| frame | " + " | ".join(methods) + " |")
    w("|---|" + "---|" * len(methods))
    for L_ms in lens:
        cells = []
        for m in methods:
            a = [r["raw_score"] for r in A if r["method"] == m
                 and r["frame_len_ms"] == L_ms and r["stratum"] == "voiced"]
            b = [r["raw_score"] for r in B if r["method"] == m
                 and r["frame_len_ms"] == L_ms and r["stratum"] == "voiced"]
            v = auc(a, b)
            cells.append(f"{v:.3f}" if np.isfinite(v) else "-")
            out.append({"section": "auc", "method": m, "ref_kind": "ALL",
                        "frame_len_ms": L_ms, "stratum": "voiced", "auc": v,
                        "n_a": len(a), "n_b": len(b)})
        w(f"| {L_ms} ms | " + " | ".join(cells) + " |")

    # ---- 5. what the null bought -------------------------------------------
    w("\n## 5. What experiment B is worth: false alarms without calibration\n")
    w("Section 2's 1% holds **by construction** -- tau was chosen to make it so, so "
      "it says nothing about how bad the problem is. This is the counterfactual: "
      "how often does unrelated audio get accepted at a threshold you would have "
      "picked WITHOUT running a null?\n")
    w("- **accept-all** -- fraction of mismatched pairs that still come back with "
      "an offset. Neither method has a 'not present' answer, so this is the "
      "baseline the confidence score has to fix.")
    w("- **conf>=0.5** -- the naive midpoint of the squashed [0,1] confidence, and "
      "the same 0.5 the detectors defaulted to in "
      "`2026-08-14_detector-null-test`.")
    w("- **keep-95%** -- tau set from experiment A *alone*, to retain 95% of "
      "genuine frames. This is what you would do with no null data; the FA beside "
      "it is the price of that choice.\n")
    w("| frame | method | accept-all | conf>=0.5 | keep-95% tau | its FA |")
    w("|---|---|---|---|---|---|")
    for L_ms in lens:
        for m in methods:
            a = [r for r in A if r["method"] == m and r["frame_len_ms"] == L_ms
                 and r["stratum"] == "voiced"]
            b = [r for r in B if r["method"] == m and r["frame_len_ms"] == L_ms
                 and r["stratum"] == "voiced"]
            if not b:
                continue
            fa_all = sum(1 for r in b
                         if r["ok"] and np.isfinite(r["raw_score"])) / float(len(b))

            n_conf = sum(1 for r in b if np.isfinite(r["confidence"]))
            fa_half = (sum(1 for r in b if np.isfinite(r["confidence"])
                           and r["confidence"] >= 0.5) / float(n_conf)
                       if n_conf else float("nan"))

            # 5th percentile of the matched scores = the bar that keeps 95% of
            # genuine frames. A calibrator with no null data has nothing else.
            tau95 = pct([r["raw_score"] for r in a if r["ok"]], 5)
            n_b_ok = sum(1 for r in b if np.isfinite(r["raw_score"]))
            fa95 = (sum(1 for r in b if np.isfinite(r["raw_score"])
                        and r["raw_score"] >= tau95) / float(n_b_ok)
                    if n_b_ok and np.isfinite(tau95) else float("nan"))

            w(f"| {L_ms} ms | {m} | {100*fa_all:.1f}% | "
              + (f"{100*fa_half:.1f}%" if np.isfinite(fa_half) else "-") + " | "
              + (f"{tau95:.2f}" if np.isfinite(tau95) else "-") + " | "
              + (f"**{100*fa95:.1f}%**" if np.isfinite(fa95) else "-") + " |")
            out.append({"section": "naive_fa", "method": m, "ref_kind": "ALL",
                        "frame_len_ms": L_ms, "stratum": "voiced",
                        "fa_accept_all": fa_all, "fa_conf_half": fa_half,
                        "tau_keep95": tau95, "fa_keep95": fa95,
                        "n_a": len(a), "n_b": len(b)})
    w("\nThe last column is the cost of skipping this experiment: the false-accept "
      "rate a matched-data-only calibration would have shipped.\n")

    # ---- 6. comparability --------------------------------------------------
    w("\n## 6. Is one global threshold valid across reference lengths?\n")
    w("Null-score percentiles per reference kind. If p99 moves materially with "
      "reference length, a single threshold is INVALID and tau must be set per "
      "condition. This is the threat to validity named in the run README.\n")
    w("| method | ref | n | median | p95 | p99 | max |")
    w("|---|---|---|---|---|---|---|")
    for m in methods:
        p99s = []
        for kind in kinds:
            v = [r["raw_score"] for r in B if r["method"] == m and r["ref_kind"] == kind]
            v = [x for x in v if np.isfinite(x)]
            if not v:
                continue
            p99 = pct(v, 99)
            p99s.append(p99)
            w(f"| {m} | {kind} | {len(v)} | {pct(v,50):.2f} | {pct(v,95):.2f} "
              f"| {p99:.2f} | {max(v):.2f} |")
            out.append({"section": "null_dist", "method": m, "ref_kind": kind,
                        "frame_len_ms": "ALL", "stratum": "ALL",
                        "null_p50": pct(v, 50), "null_p95": pct(v, 95),
                        "null_p99": p99, "n_b": len(v)})
        if len(p99s) > 1:
            spread = max(p99s) / max(min(p99s), 1e-9)
            verdict = ("one global threshold is defensible" if spread < 1.25 else
                       "THRESHOLD MUST BE SET PER REFERENCE LENGTH")
            w(f"| **{m}** | *p99 spread* | | | | **{spread:.2f}x** | {verdict} |")

    # ---- 7. cost -----------------------------------------------------------
    w("\n## 7. Runtime\n")
    w("| method | ref | median s/call | p95 |")
    w("|---|---|---|---|")
    for m in methods:
        for kind in kinds:
            v = [fnum(r["runtime_s"]) for r in rows
                 if r["method"] == m and r["ref_kind"] == kind]
            v = [x for x in v if np.isfinite(x)]
            if v:
                w(f"| {m} | {kind} | {pct(v,50):.3f} | {pct(v,95):.3f} |")

    # ---- integrity ---------------------------------------------------------
    w("\n## Data integrity\n")
    n_fail = sum(1 for r in rows if not r["ok"])
    n_xf = sum(1 for r in A if r["touches_xfade"])
    w(f"- rows where the method failed: **{n_fail}** "
      f"({100.0*n_fail/max(1,len(rows)):.2f}%) -- scored as misses, not dropped")
    w(f"- matched frames overlapping a crossfade region: **{n_xf}** "
      f"({100.0*n_xf/max(1,len(A)):.2f}%) -- see `refs.py`; exclude and re-score "
      f"if this is large")
    miss_raw = sum(1 for r in rows if not np.isfinite(r["raw_score"]))
    w(f"- rows with no parsable raw score: **{miss_raw}** "
      f"({100.0*miss_raw/max(1,len(rows)):.2f}%) -- if this is not ~0 the note "
      f"format in `methods.py` changed and section 2 is unreliable")

    w("\n## Conclusion\n")
    w("*(write this by hand after reading the tables -- results/README.md rule 2)*")

    os.makedirs(DATA_DIR, exist_ok=True)
    if out:
        cols = sorted({k for d in out for k in d})
        with open(os.path.join(DATA_DIR, "metrics.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(out)

    txt = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {os.path.join(RESULTS_DIR, 'summary.md')}")
    print(f"wrote {os.path.join(DATA_DIR, 'metrics.csv')}")


if __name__ == "__main__":
    main(sys.argv[1:])
