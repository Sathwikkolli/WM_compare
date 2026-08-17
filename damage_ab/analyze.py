"""
damage_ab/analyze.py -- turn data/raw.csv into summary.md + figures.

THE PRE-REGISTERED QUESTION, evaluated here exactly as it was written in the run
README before the run:

    An attack is DESTRUCTIVE (as opposed to AWARE being fragile) when

        (a) src_pesq < PESQ_FLOOR                 audio with NO watermark in it
                                                  is already unusable, and
        (b) wm_pesq is not materially worse       the watermark is not what made
            than src_pesq                         it worse

    Both must hold. (a) alone would say the attack is harsh; (b) is what rules
    out "the watermark made the audio fragile". MATERIAL is fixed at 0.3 PESQ,
    declared before the run so it cannot be tuned to the answer.

WHAT THIS FILE WILL NOT DO. n=5. It reports every clip's value, the median, and
the range. It does not report p-values: a paired sign test on 5 clips bottoms out
at p=0.0625 and could not reach 0.05 even if all 5 clips agreed perfectly. The
effect this run is built to show is large and mechanical, and 5 clips either show
it plainly or they do not. Anything dressed up as significance here would be
decoration.

METRIC MEANINGS (the four PESQ columns have four different references) are
documented at the top of run_damage.py. Read that before quoting any number.

Usage:
    python analyze.py                 # -> summary.md, figures/, data/by_condition.csv
    python analyze.py --no-figures
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

from sweep import (ORDER, AWARE_BAND, HP_HZ, Q_LEVELS, group_of,  # noqa: E402
                   label_of, STRENGTH_X)

RUN = os.environ.get("DAMAGE_RUN", "2026-08-17_attack-damage-control")
RUN_DIR = os.path.join(BASE, "results", RUN)
DATA_DIR = os.path.join(RUN_DIR, "data")
FIG_DIR = os.path.join(RUN_DIR, "figures")

# --- pre-registered constants. Do not change these to fit an outcome. ---------
PESQ_FLOOR = 2.0          # below this, unwatermarked audio counts as unusable
MATERIAL = 0.3            # PESQ gap counted as a real arm difference
CONF_THRESHOLD = 0.5      # the production threshold; see results/THRESHOLD_DECISION.md


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_rows():
    paths = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR))
             if f.startswith("raw") and f.endswith(".csv")]
    if not paths:
        raise SystemExit(f"no raw*.csv in {DATA_DIR} -- run run_damage.py first.")
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def col(rows, name):
    """Numeric column with missing values dropped."""
    return np.array([v for v in (fnum(r[name]) for r in rows) if v is not None])


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


def fmt(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def by_condition(rows):
    """cond -> dict of medians + the per-clip lists the medians came from."""
    out = {}
    for cond in ORDER:
        rs = [r for r in rows if r["condition"] == cond and r["status"] == "ok"]
        if not rs:
            continue
        d = {"n": len(rs), "group": group_of(cond), "label": label_of(cond),
             "strength": STRENGTH_X.get(cond)}
        for c in ("src_pesq", "wm_pesq", "comb_pesq", "wmcost_pesq",
                  "src_snr_db", "wm_snr_db", "src_band_keep", "wm_band_keep",
                  "src_conf", "wm_conf", "wm_bit_acc"):
            v = col(rs, c)
            d[c] = med(v)
            d[c + "_all"] = v
        # Paired arm gap, computed per clip then aggregated -- never as a
        # difference of two medians, which is not the median of differences.
        gaps = []
        for r in rs:
            a, b = fnum(r["src_pesq"]), fnum(r["wm_pesq"])
            if a is not None and b is not None:
                gaps.append(a - b)
        d["arm_gap"] = med(np.array(gaps)) if gaps else float("nan")
        d["arm_gap_all"] = np.array(gaps)
        d["det_rate"] = float(np.mean(d["wm_conf_all"] >= CONF_THRESHOLD)) \
            if len(d["wm_conf_all"]) else float("nan")
        d["fp_rate"] = float(np.mean(d["src_conf_all"] >= CONF_THRESHOLD)) \
            if len(d["src_conf_all"]) else float("nan")
        out[cond] = d
    return out


def verdict(d):
    """Apply the pre-registered rule to one condition."""
    src, gap = d["src_pesq"], d["arm_gap"]
    if src != src:
        return "no data"
    destroyed = src < PESQ_FLOOR
    # gap = src_pesq - wm_pesq. Positive means the WATERMARKED arm scored worse.
    wm_worse = gap == gap and gap > MATERIAL
    if destroyed and not wm_worse:
        return "DESTRUCTIVE"
    if destroyed and wm_worse:
        return "destructive + wm worse"
    if wm_worse:
        return "wm-specific"
    return "survivable"


def baseline_ok(bc):
    """Is the watermark detected at all, with no attack?

    Every 'breaks at X' statement is conditional on this. If the clean control is
    already below threshold the sweep has no working baseline to fall from, and a
    'breaking point' would just be the first row scanned. Caught by the stub
    during development, where clean itself scored 0.33 and the sweep cheerfully
    reported that high-pass 'breaks' at 200 Hz.
    """
    d = bc.get("clean")
    return bool(d) and d["wm_conf"] >= CONF_THRESHOLD


def breaking_point(bc, group, values):
    """First strength in `values` (mildest-first) at which median wm_conf drops
    below the threshold, plus whether it ever recovers afterwards.

    Returns (strength, d, recovers) or (None, None, False). Only meaningful when
    baseline_ok(bc) -- the caller must check.
    """
    conds = [(v, f"hp_{v}hz" if group == "highpass" else f"quant_{v}lvl")
             for v in values]
    hit = None
    for v, c in conds:
        d = bc.get(c)
        if d is None:
            continue
        if hit is None and d["wm_conf"] < CONF_THRESHOLD:
            hit = (v, d)
        elif hit is not None and d["wm_conf"] >= CONF_THRESHOLD:
            # Non-monotone: it came back at a HARSHER setting. Almost always a
            # detector artefact (cf. the 440 Hz tonal false positive in the null
            # test), and worth surfacing rather than smoothing over.
            return hit[0], hit[1], True
    return (hit[0], hit[1], False) if hit else (None, None, False)


def figures(bc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}) -- skipping figures")
        return []

    os.makedirs(FIG_DIR, exist_ok=True)
    made = []
    for group, values, xlabel, invert in (
        ("highpass", HP_HZ, "high-pass cutoff (Hz)", False),
        ("quantize", Q_LEVELS, "quantisation levels", True),
    ):
        conds = [f"hp_{v}hz" if group == "highpass" else f"quant_{v}lvl" for v in values]
        conds = [c for c in conds if c in bc]
        if not conds:
            continue
        x = [bc[c]["strength"] for c in conds]

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(x, [bc[c]["src_pesq"] for c in conds], "o-", label="unwatermarked (control)")
        ax[0].plot(x, [bc[c]["wm_pesq"] for c in conds], "s--", label="watermarked")
        ax[0].axhline(PESQ_FLOOR, color="crimson", ls=":", lw=1.5,
                      label=f"pre-registered floor {PESQ_FLOOR}")
        ax[0].set_ylabel("PESQ (median over clips)")
        ax[0].set_ylim(1.0, 4.6)

        ax[1].plot(x, [bc[c]["wm_conf"] for c in conds], "s--", color="tab:orange",
                   label="watermarked")
        ax[1].plot(x, [bc[c]["src_conf"] for c in conds], "o-", color="tab:blue",
                   label="unwatermarked (control)")
        ax[1].axhline(CONF_THRESHOLD, color="crimson", ls=":", lw=1.5,
                      label=f"threshold {CONF_THRESHOLD}")
        ax[1].set_ylabel("AWARE confidence (median)")
        ax[1].set_ylim(-0.02, 1.02)

        for a in ax:
            a.set_xlabel(xlabel)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
            if invert:
                a.set_xscale("log", base=2)
                a.invert_xaxis()            # harsher to the right, as in the tables
        fig.suptitle(f"{group}: quality damage vs. detection, matched arms "
                     f"(n={bc[conds[0]]['n']} clips)")
        fig.tight_layout()
        p = os.path.join(FIG_DIR, f"sweep_{group}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
        print(f"wrote {p}")
    return made


def main(argv):
    rows = load_rows()
    ok = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"] != "ok"]
    bc = by_condition(rows)
    clips = sorted({r["clip_id"] for r in rows})

    L = []
    A = L.append
    A("# Summary — does the attack destroy the audio, or is AWARE fragile?\n")
    A(f"{len(ok)} scored rows over {len(clips)} clips × {len(ORDER)} conditions"
      + (f"; {len(skipped)} skipped/errored" if skipped else "") + ".\n")
    A("Matched arms: every clip appears both unwatermarked (`src`) and "
      "AWARE-embedded (`wm`), so each row's arm comparison is within-clip.\n")

    # --- the pre-registered test --------------------------------------------
    A("## The pre-registered test\n")
    A(f"An attack is **destructive** when unwatermarked audio falls below "
      f"PESQ {PESQ_FLOOR} under it (`src_pesq`), and the watermarked arm is not "
      f"worse by more than {MATERIAL} PESQ. Both criteria and both constants were "
      f"fixed in the run README before the run.\n")

    A("| condition | src PESQ (no wm) | wm PESQ | arm gap | wm conf | detected @0.5 | verdict |")
    A("|---|---|---|---|---|---|---|")
    for cond in ORDER:
        d = bc.get(cond)
        if not d:
            continue
        A(f"| `{cond}` | {fmt(d['src_pesq'])} | {fmt(d['wm_pesq'])} | "
          f"{fmt(d['arm_gap'])} | {fmt(d['wm_conf'], 3)} | "
          f"{int(round(d['det_rate'] * d['n']))}/{d['n']} | {verdict(d)} |")
    A("")
    A("`arm gap` = median per-clip (`src_pesq` − `wm_pesq`). Positive means the "
      "watermarked file was hurt more. Computed per clip and then aggregated, "
      "not as a difference of medians.\n")

    # --- the two A/B anchor cells -------------------------------------------
    A("## The two cells the A/B reported as failures\n")
    for cond, ab in (("hp_3200hz", "highpass_0.2 — 0/20 in the A/B"),
                     ("quant_8lvl", "quantize_8lvl — 1/20 in the A/B")):
        d = bc.get(cond)
        if not d:
            continue
        A(f"**`{cond}`** ({ab})\n")
        A(f"- unwatermarked audio under this attack: PESQ **{fmt(d['src_pesq'])}** "
          f"(clean reference is the `clean` row's {fmt(bc['clean']['wmcost_pesq'])})")
        A(f"- watermarked audio under this attack: PESQ **{fmt(d['wm_pesq'])}**, "
          f"arm gap {fmt(d['arm_gap'])}")
        A(f"- AWARE-band ({AWARE_BAND[0]:.0f}–{AWARE_BAND[1]:.0f} Hz) energy surviving: "
          f"**{fmt(100 * d['wm_band_keep'], 1)}%** of the watermarked signal's, "
          f"{fmt(100 * d['src_band_keep'], 1)}% of the unwatermarked signal's")
        A(f"- detection: {int(round(d['det_rate'] * d['n']))}/{d['n']} at conf ≥ "
          f"{CONF_THRESHOLD}, median conf {fmt(d['wm_conf'], 3)}, "
          f"median bit accuracy {fmt(d['wm_bit_acc'], 3)}")
        A(f"- **verdict: {verdict(d)}**\n")

    # --- where each sweep breaks --------------------------------------------
    A("## Where each attack breaks\n")
    if not baseline_ok(bc):
        cd = bc.get("clean")
        A(f"**NO BREAKING POINT CAN BE REPORTED.** The `clean` control itself sits "
          f"at median confidence {fmt(cd['wm_conf'], 3) if cd else '--'}, below the "
          f"{CONF_THRESHOLD} threshold — the watermark is not detected before any "
          f"attack, so there is nothing for the sweep to break. Every number in "
          f"this file below the quality columns is uninterpretable. Check "
          f"`embed_pairs.py`'s round-trip output before reading further.\n")
    else:
        for group, values, unit in (("highpass", HP_HZ, "Hz"),
                                    ("quantize", Q_LEVELS, "levels")):
            v, d, recovers = breaking_point(bc, group, values)
            if v is None:
                A(f"- **{group}**: median confidence never fell below "
                  f"{CONF_THRESHOLD} anywhere in the sweep.")
            else:
                A(f"- **{group}**: median confidence first drops below "
                  f"{CONF_THRESHOLD} at **{v} {unit}** (conf {fmt(d['wm_conf'], 3)}, "
                  f"unwatermarked PESQ there {fmt(d['src_pesq'])}, band kept "
                  f"{fmt(100 * d['wm_band_keep'], 1)}%)."
                  + ("  **Non-monotone** — detection returns at a harsher setting "
                     "further down the sweep. Treat that as a detector artefact to "
                     "investigate, not as robustness." if recovers else ""))
        A("")
        cd = bc["clean"]
        A(f"Baseline for all of the above: the `clean` control detects "
          f"{int(round(cd['det_rate'] * cd['n']))}/{cd['n']} at conf ≥ "
          f"{CONF_THRESHOLD} (median {fmt(cd['wm_conf'], 3)}).")
    A("")

    # --- full per-condition table -------------------------------------------
    A("## Every condition\n")
    A("| condition | group | src PESQ | wm PESQ | comb PESQ | src SNR | wm SNR | "
      "src band | wm band | src conf | wm conf | bit acc |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in ORDER:
        d = bc.get(cond)
        if not d:
            continue
        A(f"| `{cond}` | {d['group']} | {fmt(d['src_pesq'])} | {fmt(d['wm_pesq'])} | "
          f"{fmt(d['comb_pesq'])} | {fmt(d['src_snr_db'], 1)} | {fmt(d['wm_snr_db'], 1)} | "
          f"{fmt(100 * d['src_band_keep'], 1)}% | {fmt(100 * d['wm_band_keep'], 1)}% | "
          f"{fmt(d['src_conf'], 3)} | {fmt(d['wm_conf'], 3)} | {fmt(d['wm_bit_acc'], 3)} |")
    A("")

    # --- per-clip spread, because n=5 ---------------------------------------
    A("## Per-clip spread on the two failing cells\n")
    A(f"With n={len(clips)} the median alone can hide disagreement. These are the "
      f"individual clips.\n")
    A("| condition | metric | " + " | ".join(clips) + " |")
    A("|---|---|" + "---|" * len(clips))
    for cond in ("hp_3200hz", "quant_8lvl"):
        rs = {r["clip_id"]: r for r in rows if r["condition"] == cond}
        for metric, nd in (("src_pesq", 2), ("wm_pesq", 2), ("wm_conf", 3)):
            vals = [fmt(fnum(rs[c][metric]), nd) if c in rs else "--" for c in clips]
            A(f"| `{cond}` | {metric} | " + " | ".join(vals) + " |")
    A("")

    A("## Limits\n")
    # Smallest two-sided sign-test p reachable at this n, computed rather than
    # quoted: 2 * 0.5**n. At n=5 that is 0.0625; the number moves with --n and
    # the sentence must move with it.
    min_p = 2.0 * 0.5 ** len(clips)
    A(f"1. **n={len(clips)}.** No p-values are reported: the smallest two-sided "
      f"sign-test p reachable with {len(clips)} clips is {min_p:.4g}"
      + (", which cannot reach 0.05 even if every clip agrees."
         if min_p > 0.05 else
         ", so significance here would rest entirely on n and not on the effect.")
      + " Read the per-clip table above, not an interval.")
    A("2. **This run has no false-positive rate.** The `src` arm is the paired "
      "quality control, not an independent negative sample. FPR lives in "
      "`results/2026-08-14_detector-null-test/` (n=300).")
    A("3. **AWARE solo at 16 kHz**, as in the A/B. Production embeds through the "
      "full cascade at 22050 Hz with two resampling round-trips.")
    A("4. **Speech only**, ~10 s Emilia clips.")
    A("5. **PESQ is a speech-quality model.** At the bottom of its range it "
      "saturates near 1.0, so 'PESQ 1.04 vs 1.31' is not a meaningful ordering — "
      "both mean destroyed. Listen to `work/listen/` before quoting a difference "
      "down there.")

    os.makedirs(RUN_DIR, exist_ok=True)
    out = os.path.join(RUN_DIR, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}")

    # Tidy per-condition CSV for plotting elsewhere.
    os.makedirs(DATA_DIR, exist_ok=True)
    keys = ["condition", "group", "strength", "n", "src_pesq", "wm_pesq", "comb_pesq",
            "arm_gap", "src_snr_db", "wm_snr_db", "src_band_keep", "wm_band_keep",
            "src_conf", "wm_conf", "wm_bit_acc", "det_rate", "fp_rate"]
    p = os.path.join(DATA_DIR, "by_condition.csv")

    def cell(cond, d, k):
        if k == "condition":
            return cond
        v = d.get(k)
        if isinstance(v, float):
            return "" if v != v else round(v, 6)     # NaN -> blank, never "nan"
        return "" if v is None else v

    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["verdict"])
        for cond in ORDER:
            d = bc.get(cond)
            if not d:
                continue
            w.writerow([cell(cond, d, k) for k in keys] + [verdict(d)])
    print(f"wrote {p}")

    if "--no-figures" not in argv:
        figures(bc)

    print("\n" + "\n".join(L[:2]))


if __name__ == "__main__":
    main(sys.argv[1:])
