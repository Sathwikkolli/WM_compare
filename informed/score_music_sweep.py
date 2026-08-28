"""
informed/score_music_sweep.py -- does a usable-but-undetectable band exist?

Reads results/<run>/data/music_clip*.csv, writes summary.md and
data/music_metrics.csv.

THE CENTRAL COMPUTATION

Lowering SNR adds music: both detection confidence and quality fall. Two
crossings matter, per clip:

    snr_det   the SNR where confidence falls through 0.50 -- the watermark dies
    snr_qual  the SNR where DNSMOS falls through 3.0      -- the audio dies

    window = snr_det - snr_qual

  window > 0   detection dies FIRST. Between snr_qual and snr_det the audio is
               still usable and the watermark is undetectable. THAT BAND IS THE
               VULNERABILITY, and its width is how much room an attacker has.
  window <= 0  quality dies first: the watermark outlives usability, exactly as
               2026-08-17_attack-damage-control found for high-pass and
               quantisation. No vulnerability, and the parent plan loses its
               lead candidate.

Everything else here tests one of the five registered predictions in
results/2026-08-28_informed-detection/PHASE_A_PLAN.md.

Usage:
    python score_music_sweep.py
    python score_music_sweep.py --floor 3.5      # the strict usability floor
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

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")

DETECT_THRESHOLD = 0.50
DNSMOS_FLOOR = 3.0
PESQ_FLOOR = 2.0          # the conventional "still acceptable" PESQ line


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_rows():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "music_clip*.csv")))
    if not files:
        raise SystemExit(f"no music_clip*.csv in {DATA_DIR} -- run music_sweep.py")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("snr_db", "conf", "bit_acc", "dnsmos_ovrl", "dnsmos_sig",
                          "dnsmos_bak", "pesq", "stoi", "si_snr_db", "runtime_s"):
                    r[k] = fnum(r.get(k))
                r["detected"] = r.get("detected") == "1"
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} file(s)")
    return rows


def crossing(points, threshold):
    """SNR at which `value` falls through `threshold` as SNR decreases.

    `points` is [(snr, value), ...]. Linearly interpolated between the two
    bracketing measurements. Returns nan if it never crosses inside the swept
    range -- reported as nan rather than clamped to an endpoint, because
    "never crossed" and "crossed at the edge" are different facts.
    """
    pts = sorted(((s, v) for s, v in points if np.isfinite(s) and np.isfinite(v)),
                 key=lambda p: -p[0])                    # high SNR -> low
    for (s1, v1), (s2, v2) in zip(pts, pts[1:]):
        if v1 >= threshold > v2:
            if v1 == v2:
                return float(s2)
            return float(s2 + (threshold - v2) * (s1 - s2) / (v1 - v2))
    return float("nan")


def value_at(points, snr):
    """Interpolate a swept quantity at an arbitrary SNR."""
    pts = sorted(((s, v) for s, v in points if np.isfinite(s) and np.isfinite(v)),
                 key=lambda p: p[0])
    if not pts or not np.isfinite(snr):
        return float("nan")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if snr <= xs[0] or snr >= xs[-1]:
        return float(ys[0] if snr <= xs[0] else ys[-1])
    return float(np.interp(snr, xs, ys))


def stat(vals):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "min": float("nan"), "max": float("nan"), "median": float("nan")}
    a = np.array(v, dtype=float)
    return {"n": len(v), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "median": float(np.median(a))}


def fmt(v, nd=2, dash="-"):
    return f"{v:.{nd}f}" if np.isfinite(v) else dash


def main(argv):
    floor = get_arg(argv, "--floor", DNSMOS_FLOOR, float)
    rows = load_rows()

    musics = sorted({r["music"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    backends = sorted({r.get("nr_backend") or "?" for r in rows})
    print(f"  clips={len(clips)} music={musics} backends={backends}")

    # ---- per (clip, music) crossings --------------------------------------
    per = []
    for music in musics:
        for cid in clips:
            wm = [r for r in rows if r["music"] == music and r["clip_id"] == cid
                  and r["arm"] == "wm"]
            un = [r for r in rows if r["music"] == music and r["clip_id"] == cid
                  and r["arm"] == "unwm"]
            if not wm:
                continue
            conf_pts = [(r["snr_db"], r["conf"]) for r in wm]
            dns_pts = [(r["snr_db"], r["dnsmos_ovrl"]) for r in wm]
            pesq_pts = [(r["snr_db"], r["pesq"]) for r in wm]
            bacc_pts = [(r["snr_db"], r["bit_acc"]) for r in wm]

            s_det = crossing(conf_pts, DETECT_THRESHOLD)
            s_qual = crossing(dns_pts, floor)
            s_pesq = crossing(pesq_pts, PESQ_FLOOR)

            per.append({
                "music": music, "clip_id": cid,
                "snr_detection_dies": s_det,
                "snr_quality_dies": s_qual,
                "snr_pesq_dies": s_pesq,
                "window_db": s_det - s_qual if np.isfinite(s_det) and np.isfinite(s_qual)
                             else float("nan"),
                "bit_acc_at_detection_death": value_at(bacc_pts, s_det),
                "dnsmos_at_detection_death": value_at(dns_pts, s_det),
                "pesq_at_detection_death": value_at(pesq_pts, s_det),
                "arm_gap_dnsmos": float(np.nanmean(
                    [a["dnsmos_ovrl"] - b["dnsmos_ovrl"]
                     for a in un for b in wm if a["snr_db"] == b["snr_db"]]
                )) if un else float("nan"),
            })

    L = []
    w = L.append
    w("# Phase A-1 — the music sweep\n")
    w(f"Run `{RUN_SLUG}`. {len(rows)} rows, {len(clips)} clips, music "
      f"{', '.join(musics)}. Quality backend: {', '.join(backends)}.\n")
    w(f"Detection threshold **{DETECT_THRESHOLD}**, usability floor "
      f"**DNSMOS {floor}**.\n")
    w("Metric definitions and the crossing computation are documented at the "
      "top of `score_music_sweep.py`. Read them before quoting anything here.\n")

    # ---- 1. the headline ---------------------------------------------------
    w("\n## 1. Does a usable-but-undetectable band exist?  <- PREDICTION 1\n")
    w("`window = snr_detection_dies − snr_quality_dies`. Positive means "
      "detection dies while the audio is still usable, and the window is how "
      "much room the attacker has.\n")
    w("| music | n | detection dies (dB) | quality dies (dB) | **window (dB)** | clips with window>0 |")
    w("|---|---|---|---|---|---|")
    for music in musics:
        sel = [p for p in per if p["music"] == music]
        d, q, win = (stat([p["snr_detection_dies"] for p in sel]),
                     stat([p["snr_quality_dies"] for p in sel]),
                     stat([p["window_db"] for p in sel]))
        pos = sum(1 for p in sel if np.isfinite(p["window_db"]) and p["window_db"] > 0)
        n_win = sum(1 for p in sel if np.isfinite(p["window_db"]))
        w(f"| {music} | {len(sel)} | {fmt(d['mean'])} ± {fmt(d['sd'])} | "
          f"{fmt(q['mean'])} ± {fmt(q['sd'])} | **{fmt(win['mean'])} ± {fmt(win['sd'])}** | "
          f"{pos}/{n_win} |")
    w("\nA window that is positive for most clips **confirms prediction 1** and "
      "the music bed is a genuine vulnerability. Negative or ~zero **refutes "
      "it**: the watermark outlives usability here as it does for high-pass and "
      "quantisation, and the parent plan loses its lead candidate.")
    w("\n`quality dies = nan` means DNSMOS never fell through the floor anywhere "
      "in the swept range — the audio stayed usable at every SNR tested, which "
      "is the strongest possible form of prediction 1 holding.")

    # ---- 2. is the watermark still there? ---------------------------------
    w("\n## 2. Is the watermark still present when detection fails?  <- PREDICTION 5\n")
    w("Bit accuracy at the SNR where confidence crosses 0.50. High values mean "
      "the mark is readable but the detector cannot clear threshold — precisely "
      "the condition informed detection should fix. If the bits are already "
      "gone, **informed detection has nothing to recover** and the parent plan "
      "needs rethinking before any code is written for it.\n")
    w("| music | bit accuracy at detection death | min | DNSMOS there | PESQ there |")
    w("|---|---|---|---|---|")
    for music in musics:
        sel = [p for p in per if p["music"] == music]
        b = stat([p["bit_acc_at_detection_death"] for p in sel])
        dq = stat([p["dnsmos_at_detection_death"] for p in sel])
        pq = stat([p["pesq_at_detection_death"] for p in sel])
        w(f"| {music} | {fmt(b['mean'], 3)} ± {fmt(b['sd'], 3)} | {fmt(b['min'], 3)} "
          f"| {fmt(dq['mean'])} | {fmt(pq['mean'])} |")
    w("\nPrediction 5 expects **≥ 0.85**.")

    # ---- 3. metric disagreement -------------------------------------------
    w("\n## 3. Do DNSMOS and PESQ disagree?  <- PREDICTION 2\n")
    w("The parent plan chose DNSMOS over PESQ on the argument that PESQ measures "
      "fidelity while an attacker only needs the audio to *sound* acceptable. "
      "This is that argument's empirical test. If they agree, the argument is "
      "wrong and the screen can use PESQ.\n")
    w(f"| music | SNR where DNSMOS < {floor} | SNR where PESQ < {PESQ_FLOOR} | disagreement (dB) |")
    w("|---|---|---|---|")
    for music in musics:
        sel = [p for p in per if p["music"] == music]
        dq = stat([p["snr_quality_dies"] for p in sel])
        pq = stat([p["snr_pesq_dies"] for p in sel])
        gap = (pq["mean"] - dq["mean"]) if np.isfinite(pq["mean"]) and np.isfinite(dq["mean"]) else float("nan")
        w(f"| {music} | {fmt(dq['mean'])} | {fmt(pq['mean'])} | **{fmt(gap)}** |")
    w("\nA large positive disagreement means PESQ condemns the audio at a much "
      "higher SNR than DNSMOS does — i.e. PESQ would have filed a normal "
      "speech-over-music mix as destroyed, **confirming prediction 2**.")

    # ---- 4. does the watermark itself cost quality? ------------------------
    w("\n## 4. Watermarked vs unwatermarked mixes  <- PREDICTION 3\n")
    w("Mean DNSMOS gap between the two arms at matched SNR. "
      "`2026-08-17_attack-damage-control` measured the watermark's own cost at "
      "0.06 PESQ across 15 conditions; prediction 3 expects **< 0.1** here, "
      "which would mean the music does all the damage and the control arm can "
      "be dropped from later phases.\n")
    w("| music | mean arm gap (unwm − wm) | max |")
    w("|---|---|---|")
    for music in musics:
        sel = [p for p in per if p["music"] == music]
        g = stat([abs(p["arm_gap_dnsmos"]) for p in sel])
        w(f"| {music} | {fmt(g['mean'], 3)} | {fmt(g['max'], 3)} |")

    # ---- 5. does one number describe it? ----------------------------------
    w("\n## 5. Spread across clips  <- PREDICTION 4\n")
    w("`real_attacks_experiment.py` reported **4.9 dB** from a single "
      "speech/music pair. Prediction 4 says the 50-clip mean lands within ±3 dB "
      "of that but the spread is several dB, because masking depends on the "
      "speech content — so a single number does not describe this.\n")
    w("| music | mean | sd | min | max | range |")
    w("|---|---|---|---|---|---|")
    for music in musics:
        sel = [p for p in per if p["music"] == music]
        d = stat([p["snr_detection_dies"] for p in sel])
        rng = d["max"] - d["min"] if np.isfinite(d["max"]) else float("nan")
        w(f"| {music} | {fmt(d['mean'])} | {fmt(d['sd'])} | {fmt(d['min'])} "
          f"| {fmt(d['max'])} | **{fmt(rng)}** |")
    if len(musics) > 1:
        w("\n**Music-track dependence.** If the mean detection-death SNR moves "
          "materially between tracks, the headline number is track-specific and "
          "must be reported that way.")

    # ---- integrity ---------------------------------------------------------
    w("\n## Data integrity\n")
    n_bad_conf = sum(1 for r in rows if not np.isfinite(r["conf"]))
    n_bad_q = sum(1 for r in rows if not np.isfinite(r["dnsmos_ovrl"]))
    w(f"- rows with no confidence: **{n_bad_conf}** ({100.0*n_bad_conf/max(1,len(rows)):.2f}%)")
    w(f"- rows with no DNSMOS: **{n_bad_q}** ({100.0*n_bad_q/max(1,len(rows)):.2f}%) "
      f"— if this is not ~0 the quality axis is incomplete and section 1 is unreliable")
    n_nocross = sum(1 for p in per if not np.isfinite(p["snr_detection_dies"]))
    w(f"- (clip, music) pairs where detection never fell through {DETECT_THRESHOLD}: "
      f"**{n_nocross}/{len(per)}** — the watermark survived the whole sweep")
    if len(backends) > 1:
        w(f"- **{len(backends)} quality backends appear in this data: {backends}.** "
          f"Scores from different backends are not comparable; do not pool them.")

    w("\n## Conclusion\n")
    w("*(write this by hand after reading the tables — results/README.md rule 2)*")

    os.makedirs(DATA_DIR, exist_ok=True)
    if per:
        cols = sorted({k for d in per for k in d})
        with open(os.path.join(DATA_DIR, "music_metrics.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per)

    txt = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary_phase_a1.md"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {os.path.join(RESULTS_DIR, 'summary_phase_a1.md')}")
    print(f"wrote {os.path.join(DATA_DIR, 'music_metrics.csv')}")


if __name__ == "__main__":
    main(sys.argv[1:])
