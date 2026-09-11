"""
informed/null_probe.py -- both detectors, clean vs watermarked, across filter cutoffs.

Phase B's lowpass null (data/null/null_lowpass.csv) showed two separate things:

  blind     AWARE conf on CLEAN audio is high only at cutoff 0.02 (hum, 0.9948),
            normal everywhere else. Confirmed by lowpass_probe.py (48/50 > 0.5).
  informed  correlation on CLEAN audio is ~0.5-0.6 at EVERY strength, even with
            almost no filtering -- vs ~0.009 for gaussian noise.

Candidate causes for the informed one, which this test separates:

  H1  RESAMPLING. AWARE embeds at 16 kHz and the file comes back to 22.05 kHz,
      so the watermarked file holds nothing above 8 kHz. The reference
      w_clean = wm - org is then (watermark) MINUS (the voice above 8 kHz). A
      lowpass residual is also minus the voice above the cutoff, so the two
      match for reasons unrelated to any watermark.
      -> test: the same detector run at 16 kHz, where the reference is the
         true watermark. Columns inf16_*.
  H2  WINDOWING. The score is the mean correlation over 42 ms windows, and
      windows are gated on the REFERENCE's energy, not the residual's. A window
      where the attack changed almost nothing still gets a full vote.
      -> test: one correlation over the whole clip. Columns *_glob.
  H3  HOST SHAPE. The residual of a filter IS voice, and AWARE shapes its
      watermark to the voice. That would survive both fixes above.

Groups:
  null  200 clips from real_audio/null_cache.npz -- the SAME clean set that set
        Phase B's thresholds. Attack the UNwatermarked audio; the reference is
        the watermark each clip WOULD have carried.
  wm    the 50 Phase B clips, embedded with a random 20-bit key each.

Settings: no attack, then lowpass and highpass at 10 cutoffs each.

    python null_probe.py --n-null 5 --n-wm 2       # quick check
    python null_probe.py                           # full run (sbatch)
    python null_probe.py --summary                 # re-print tables from the CSV
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                     # noqa: E402
import informed_detector as ID                 # noqa: E402

OUT_DIR = os.path.join(BASE, "results", "2026-09-11_aware-lowpass-null")
OUT_CSV = os.path.join(OUT_DIR, "null_probe.csv")
CACHE = os.path.join(BASE, "real_audio", "null_cache.npz")
CLIPS_JSON = os.path.join(HERE, "clips.json")

CLIP_SECONDS = 10.0
CUTOFFS = [0.45, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]
FPR = 0.01

# BRH_activation_to_probability, full_length (aware/src/aware/utils/utils.py)
X0, K_L, K_R = 0.041, 137.3265360835137, 91.55102405567581

SCORES = [
    ("conf",       "blind: AWARE conf"),
    ("inf22_win",  "informed 22 kHz, windowed  (= Phase B)"),
    ("inf22_glob", "informed 22 kHz, whole clip"),
    ("inf16_win",  "informed 16 kHz, windowed"),
    ("inf16_glob", "informed 16 kHz, whole clip"),
]

FIELDS = ["group", "clip", "filter", "cutoff", "cutoff_hz", "conf", "raw", "bit_acc",
          "inf22_win", "inf22_glob", "inf16_win", "inf16_glob",
          "resid_db", "wref_hi_frac"]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def conf_to_raw(p):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return X0 + np.log(p / (1 - p)) / (K_R if p >= 0.5 else K_L)


def hi_frac(w, sr, f_hz=8000.0):
    """Share of w's energy above f_hz. H1 predicts this is large for w_clean at 22 kHz."""
    P = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), 1.0 / sr)
    tot = float(P.sum())
    return float(P[f > f_hz].sum() / tot) if tot > 0 else float("nan")


def settings(filters):
    return [("none", 0.0)] + [(fl, c) for fl in filters for c in CUTOFFS]


def fnum(x, nd=6):
    return round(float(x), nd) if x is not None and np.isfinite(x) else ""


def measure(cl, adapter, group, clip, org, wm, bits, filters, writer, sr):
    """Every setting for one clip. `src` is what gets attacked: org for the null
    group (no watermark present), wm for the watermarked group."""
    src = org if group == "null" else wm
    o16, w16 = cl.resample(org, sr, 16000), cl.resample(wm, sr, 16000)
    whf = hi_frac(wm - org, sr)
    e_org = float(np.dot(org, org)) + 1e-20

    for fl, c in settings(filters):
        z = src if fl == "none" else A.apply(fl, c, src, sr)
        if z is None:
            print(f"    {clip} {fl} {c}: attack unavailable")
            continue
        z = np.asarray(z, dtype="float32")[:len(src)]

        adapter.truth = bits
        conf, _b, acc = adapter.detect(z)
        r22 = ID.score(org, wm, z, sr=sr, method="scalar")
        r16 = ID.score(o16, w16, cl.resample(z, sr, 16000), sr=16000, method="scalar")
        d = z - src
        writer.writerow({
            "group": group, "clip": clip, "filter": fl, "cutoff": c,
            "cutoff_hz": int(round(c * sr)),
            "conf": fnum(conf), "raw": fnum(conf_to_raw(conf)), "bit_acc": fnum(acc, 4),
            "inf22_win": fnum(r22["corr_windowed"]), "inf22_glob": fnum(r22["corr_global"]),
            "inf16_win": fnum(r16["corr_windowed"]), "inf16_glob": fnum(r16["corr_global"]),
            # how much the attack changed the audio, relative to the voice
            "resid_db": fnum(10 * np.log10(float(np.dot(d, d)) / e_org + 1e-20), 2),
            "wref_hi_frac": fnum(whf, 4),
        })


def main(argv):
    import cascade_lib as cl

    n_null = get_arg(argv, "--n-null", 200, int)
    n_wm = get_arg(argv, "--n-wm", 50, int)
    filters = get_arg(argv, "--filters", "lowpass,highpass").split(",")
    sr = cl.SR_MASTER
    n_keep = int(CLIP_SECONDS * sr)

    if not os.path.exists(CACHE):
        raise SystemExit(f"{CACHE} missing -- it is built by `null_calibrate.py --prep`")
    zc = np.load(CACHE, allow_pickle=False)
    if int(zc["sr"]) != sr:
        raise SystemExit(f"null cache is at {int(zc['sr'])} Hz, expected {sr}")
    orgs, wcs, msgs = zc["org"][:n_null], zc["w_clean"][:n_null], zc["messages"][:n_null]
    pos = json.load(open(CLIPS_JSON))["clips"][:n_wm]

    print("loading AWARE adapter...")
    adapter = cl.get_adapter("aware")
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(11)
    t0 = time.time()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        for i in range(len(orgs)):
            org = orgs[i].astype("float32")
            measure(cl, adapter, "null", f"n{i:03d}", org, org + wcs[i].astype("float32"),
                    str(msgs[i]), filters, w, sr)
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  null {i + 1}/{len(orgs)}  ({time.time() - t0:.0f}s)")

        for j, clip in enumerate(pos):
            org = cl.read_wav(clip["path"])[:n_keep].astype("float32")
            bits = "".join(rng.choice(["0", "1"]) for _ in range(20))
            adapter.truth = bits
            wm = np.asarray(adapter.embed(org), dtype="float32")[:n_keep]
            measure(cl, adapter, "wm", clip["clip_id"], org, wm, bits, filters, w, sr)
            f.flush()
            print(f"  wm {j + 1}/{len(pos)} {clip['clip_id']}  ({time.time() - t0:.0f}s)")

    summarise()


# --------------------------------------------------------------------------- #
#  summary
# --------------------------------------------------------------------------- #
def threshold_at(scores, fpr):
    """Same rule as null_calibrate.threshold_at: lowest score admitting <= fpr of the null."""
    s = np.sort(np.asarray(scores, dtype=float))
    if not len(s):
        return float("nan")
    k = int(np.ceil((1.0 - fpr) * len(s)))
    return float(s[-1]) + 1e-9 if k >= len(s) else float(s[k])


def summarise():
    rows = list(csv.DictReader(open(OUT_CSV)))
    n_null = len({r["clip"] for r in rows if r["group"] == "null"})
    n_wm = len({r["clip"] for r in rows if r["group"] == "wm"})
    keys = []
    for r in rows:
        k = (r["filter"], r["cutoff"])
        if k not in keys:
            keys.append(k)

    def vals(group, fl, c, col):
        return np.array([float(r[col]) for r in rows
                         if r["group"] == group and r["filter"] == fl
                         and r["cutoff"] == c and r[col] != ""])

    def label(fl, c):
        if fl == "none":
            return "no attack"
        hz = int(round(float(c) * 22050))
        return f"{fl} {c} ({'<' if fl == 'lowpass' else '>'}{hz} Hz)"

    L = ["# Null probe: both detectors, clean vs watermarked", "",
         f"{n_null} clean clips (Phase B null set), {n_wm} watermarked clips. "
         f"Threshold = {FPR:.0%} false alarms on the clean clips.", ""]

    whf = np.array([float(r["wref_hi_frac"]) for r in rows
                    if r["group"] == "null" and r["filter"] == "none" and r["wref_hi_frac"]])
    if len(whf):
        L += [f"**H1 check:** share of the 22 kHz reference `wm - org` lying above 8 kHz "
              f"(where AWARE puts no watermark): median {np.median(whf):.1%}, "
              f"max {whf.max():.1%}.", ""]

    # 1. detection rate matrix -- the headline
    L += ["## Detection rate at 1% false alarms (share of watermarked clips caught)", "",
          "| setting | " + " | ".join(c for c, _ in SCORES) + " |",
          "|---|" + "---|" * len(SCORES)]
    for fl, c in keys:
        cells = []
        for col, _ in SCORES:
            nv, wv = vals("null", fl, c, col), vals("wm", fl, c, col)
            if not len(nv) or not len(wv):
                cells.append("--")
                continue
            thr = threshold_at(nv, FPR)
            cells.append(f"{np.mean(wv > thr):.0%}")
        L.append(f"| {label(fl, c)} | " + " | ".join(cells) + " |")

    # 2. per-score detail
    for col, title in SCORES:
        L += ["", f"## {title}", "",
              "| setting | clean median | clean max | **1% threshold** | "
              "watermarked median | detected | n clean / wm |",
              "|---|---|---|---|---|---|---|"]
        for fl, c in keys:
            nv, wv = vals("null", fl, c, col), vals("wm", fl, c, col)
            if not len(nv):
                L.append(f"| {label(fl, c)} | -- | -- | -- | -- | -- | 0 / {len(wv)} |")
                continue
            thr = threshold_at(nv, FPR)
            wmed = f"{np.median(wv):.3f}" if len(wv) else "--"
            det = f"{np.mean(wv > thr):.0%}" if len(wv) else "--"
            L.append(f"| {label(fl, c)} | {np.median(nv):.3f} | {nv.max():.3f} | "
                     f"**{thr:.3f}** | {wmed} | {det} | {len(nv)} / {len(wv)} |")

    # 3. bit accuracy -- does a high conf come with the RIGHT bits?
    L += ["", "## Blind bit accuracy (0.5 = chance)", "",
          "| setting | clean median | watermarked median |", "|---|---|---|"]
    for fl, c in keys:
        nv, wv = vals("null", fl, c, "bit_acc"), vals("wm", fl, c, "bit_acc")
        L.append(f"| {label(fl, c)} | {np.median(nv) if len(nv) else float('nan'):.2f} | "
                 f"{np.median(wv) if len(wv) else float('nan'):.2f} |")

    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT_DIR, "null_probe_summary.md"), "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarise()
    else:
        main(sys.argv)
