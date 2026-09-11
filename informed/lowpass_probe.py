"""
informed/lowpass_probe.py -- does AWARE score UNWATERMARKED lowpassed speech high?

Phase B gave lowpass no data: the clean-audio (null) threshold came out at 0.9948.
Two claims to test on the same 50 clips Phase B used, with NO watermark embedded:

  1. Mild lowpass (cutoff near 0.5) leaves speech intact and scores like clean
     speech -- low, around the 0.27 ceiling from 2026-08-14_detector-null-test.
  2. The score only jumps once the cutoff is so low that just a hum is left
     (cutoff 0.02 = ~440 Hz at 22.05 kHz, below AWARE's 500-4000 Hz band).

Highpass at the same cutoffs is the control. At high cutoffs it ALSO empties
AWARE's band, but leaves hiss instead of hum. If highpass stays low while
lowpass jumps, the trigger is tone-like content, not a missing band.

Written per clip and cutoff:
  conf         AWARE's reported score
  raw          the activation fed into AWARE's sigmoid, recovered by inverting
               BRH_activation_to_probability (full_length: x0=0.041,
               k_L=137.33, k_R=91.55). Readable where conf is saturated at ~1.
  bit_acc      decoded bits vs this run's random key -- should sit near 0.5
  flatness     spectral flatness of what is left: ~1 = noise/hiss, ~0 = tonal
  rms_db       level of what is left (rules out "it is just silence")

    python lowpass_probe.py              # all 50 clips
    python lowpass_probe.py --n 5        # quick check
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                     # noqa: E402

OUT_DIR = os.path.join(BASE, "results", "2026-09-11_aware-lowpass-null")
CLIPS_JSON = os.path.join(HERE, "clips.json")
CLIP_SECONDS = 10.0

# cutoff = fraction of the sample rate (julius convention); x 22050 = Hz
CUTOFFS = [0.45, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]
FILTERS = ["lowpass", "highpass"]

# BRH_activation_to_probability, mode full_length (aware/src/aware/utils/utils.py)
X0, K_L, K_R = 0.041, 137.3265360835137, 91.55102405567581


def conf_to_raw(p):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    k = K_R if p >= 0.5 else K_L
    return X0 + np.log(p / (1 - p)) / k


def flatness(y):
    """Spectral flatness over the whole clip: geomean / mean of the power spectrum."""
    ps = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))


def rms_db(y):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(y))) + 1e-12))


def main(argv):
    import cascade_lib as cl

    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else None
    clips = json.load(open(CLIPS_JSON))["clips"][:n]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "probe.csv")

    adapter = cl.get_adapter("aware")
    # A random key, so bit_acc is not measured against the fixed default message.
    rng = np.random.RandomState(0)
    adapter.truth = "".join(str(b) for b in rng.randint(0, 2, 20))

    sr = cl.SR_MASTER
    n_keep = int(CLIP_SECONDS * sr)
    fields = ["clip_id", "filter", "cutoff", "cutoff_hz", "conf", "raw",
              "bit_acc", "flatness", "rms_db"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, clip in enumerate(clips):
            org = cl.read_wav(clip["path"])[:n_keep].astype("float32")
            settings = [("none", 0.0)] + [(fl, c) for fl in FILTERS for c in CUTOFFS]
            for fl, c in settings:
                z = org if fl == "none" else A.apply(fl, c, org, sr)
                if z is None:
                    print(f"  {clip['clip_id']} {fl} {c}: attack unavailable")
                    continue
                z = np.asarray(z, dtype="float32")
                conf, _bits, acc = adapter.detect(z)
                w.writerow({
                    "clip_id": clip["clip_id"], "filter": fl, "cutoff": c,
                    "cutoff_hz": round(c * sr), "conf": round(conf, 6),
                    "raw": round(conf_to_raw(conf), 6), "bit_acc": round(acc, 4),
                    "flatness": round(flatness(z), 6), "rms_db": round(rms_db(z), 2),
                })
            f.flush()
            print(f"[{i + 1}/{len(clips)}] {clip['clip_id']} done")

    summarise(out_csv)


def summarise(out_csv):
    rows = list(csv.DictReader(open(out_csv)))
    lines = ["# AWARE on UNWATERMARKED speech, lowpass vs highpass", "",
             f"{len({r['clip_id'] for r in rows})} clips, no watermark embedded. "
             "Clean-speech ceiling from 2026-08-14_detector-null-test: 0.27.", "",
             "| filter | cutoff | Hz kept | conf median | conf max | "
             "# clips > 0.5 | raw median | bit_acc median | flatness median | rms dB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    keys = [("none", "0.0")] + [(fl, str(c)) for fl in FILTERS for c in CUTOFFS]
    for fl, c in keys:
        sub = [r for r in rows if r["filter"] == fl and r["cutoff"] == c]
        if not sub:
            continue
        col = lambda k: np.array([float(r[k]) for r in sub])
        hz = int(sub[0]["cutoff_hz"])
        kept = "all" if fl == "none" else (f"< {hz}" if fl == "lowpass" else f"> {hz}")
        conf = col("conf")
        lines.append(
            f"| {fl} | {c} | {kept} | {np.median(conf):.3f} | {conf.max():.3f} | "
            f"{int((conf > 0.5).sum())}/{len(sub)} | {np.median(col('raw')):.4f} | "
            f"{np.median(col('bit_acc')):.2f} | {np.median(col('flatness')):.4f} | "
            f"{np.median(col('rms_db')):.1f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(os.path.dirname(out_csv), "summary.md"), "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarise(os.path.join(OUT_DIR, "probe.csv"))
    else:
        main(sys.argv)
