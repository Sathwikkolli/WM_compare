"""
fsss/exp_v9_band_compare.py -- clean comparison: stock AWARE at its ORIGINAL band
(1000-4000) vs the WIDENED band (500-4000), across tolerance thresholds, on 3 Emilia
clips, clean + basic attacks. NO hopping (plain stock embedder), NO PESQ.

WHY THIS IS THE FIRST *CORRECT* WIDE-BAND TEST
----------------------------------------------
The AWARE DETECTOR also zeros everything outside its own embedding_bands before
reading (src/aware/detection/multibit_detector.py: mask = ~(freqs in band);
magnitude[ids] = 0.0). Our earlier v4b test set the band on the EMBEDDER only and
left the detector at 1000-4000, so the 500-1000 watermark was embedded then ZEROED
at detection -> wasted -> 500-4000 looked worse. Here we set embedding_bands on
BOTH the embedder AND the detector, so 500-4000 is embedded and read consistently.

Each config = the plain stock AWAREEmbedder with two attributes set:
  embedder.embedding_bands = band ;  detector.embedding_bands = band
  embedder.tolerance_db    = tol   (swept; lower = louder/stronger)

Attacks (basic set); highpass is the KEY differentiator -- it removes low
frequencies, so it should hurt the 500-4000 band (which puts energy in 500-1000)
more than 1000-4000. Metrics: bit_acc + detection (conf>=0.5), averaged over clips.

Run on a GPU node in wmcompare (a few minutes: 3 clips x 2 bands x len(tols) embeds):
    conda activate wmcompare
    python -m fsss.exp_v9_band_compare
    python -m fsss.exp_v9_band_compare --tols 6,0,-6 --attacks mp3,highpass --nclips 3
"""

import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

import vox_attacks
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
DEFAULT_BANDS = [(1000, 4000), (500, 4000), (500, 6000), (500, 8000)]
DEFAULT_TOLS = [6.0, 3.0, 0.0, -3.0, -6.0]
DEFAULT_ATTACKS = ["mp3", "lowpass", "highpass", "gaussian_noise"]   # highpass = differentiator
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def get_bands(argv):
    """--bands 1000-4000,500-4000,500-6000,500-8000 -> [(lo,hi), ...]."""
    if "--bands" in argv:
        spec = argv[argv.index("--bands") + 1]
        out = []
        for tok in spec.split(","):
            lo, hi = tok.split("-")
            out.append((int(lo), int(hi)))
        return out
    return DEFAULT_BANDS


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def main(argv):
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    nclips = get_arg(argv, "--nclips", 3, int)
    tols = get_list(argv, "--tols", DEFAULT_TOLS, float)
    attacks = get_list(argv, "--attacks", DEFAULT_ATTACKS, str)
    bands = [(f"{lo}-{hi}", (lo, hi)) for (lo, hi) in get_bands(argv)]

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    print(f"clips    : {len(audios)}  (5s each)")
    print(f"bands    : {[bn for bn, _ in bands]}")
    print(f"tols     : {tols}   (lower = louder/stronger)")
    print(f"attacks  : {attacks}   (highpass = the key band differentiator)")
    print("PESQ     : skipped (per request)\n")

    embedder, detector = load()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v9_band_compare.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "band", "band_lo", "band_hi", "tol", "attack", "param",
                     "bit_acc", "conf", "detected"])

    # agg[(band, tol, attack)] = list of (bit_acc, detected) over clips x strengths
    agg = {}

    def record(band_name, tol, attack, acc, conf):
        agg.setdefault((band_name, tol, attack), []).append((acc, int(conf >= DET_CONF)))

    for ci, (clip, audio) in enumerate(zip(clips, audios)):
        print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
        for band_name, band in bands:
            embedder.embedding_bands = band
            detector.embedding_bands = band          # <-- the fix: detector matches embedder
            for tol in tols:
                embedder.tolerance_db = tol
                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=embedder)
                # clean
                pat, conf = detect_watermark(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                record(band_name, tol, "clean", acc, conf)
                writer.writerow([ci, band_name, band[0], band[1], tol, "clean", "",
                                 round(acc, 4), round(float(conf), 4), int(conf >= DET_CONF)])
                # attacks
                for attack in attacks:
                    for label, param in vox_attacks.VOX_GRID.get(attack, []):
                        try:
                            wa = vox_attacks.apply(attack, param, wm.astype("float32"), WORK_SR)
                        except Exception:
                            continue
                        if wa is None:
                            continue
                        pat, conf = detect_watermark(wa, WORK_SR, detector)
                        acc = bit_acc(bits, pat)
                        record(band_name, tol, attack, acc, conf)
                        writer.writerow([ci, band_name, band[0], band[1], tol, attack,
                                         label, round(acc, 4), round(float(conf), 4),
                                         int(conf >= DET_CONF)])
    fout.close()

    # ---- side-by-side summary per tolerance ------------------------------- #
    def stats(band_name, tol, attack):
        rows = agg.get((band_name, tol, attack), [])
        if not rows:
            return float("nan"), 0, 0
        accs = [r[0] for r in rows]
        dets = sum(r[1] for r in rows)
        return float(np.mean(accs)), dets, len(rows)

    rowlabels = ["clean"] + attacks
    names = [bn for bn, _ in bands]
    W = 14
    for tol in tols:
        width = 12 + W * len(names)
        print("\n" + "#" * width)
        print(f"###  tolerance = {tol}    cell = detected/total  mean_bit_acc")
        print("#" * width)
        print(f"  {'metric':10s}" + "".join(f"{n:>{W}s}" for n in names))
        for attack in rowlabels:
            line = f"  {attack:10s}"
            for bn in names:
                a, d, t = stats(bn, tol, attack)
                line += f"{d:>2d}/{t:<2d} {a:4.2f}".rjust(W)
            print(line)

    print("\n" + "=" * width)
    print(f"wrote {csv_path}")
    print("read: all bands share the low edge 500 except '1000-4000'. Expect wider "
          "bands (500-6000/8000) to lose under MP3 & LOWPASS (those strip >4kHz), and "
          "the 500-* bands to lose under HIGHPASS (strips 500-1000). Clean should stay "
          "1.0 for all now that the detector band matches.")


if __name__ == "__main__":
    main(sys.argv[1:])
