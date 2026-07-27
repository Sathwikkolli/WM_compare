"""
fsss/exp_v10_hopping_bands.py -- key-driven subband HOPPING N-sweep across three
bands (1000-4000, 500-4000, 500-5000), N=1..6, on 3 Emilia clips. Clean first,
then the METAPXYL-proxy mastering chain. NO PESQ (per request).

WHY: a wider band keeps more FFT bins per sub-band, so higher N stays embeddable.
Bins/sub-band at N=6: 1000-4000->32, 500-4000->37, 500-5000->48. This tests whether
500-5000's extra room lets more sub-bands (more key-hopping entropy) survive.

DETECTOR-BAND FIX (critical): the AWARE detector ALSO zeros outside its own
embedding_bands at detection. Earlier hopping runs (v4/v6) set the band on the
embedder only, so for bands != 1000-4000 the low/high slice was embedded then
zeroed at detection -> artificially weak. Here we set embedding_bands on BOTH the
staircase embedder AND the detector, so every band is embedded+read consistently.

Config = StaircaseAWAREEmbedder (fixed 0.5s dwell, key-driven HMAC hop) at
tolerance_db=-6 (loud; PESQ ignored). n1 = full-band stripe every frame = the
no-hop baseline (== plain AWARE at that band). Detection = stock detector.

Run on a GPU node (a few minutes: 3 clips x 3 bands x 6 N = 54 embeds):
    conda activate wmcompare
    python -m fsss.exp_v10_hopping_bands
    python -m fsss.exp_v10_hopping_bands --nlist 2,3,4,5,6 --tol -6
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
from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
DEFAULT_BANDS = [(1000, 4000), (500, 4000), (500, 5000)]
DEFAULT_NLIST = [1, 2, 3, 4, 5, 6]      # n1 = no-hop baseline
# METAPXYL-proxy mastering chain (exp_v2 mapping): RComp+L1, D-Verb, 256k MP3,
# 16-bit dither, EQ/band-limit, everyday noise.
DEFAULT_ATTACKS = ["dynamic_compression", "echo", "mp3", "quantization",
                   "lowpass", "gaussian_noise"]
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def get_bands(argv):
    if "--bands" in argv:
        out = []
        for tok in argv[argv.index("--bands") + 1].split(","):
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
    tol = get_arg(argv, "--tol", -6.0, float)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    key = get_arg(argv, "--key", "thesis")
    nlist = get_list(argv, "--nlist", DEFAULT_NLIST, int)
    attacks = get_list(argv, "--attacks", DEFAULT_ATTACKS, str)
    bands = [(f"{lo}-{hi}", (lo, hi)) for (lo, hi) in get_bands(argv)]

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    print(f"clips    : {len(audios)}   bands: {[b[0] for b in bands]}")
    print(f"N        : {nlist}   (n1 = no-hop baseline)")
    print(f"tol      : {tol}   seglen: {seglen}   key: '{key}'   PESQ: skipped")
    print(f"attacks  : {attacks}  (METAPXYL-proxy)\n")

    embedder, detector = load()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v10_hopping_bands.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "band", "n_bands", "tol", "attack", "param",
                     "bit_acc", "conf", "detected"])

    agg = {}    # (band, N, attack) -> [(acc, det), ...]

    def rec(bn, n, atk, acc, conf):
        agg.setdefault((bn, n, atk), []).append((acc, int(conf >= DET_CONF)))

    for ci, (clip, audio) in enumerate(zip(clips, audios)):
        print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
        for bn, band in bands:
            detector.embedding_bands = band        # <-- the fix: detector matches band
            for n in nlist:
                st = StaircaseAWAREEmbedder.from_embedder(
                    embedder, key=key, n_bands=n, band_range=band,
                    segment_mode="fixed", segment_len_s=seglen, tolerance_db=tol)
                st.embedding_bands = band
                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=st)
                pat, conf = detect_watermark(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                rec(bn, n, "clean", acc, conf)
                writer.writerow([ci, bn, n, tol, "clean", "", round(acc, 4),
                                 round(float(conf), 4), int(conf >= DET_CONF)])
                for atk in attacks:
                    for label, param in vox_attacks.VOX_GRID.get(atk, []):
                        try:
                            wa = vox_attacks.apply(atk, param, wm.astype("float32"), WORK_SR)
                        except Exception:
                            continue
                        if wa is None:
                            continue
                        pat, conf = detect_watermark(wa, WORK_SR, detector)
                        acc = bit_acc(bits, pat)
                        rec(bn, n, atk, acc, conf)
                        writer.writerow([ci, bn, n, tol, atk, label, round(acc, 4),
                                         round(float(conf), 4), int(conf >= DET_CONF)])
    fout.close()

    def stats(bn, n, atk):
        rows = agg.get((bn, n, atk), [])
        if not rows:
            return float("nan"), 0, 0
        return float(np.mean([r[0] for r in rows])), sum(r[1] for r in rows), len(rows)

    names = [bn for bn, _ in bands]

    # ---- 1) CLEAN grid: which (band, N) detect clean? --------------------- #
    print("\n" + "=" * (8 + 15 * len(names)))
    print("CLEAN detection  (mean bit_acc, detected/clips)")
    print(f"  {'N':3s}" + "".join(f"{n:>15s}" for n in names))
    for n in nlist:
        line = f"  {('n'+str(n)):3s}"
        for bn in names:
            a, d, t = stats(bn, n, "clean")
            line += f"{a:.2f} {d}/{t}".rjust(15)
        print(line)

    # ---- 2) per band: N x attack detection matrix ------------------------- #
    for bn in names:
        print("\n" + "#" * 60)
        print(f"###  band {bn}   (det/total per N x METAPXYL-proxy stage)")
        print("#" * 60)
        hdr = f"  {'N':4s}" + "".join(f"{a[:8]:>9s}" for a in attacks)
        print(hdr)
        for n in nlist:
            line = f"  n{n:<3d}"
            for atk in attacks:
                _, d, t = stats(bn, n, atk)
                line += f"{d:>3d}/{t:<3d}".rjust(9)
            print(line)

    print("\n" + "=" * 60)
    print(f"wrote {csv_path}")
    print("read: CLEAN grid first — which (band,N) even detect (conf>=0.5). Wider "
          "bands (500-5000) keep more bins/sub-band so should support higher N. Then "
          "per-band matrices: does hopping survive the mastering-proxy chain, and does "
          "500-5000 let you push N higher than 1000-4000? (compress/echo usually all-pass;"
          " lowpass/noise/mp3 are the discriminators.)")


if __name__ == "__main__":
    main(sys.argv[1:])
