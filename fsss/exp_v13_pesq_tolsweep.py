"""
fsss/exp_v13_pesq_tolsweep.py -- PESQ (and clean detection) vs tolerance_db, to map
the imperceptibility/strength trade-off and enable matched-PESQ comparisons.

tolerance_db is AWARE's loudness dial: delta = coeff * 10**(-tol/20), so LOWER tol =
LOUDER watermark = MORE robust but LESS imperceptible (lower PESQ). Stock AWARE = 6.

Sweeps tol in {-6,-3,0,3,6} for three configs, clean only (no attacks):
  orig_1000-4000   no-hop, native band 1000-4000  (tol=6 row == AWARE original)
  n1_500-4000      no-hop, band 500-4000           (the v12 matched-loudness control)
  hop_500-4000_N3  librosa-anchored hopping, N=3, band 500-4000  (thesis config)

All built as StaircaseAWAREEmbedder (n_bands=1 => no-hop, byte-identical to stock at
that band+tol) so tol is overridden cleanly without mutating the loaded embedder.
Detection = stock detector with the detector-band fix. Reports mean PESQ + mean clean
conf + bit_acc per (config, tol) over the clips.

    conda activate wmcompare
    python -m fsss.exp_v13_pesq_tolsweep
    python -m fsss.exp_v13_pesq_tolsweep --nclips 10 --tols -6,-4,-2,0,2,4,6
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

from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
DEFAULT_TOLS = [-6.0, -3.0, 0.0, 3.0, 6.0]
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def main(argv):
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    nclips = get_arg(argv, "--nclips", 3, int)
    nbands = get_arg(argv, "--nbands", 3, int)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    rate = get_arg(argv, "--rate", 3.5, float)
    key = get_arg(argv, "--key", "thesis")
    tols = get_list(argv, "--tols", DEFAULT_TOLS, float)

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    embedder, detector = load()
    stock_band = tuple(int(x) for x in embedder.embedding_bands)

    from aware.metrics.audio import PESQ
    pesq_metric = PESQ()

    # (label, n_bands, band, segment_mode)
    configs = [
        (f"orig_{stock_band[0]}-{stock_band[1]}", 1, stock_band, "fixed"),
        ("n1_500-4000", 1, (500, 4000), "fixed"),
        (f"hop_500-4000_N{nbands}", nbands, (500, 4000), "librosa"),
    ]

    print(f"clips    : {len(audios)}   tols: {tols}")
    print(f"configs  : {[c[0] for c in configs]}")
    print("tolerance_db: lower = louder (more robust, less imperceptible). stock=6.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v13_pesq_tolsweep.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "config", "band", "tol", "pesq", "conf", "bit_acc",
                     "detected"])

    agg = {}   # (label, tol) -> [(pesq, conf, acc), ...]

    for ci, (clip, audio) in enumerate(zip(clips, audios)):
        print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
        for (label, nb, band, mode) in configs:
            detector.embedding_bands = band
            for tol in tols:
                st = StaircaseAWAREEmbedder.from_embedder(
                    embedder, key=key, n_bands=nb, band_range=band,
                    segment_mode=mode, anchor="librosa_flux", anchor_rate=rate,
                    segment_len_s=seglen, tolerance_db=tol)
                st.embedding_bands = band
                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=st)
                pat, conf = detect_watermark(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                try:
                    m = min(len(wm), len(audio))
                    pv = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
                except Exception as e:
                    pv = float("nan")
                    print(f"    PESQ failed ({label} tol{tol}): {e}")
                agg.setdefault((label, tol), []).append((pv, float(conf), acc))
                writer.writerow([ci, label, f"{band[0]}-{band[1]}", tol,
                                 round(pv, 4), round(float(conf), 4), round(acc, 4),
                                 int(conf >= DET_CONF)])
                print(f"    {label:16s} tol={tol:>4}  PESQ={pv:.3f}  "
                      f"conf={float(conf):.3f}  acc={acc:.3f}")
    fout.close()

    def mean_of(label, tol, i):
        rows = agg.get((label, tol), [])
        vals = [r[i] for r in rows if not np.isnan(r[i])]
        return float(np.mean(vals)) if vals else float("nan")

    labels = [c[0] for c in configs]
    colw = 9

    def table(title, i, fmt="{:.3f}"):
        hdr = f"  {'config':16s}" + "".join(f"{('tol'+str(int(t))):>{colw}s}" for t in tols)
        print("\n" + "=" * len(hdr))
        print(title)
        print(hdr)
        for label in labels:
            line = f"  {label:16s}"
            for tol in tols:
                line += f"{fmt.format(mean_of(label, tol, i)):>{colw}s}"
            print(line)

    table("PESQ  (mean over clips; higher = more imperceptible)", 0)
    table("CLEAN CONF  (mean over clips; higher = stronger readout)", 1)
    table("BIT-ACC  (mean over clips)", 2)

    print(f"\nwrote {csv_path}")
    print("read: PESQ table = pick tols that MATCH PESQ across configs, then compare "
          "robustness at equal imperceptibility (Zhang protocol). hop perturbs 1/N "
          "bins/frame -> at the same tol its PESQ is HIGHER (quieter) than n1 full-band, "
          "so equal-PESQ needs the hop config run at a LOWER (louder) tol than n1.")


if __name__ == "__main__":
    main(sys.argv[1:])
