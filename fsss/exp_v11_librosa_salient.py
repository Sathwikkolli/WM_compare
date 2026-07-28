"""
fsss/exp_v11_librosa_salient.py -- salient-point (librosa) subband HOPPING vs
baseline AWARE, confidence under the full VoxWatermark attack set (+ METAPXYL
stages, which are a subset). ONE 5-second Emilia clip.

3 models:
  AWARE          -- stock embedder, band 1000-4000, NO hopping (baseline).
  librosa 500-4000 -- StaircaseAWAREEmbedder, segment_mode="librosa" (hop at
                    spectral-flux salient points), band 500-4000, detector matched.
  librosa 500-5000 -- same, band 500-5000.

4 tables = N in {3,4} x tolerance_db in {3,6}. Each cell = detector CONFIDENCE.
(The baseline AWARE has no N; it appears in every table at that table's tolerance.)

Detector-band fix applied: embedding_bands set on BOTH embedder and detector.

Run on a GPU node (a few embeds + the full vox attack grid on each; ~10-15 min):
    conda activate wmcompare
    python -m fsss.exp_v11_librosa_salient
    python -m fsss.exp_v11_librosa_salient --clip audio/x.wav --key thesis
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
BASELINE_BAND = (1000, 4000)
LIBROSA_MODELS = [("lib500-4000", (500, 4000)), ("lib500-5000", (500, 5000))]
N_LIST = [3, 4]
TOL_LIST = [6.0, 3.0]
# METAPXYL-chain proxy stages (a subset of the vox families) -- marked with * .
METAPXYL = {"dynamic_compression", "echo", "mp3", "quantization", "lowpass", "gaussian_noise"}
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no Emilia clip found; pass --clip PATH")
            return
        clip = clips[0]
    audio = load_16k(clip)[:int(round(dur * WORK_SR))]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    families = list(vox_attacks.VOX_GRID.keys())
    rows = [("clean", "")] + [(fam, label) for fam in families
                              for label, _ in vox_attacks.VOX_GRID[fam]]

    print(f"clip : {clip}   ({len(audio)/WORK_SR:.2f}s)")
    print(f"key  : '{key}'   attacks: {len(rows)-1} vox strengths (+clean)")
    print(f"models: AWARE(1000-4000, no hop) | librosa 500-4000 | librosa 500-5000")
    print(f"tables: N in {N_LIST} x tol in {TOL_LIST}\n")

    embedder, detector = load()

    # ---- build every embed once --------------------------------------------- #
    embeds = []   # dict(id, model, N, tol, band, wm)

    for tol in TOL_LIST:                                   # baseline AWARE (no hop)
        embedder.embedding_bands = BASELINE_BAND
        embedder.tolerance_db = tol
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=embedder)
        embeds.append(dict(id=f"aware_t{tol}", model="AWARE", N=None,
                           tol=tol, band=BASELINE_BAND, wm=wm))
        print(f"  embedded  AWARE            tol={tol}")

    for mlabel, band in LIBROSA_MODELS:                    # librosa salient-point hop
        for N in N_LIST:
            for tol in TOL_LIST:
                st = StaircaseAWAREEmbedder.from_embedder(
                    embedder, key=key, n_bands=N, band_range=band,
                    segment_mode="librosa", anchor="librosa_flux", tolerance_db=tol)
                st.embedding_bands = band
                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=st)
                embeds.append(dict(id=f"{mlabel}_N{N}_t{tol}", model=mlabel, N=N,
                                   tol=tol, band=band, wm=wm))
                print(f"  embedded  {mlabel:14s} N={N} tol={tol}")

    # ---- confidence of each embed under clean + every vox attack ------------ #
    conf = {}   # id -> {row_label: conf}
    for rec in embeds:
        detector.embedding_bands = rec["band"]             # detector matches embed band
        wm = rec["wm"]
        d = {}
        _, c = detect_watermark(wm, WORK_SR, detector)
        d["clean/"] = float(c)
        for fam in families:
            for label, param in vox_attacks.VOX_GRID[fam]:
                rk = f"{fam}/{label}"
                try:
                    wa = vox_attacks.apply(fam, param, wm.astype("float32"), WORK_SR)
                except Exception:
                    wa = None
                if wa is None:
                    d[rk] = float("nan")
                    continue
                _, c = detect_watermark(wa, WORK_SR, detector)
                d[rk] = float(c)
        conf[rec["id"]] = d
        print(f"  scored    {rec['id']}")

    # ---- 4 tables + CSV ----------------------------------------------------- #
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v11_librosa_salient.csv")
    fout = open(csv_path, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["N", "tol", "attack", "param", "metapxyl",
                "AWARE_conf", "lib500-4000_conf", "lib500-5000_conf"])

    def cval(mid, rk):
        return conf.get(mid, {}).get(rk, float("nan"))

    for N in N_LIST:
        for tol in TOL_LIST:
            aid = f"aware_t{tol}"
            l4 = f"lib500-4000_N{N}_t{tol}"
            l5 = f"lib500-5000_N{N}_t{tol}"
            print("\n" + "=" * 70)
            print(f"  N = {N}    tol = {tol}    (detector confidence;  * = METAPXYL stage)")
            print("=" * 70)
            print(f"  {'attack':26s}{'AWARE':>11s}{'lib500-4k':>12s}{'lib500-5k':>12s}")
            for fam, label in rows:
                rk = f"{fam}/{label}"
                mark = "*" if fam in METAPXYL else " "
                name = (fam if not label else f"{fam}/{label}")[:24]
                a, b4, b5 = cval(aid, rk), cval(l4, rk), cval(l5, rk)
                print(f" {mark}{name:25s}{a:11.3f}{b4:12.3f}{b5:12.3f}")
                w.writerow([N, tol, fam, label, int(fam in METAPXYL),
                            round(a, 4), round(b4, 4), round(b5, 4)])
    fout.close()
    print("\n" + "=" * 70)
    print(f"wrote {csv_path}")
    print("read: conf>=0.5 = detected. Compare baseline AWARE vs librosa-salient hopping "
          "(500-4000 vs 500-5000) per attack, across the 4 N x tol tables.")


if __name__ == "__main__":
    main(sys.argv[1:])
