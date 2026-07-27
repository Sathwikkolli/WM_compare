"""
fsss/exp_v14_tol_attack_sweep.py -- robustness AND PESQ vs tolerance_db, for the
stock/no-hop/hop configs. This is v13 (tol->PESQ sweep) crossed with v12 (attack
suite): for each config x tol we embed, measure clean PESQ, then run the full attack
sweep. Gives robustness-vs-tol curves at known PESQ -> the data needed to compare
configs at MATCHED imperceptibility (Zhang protocol) instead of matched tol.

tolerance_db: lower = louder = more robust but lower PESQ. Stock AWARE = 6.

Configs (all via StaircaseAWAREEmbedder; n_bands=1 => no-hop == stock at that band):
  orig_1000-4000   no-hop, native band            (tol=6 == AWARE original)
  n1_500-4000      no-hop, band 500-4000
  hop_500-4000_N3  librosa-anchored hopping, N=3, band 500-4000

Detection = stock detector + detector-band fix. Attacks = METAPXYL-proxy + important
VoxWatermark, each over its VOX_GRID strengths (opus capped at 256k -- libopus max).

DEFAULT nclips=3 because this is 5 tols x v12's cost. Scale with --nclips / --tols /
--attacks. Runtime is codec-dominated (encodec/mp3/opus).
    conda activate wmcompare
    python -m fsss.exp_v14_tol_attack_sweep
    python -m fsss.exp_v14_tol_attack_sweep --nclips 10 --attacks mp3,lowpass,time_stretch
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
DEFAULT_TOLS = [-6.0, -3.0, 0.0, 3.0, 6.0]
METAPXYL_PROXY = ["dynamic_compression", "echo", "mp3", "quantization",
                  "lowpass", "gaussian_noise"]
VOX_IMPORTANT = ["time_stretch", "time_jitter", "highpass", "encodec",
                 "background_noise", "opus"]
DEFAULT_ATTACKS = METAPXYL_PROXY + VOX_IMPORTANT
OPUS_MAX_KBPS = 256                      # libopus ceiling; skip the invalid 496k strength
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
    attacks = get_list(argv, "--attacks", DEFAULT_ATTACKS, str)

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

    configs = [
        (f"orig_{stock_band[0]}-{stock_band[1]}", 1, stock_band, "fixed"),
        ("n1_500-4000", 1, (500, 4000), "fixed"),
        (f"hop_500-4000_N{nbands}", nbands, (500, 4000), "librosa"),
    ]
    labels = [c[0] for c in configs]

    print(f"clips    : {len(audios)}   tols: {tols}")
    print(f"configs  : {labels}")
    print(f"attacks  : {attacks}")
    print("tolerance_db: lower = louder (more robust, lower PESQ). stock=6.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v14_tol_attack_sweep.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "config", "band", "tol", "attack", "param",
                     "pesq", "conf", "bit_acc", "detected"])

    surv = {}    # (label, tol, attack) -> [(acc, det), ...]
    pesq_agg = {}  # (label, tol) -> [pesq, ...]

    def rec(label, tol, atk, acc, det):
        surv.setdefault((label, tol, atk), []).append((acc, det))

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
                # clean: PESQ + detection
                pat, conf = detect_watermark(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)
                try:
                    m = min(len(wm), len(audio))
                    pv = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
                except Exception:
                    pv = float("nan")
                pesq_agg.setdefault((label, tol), []).append(pv)
                rec(label, tol, "clean", acc, int(conf >= DET_CONF))
                writer.writerow([ci, label, f"{band[0]}-{band[1]}", tol, "clean", "",
                                 "" if np.isnan(pv) else round(pv, 4),
                                 round(float(conf), 4), round(acc, 4),
                                 int(conf >= DET_CONF)])
                print(f"    {label:16s} tol={tol:>4}  PESQ={pv:.3f}  "
                      f"conf={float(conf):.3f}  acc={acc:.3f}")
                # attacks
                for atk in attacks:
                    for plabel, param in vox_attacks.VOX_GRID.get(atk, []):
                        if atk == "opus" and float(param) > OPUS_MAX_KBPS:
                            continue
                        try:
                            wa = vox_attacks.apply(atk, param, wm.astype("float32"), WORK_SR)
                        except Exception:
                            continue
                        if wa is None:
                            continue
                        pat, conf = detect_watermark(wa, WORK_SR, detector)
                        acc = bit_acc(bits, pat)
                        rec(label, tol, atk, acc, int(conf >= DET_CONF))
                        writer.writerow([ci, label, f"{band[0]}-{band[1]}", tol, atk,
                                         plabel, "", round(float(conf), 4),
                                         round(acc, 4), int(conf >= DET_CONF)])
    fout.close()

    # ----------------------------------------------------------------------- #
    def pesq_mean(label, tol):
        vals = [v for v in pesq_agg.get((label, tol), []) if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def det_total(label, tol, atk):
        rows = surv.get((label, tol, atk), [])
        return sum(r[1] for r in rows), len(rows)

    def total_attacks(label, tol):
        d = sum(det_total(label, tol, a)[0] for a in attacks)
        t = sum(det_total(label, tol, a)[1] for a in attacks)
        return d, t

    colw = 10
    hdr = f"  {'config':16s}" + "".join(f"{('tol'+str(int(t))):>{colw}s}" for t in tols)

    print("\n" + "=" * len(hdr))
    print("PESQ  (mean over clips; higher = more imperceptible)")
    print(hdr)
    for label in labels:
        line = f"  {label:16s}"
        for tol in tols:
            line += f"{pesq_mean(label, tol):>{colw}.3f}"
        print(line)

    print("\n" + "=" * len(hdr))
    print("TOTAL ATTACK SURVIVAL  (detected/total over attacks x strengths x clips)")
    print(hdr)
    for label in labels:
        line = f"  {label:16s}"
        for tol in tols:
            d, t = total_attacks(label, tol)
            line += f"{f'{d}/{t}':>{colw}s}"
        print(line)

    # per-config tol x attack matrix
    for label in labels:
        print("\n" + "#" * 60)
        print(f"###  {label}   (det/total per tol x attack)")
        print("#" * 60)
        h = f"  {'tol':5s}" + "".join(f"{a[:8]:>9s}" for a in attacks)
        print(h)
        for tol in tols:
            line = f"  {int(tol):<5d}"
            for atk in attacks:
                d, t = det_total(label, tol, atk)
                line += f"{f'{d}/{t}':>9s}"
            print(line)

    print(f"\nwrote {csv_path}")
    print("read: cross the PESQ table with the survival table. To compare configs at "
          "EQUAL PESQ, read across rows to find the tol where each config hits the same "
          "PESQ, then compare their survival there. Expect hop to reach a given PESQ at a "
          "LOWER tol (louder) than n1 -> at matched PESQ the gap shrinks vs the matched-tol "
          "v12 result.")


if __name__ == "__main__":
    main(sys.argv[1:])
