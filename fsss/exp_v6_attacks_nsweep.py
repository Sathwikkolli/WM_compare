"""
fsss/exp_v6_attacks_nsweep.py -- the ROBUSTNESS test: attack key-driven subband
hopping across N, in the locked survival band (1000-4000 Hz), on ONE 5s Emilia clip.

Clean detection (v4/v5) is necessary but NOT sufficient. The thesis claim is that
key-driven subband HOPPING helps the watermark SURVIVE real edits. This runs each
config through a suite of attacks (vox_attacks, same as exp_v2) and reports how many
bits survive.

CONFIGS (band 1000-4000, tol=-6, seglen=0.5):
  stock  -- plain AWARE at its native quiet strength (tol=6). Reference only.
  n1     -- staircase N=1 = full 1000-4000 stripe every frame, NO hop. The matched-
            tol (-6) baseline: "wideband, no hopping".
  n2..n5 -- key-driven hop across N sub-bands. THE TEST vs n1.

FAIRNESS CAVEAT (read before concluding): stock is quiet (PESQ~4.1), the staircase
configs are loud (tol=-6). Also hopping makes the mark quieter as N grows (fewer
bins/frame -> higher PESQ). So the ONLY fair "does hopping help" comparison is
n1 vs n2..n5 (same tol), and even there PESQ differs -- clean PESQ is printed per
config so you can weight it. A fully imperceptibility-MATCHED sweep (equal PESQ per
config, Zhang protocol) is the follow-up; this is the exploratory first cut.

Detection: STOCK AWARE detector. detected = conf>=0.5; bit_acc is also reported
(thesis "recovered" threshold = 0.8).

Run on a GPU node in wmcompare (AWARE + Emilia + ffmpeg/encodec on Great Lakes):
    conda activate wmcompare
    python -m fsss.exp_v6_attacks_nsweep
    python -m fsss.exp_v6_attacks_nsweep --nlist 1,2,3 --attacks mp3,lowpass,gaussian_noise
    python -m fsss.exp_v6_attacks_nsweep --band 1000 4000 --tol -6
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
from aware.metrics.audio import PESQ

import vox_attacks
from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
DEFAULT_BAND = (1000, 4000)
DEFAULT_NLIST = [1, 2, 3, 4, 5]          # n1 = no-hop baseline; n2..n5 = hopping
DEFAULT_ATTACKS = ["mp3", "lowpass", "gaussian_noise", "dynamic_compression",
                   "echo", "time_stretch", "encodec"]
DET_CONF = 0.5                            # detector confidence threshold
DET_ACC = 0.8                             # thesis bit-accuracy "recovered" threshold
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_band(argv):
    if "--band" in argv:
        i = argv.index("--band")
        return (int(argv[i + 1]), int(argv[i + 2]))
    return DEFAULT_BAND


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def make_config(name, n_bands, embedder, key, band, tol, seglen):
    if n_bands == 0:                       # stock: plain AWARE, its own card band + tol
        return embedder
    st = StaircaseAWAREEmbedder.from_embedder(
        embedder, key=key, n_bands=n_bands, band_range=band,
        segment_mode="fixed", segment_len_s=seglen, tolerance_db=tol)
    st.embedding_bands = band
    return st


def main(argv):
    clip = get_arg(argv, "--clip", None)
    key = get_arg(argv, "--key", "thesis")
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    tol = get_arg(argv, "--tol", -6.0, float)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    band = get_band(argv)
    nlist = get_list(argv, "--nlist", DEFAULT_NLIST, int)
    attacks = get_list(argv, "--attacks", DEFAULT_ATTACKS, str)

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no Emilia clip found; pass --clip PATH")
            return
        clip = clips[0]

    audio = load_16k(clip)
    n_keep = int(round(dur * WORK_SR))
    audio = audio[:n_keep]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    print(f"clip     : {clip}")
    print(f"duration : {len(audio)/WORK_SR:.2f}s   payload: {bits.tolist()}")
    print(f"band     : {band[0]}-{band[1]} Hz   tol={tol}   seglen={seglen}   key='{key}'")
    print(f"configs  : stock + n{nlist}")
    print(f"attacks  : {attacks}")

    embedder, detector = load()
    pesq_metric = PESQ()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v6_attacks_nsweep.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["config", "n_bands", "pesq_clean", "attack", "param",
                     "strength_x", "bit_acc", "conf", "detected"])

    configs = [("stock", 0)] + [(f"n{n}", n) for n in nlist]
    summary = {}                            # name -> dict(det, total, accs, clean_conf, pesq)

    for name, n_bands in configs:
        print(f"\n########## {name} (band {band[0]}-{band[1]}) ##########")
        model = make_config(name, n_bands, embedder, key, band, tol, seglen)
        wm = embed_watermark(audio, sample_rate=WORK_SR, watermark_bits=bits, model=model)

        # clean baseline + PESQ
        pat, conf = detect_watermark(wm, WORK_SR, detector)
        acc = bit_acc(bits, pat)
        m = min(len(audio), len(wm))
        try:
            pesq = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
        except Exception:
            pesq = float("nan")
        print(f"  clean                 bit_acc={acc:.3f} conf={conf:.3f} PESQ={pesq:.3f}")
        writer.writerow([name, n_bands, round(pesq, 4), "clean", "", 0.0,
                         round(acc, 4), round(float(conf), 4), int(conf >= DET_CONF)])
        summary[name] = dict(det=0, total=0, accs=[], clean_conf=float(conf),
                             clean_acc=acc, pesq=pesq)

        # attack sweep
        for attack in attacks:
            grid = vox_attacks.VOX_GRID.get(attack, [])
            accs, dets, labels = [], [], []
            for label, param in grid:
                try:
                    wa = vox_attacks.apply(attack, param, wm.astype("float32"), WORK_SR)
                except Exception as e:
                    print(f"    {attack}/{label}: ERROR {type(e).__name__} {e}")
                    continue
                if wa is None:
                    continue
                pat, conf = detect_watermark(wa, WORK_SR, detector)
                a = bit_acc(bits, pat)
                det = int(conf >= DET_CONF)
                try:
                    sx = float(vox_attacks.strength_x(attack, param))
                except Exception:
                    sx = 0.0
                writer.writerow([name, n_bands, round(pesq, 4), attack, label,
                                 sx, round(a, 4), round(float(conf), 4), det])
                accs.append(a); dets.append(det); labels.append(label)
                summary[name]["total"] += 1
                summary[name]["det"] += det
                summary[name]["accs"].append(a)
            if labels:
                acc_str = " ".join(f"{l}:{a:.2f}" for l, a in zip(labels, accs))
                print(f"    {attack:20s} det {sum(dets)}/{len(dets)}  bitacc[{acc_str}]")
    fout.close()

    # ---- summary: survival per config ------------------------------------- #
    print("\n" + "=" * 72)
    print(f"{'config':7s} {'PESQ':>6s} {'clean_conf':>10s} {'det/total':>10s} "
          f"{'mean_acc':>9s} {'acc>=.8':>8s}")
    for name, _ in configs:
        s = summary[name]
        tot = max(1, s["total"])
        mean_acc = np.mean(s["accs"]) if s["accs"] else float("nan")
        frac_rec = np.mean([a >= DET_ACC for a in s["accs"]]) if s["accs"] else float("nan")
        print(f"{name:7s} {s['pesq']:6.2f} {s['clean_conf']:10.3f} "
              f"{s['det']:>4d}/{s['total']:<5d} {mean_acc:9.3f} {frac_rec:8.2f}")
    print("=" * 72)
    print(f"wrote {csv_path}")
    print("read: FAIR comparison = n1 (no-hop) vs n2..n5 (hop) at matched tol; higher "
          "survival at similar/better PESQ => hopping helps robustness. stock is a "
          "quiet reference (different tol). Next: imperceptibility-MATCHED sweep + more clips/keys.")


if __name__ == "__main__":
    main(sys.argv[1:])
