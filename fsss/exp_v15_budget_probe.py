"""
fsss/exp_v15_budget_probe.py -- WHY does N>=3 hopping fail? Budget wall, iteration
limit, or readout?

Every prior N-sweep measured the OUTCOME (conf / bit_acc after the fact). None
measured what the optimizer was actually doing while it embedded. This probe
opens that box. It runs no attacks and computes no PESQ -- it only instruments
the embedding optimization itself.

The mechanism under test: AWARE's perceptual budget is PER TF-BIN
(|delta_{f,u}| <= eta * M_{f,u}, eta = 10**(-tol/20)), with no global energy
constraint. Hopping writes only 1/N of the band per frame, so it gets ~1/N as
many writable cells at the SAME per-cell cap => ~1/N the injectable watermark
energy, for free, with nothing given back. If that is the failure, the optimizer
should be pinned against its box bounds with the bits still wrong.

Sweeps N in {1,2,3,4} x tol in {6,0,-6} over band 500-4000, librosa-anchored
segments throughout (so ONLY N differs; at N=1 the schedule is degenerate and
every frame gets the whole band = the no-hop control that makes the other
columns interpretable).

READING THE OUTPUT -- these are committed in advance:
  sat_frac ~1.0 + embed_ber > 0 + late_drop ~0
      => BUDGET WALL. The mark was never encoded. Fix = give the embedder a
         global energy target (scale eta with N) so every N injects equal energy.
  sat_frac low + embed_ber ~0
      => embedding SUCCEEDED; the bits are lost at DETECTION instead. The
         per-cell budget story is then wrong and the host-interference/readout
         story (unwritten bands feeding the frozen detector) is doing the work.
         Fix = the anchor+key-informed detector moves ahead of the budget fix.
  late_drop >> 0
      => ITERATION LIMITED, not budget limited. Cheapest fix is more iterations
         and both of the above diagnoses are overstated.
  delta_energy falling ~1/N at fixed tol
      => the mechanical claim underneath all of this, measured directly.

    conda activate wmcompare
    python -m fsss.exp_v15_budget_probe
    python -m fsss.exp_v15_budget_probe --nclips 5 --nlist 1,2,3,4,6 --tols 6,0,-6
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
DEFAULT_NLIST = [1, 2, 3, 4]
DEFAULT_TOLS = [6.0, 0.0, -6.0]
DEFAULT_BAND = (500, 4000)
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")

STAT_KEYS = ["n_vars", "n_active", "sat_frac", "embed_ber", "margin",
             "delta_energy", "final_loss", "loss_q25", "loss_q50", "loss_q75",
             "loss_q100", "late_drop"]


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
    seglen = get_arg(argv, "--seglen", 0.5, float)
    rate = get_arg(argv, "--rate", 3.5, float)
    key = get_arg(argv, "--key", "thesis")
    nlist = get_list(argv, "--nlist", DEFAULT_NLIST, int)
    tols = get_list(argv, "--tols", DEFAULT_TOLS, float)
    band = tuple(get_list(argv, "--band", list(DEFAULT_BAND), int))

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    embedder, detector = load()
    detector.embedding_bands = band

    print(f"clips : {len(audios)}   band: {band[0]}-{band[1]}   key: {key}")
    print(f"N     : {nlist}")
    print(f"tols  : {tols}   (lower = louder; stock AWARE = 6)")
    print("clean only -- no attacks, no PESQ. N=1 = no-hop control.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v15_budget_probe.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "n_bands", "tol", "conf", "bit_acc", "detected"] + STAT_KEYS)

    agg = {}   # (n_bands, tol) -> list of stat dicts (+ conf/acc)

    for ci, (clip, audio) in enumerate(zip(clips, audios)):
        print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
        for nb in nlist:
            for tol in tols:
                st = StaircaseAWAREEmbedder.from_embedder(
                    embedder, key=key, n_bands=nb, band_range=band,
                    segment_mode="librosa", anchor="librosa_flux", anchor_rate=rate,
                    segment_len_s=seglen, tolerance_db=tol)
                st.embedding_bands = band

                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=st)
                stats = dict(getattr(st, "last_stats", {}))
                pat, conf = detect_watermark(wm, WORK_SR, detector)
                acc = bit_acc(bits, pat)

                row = dict(stats)
                row["conf"] = float(conf)
                row["bit_acc"] = acc
                agg.setdefault((nb, tol), []).append(row)

                writer.writerow([ci, nb, tol, round(float(conf), 4), round(acc, 4),
                                 int(conf >= DET_CONF)] +
                                [stats.get(k, float("nan")) for k in STAT_KEYS])
                print(f"    N={nb} tol={tol:>5}  cells={stats.get('n_vars', 0):>6}  "
                      f"sat={stats.get('sat_frac', float('nan')):.3f}  "
                      f"embedBER={stats.get('embed_ber', float('nan')):.3f}  "
                      f"dE={stats.get('delta_energy', float('nan')):.4g}  "
                      f"lateDrop={stats.get('late_drop', float('nan')):.3f}  "
                      f"conf={float(conf):.3f}")
    fout.close()

    def mean_of(nb, tol, field):
        rows = agg.get((nb, tol), [])
        vals = [r[field] for r in rows
                if field in r and not np.isnan(float(r[field]))]
        return float(np.mean(vals)) if vals else float("nan")

    colw = 11

    def table(title, field, fmt="{:.3f}", note=""):
        hdr = f"  {'N':>3s}" + "".join(f"{('tol'+str(int(t))):>{colw}s}" for t in tols)
        print("\n" + "=" * max(len(hdr), len(title)))
        print(title)
        if note:
            print(f"  ({note})")
        print(hdr)
        for nb in nlist:
            line = f"  {nb:>3d}"
            for tol in tols:
                line += f"{fmt.format(mean_of(nb, tol, field)):>{colw}s}"
            print(line)

    table("WRITABLE CELLS", "n_vars", "{:.0f}",
          "should fall ~1/N -- confirms the mechanism is real")
    table("SATURATION FRACTION", "sat_frac",
          note="fraction of coeffs pinned to their budget bound; ~1.0 = out of room")
    table("EMBED-TIME BER", "embed_ber",
          note="BER against the frozen detector AT EMBED TIME; >0 = never encoded")
    table("MARGIN", "margin",
          note="mean signed detector output vs target; higher = cleaner readout")
    table("DELTA ENERGY", "delta_energy", "{:.4g}",
          note="watermark energy actually injected; expect ~1/N at fixed tol")
    table("LATE LOSS DROP", "late_drop",
          note="~0 = plateaued (budget/capacity); >>0 = still improving (needs iters)")
    table("CLEAN CONF", "conf", note="the outcome the earlier sweeps measured")
    table("CLEAN BIT-ACC", "bit_acc")

    # the one comparison the whole diagnosis rests on
    print("\n" + "=" * 60)
    print("ENERGY SCALING vs N=1 (at fixed tol) -- 1/N would mean the per-cell")
    print("budget is dividing the watermark, exactly as predicted:")
    for tol in tols:
        base = mean_of(nlist[0], tol, "delta_energy")
        parts = []
        for nb in nlist:
            e = mean_of(nb, tol, "delta_energy")
            parts.append(f"N{nb}={e/base:.3f}" if base and not np.isnan(base) else f"N{nb}=nan")
        print(f"  tol{int(tol):>3}: " + "  ".join(parts) + f"   (1/N = " +
              "  ".join(f"{1.0/nb:.3f}" for nb in nlist) + ")")

    print(f"\nwrote {csv_path}")
    print("\nVERDICT KEY (committed before the run):")
    print("  sat~1.0 + embedBER>0 + lateDrop~0 -> BUDGET WALL; fix = energy-matched eta.")
    print("  sat low  + embedBER~0            -> embedding fine, READOUT is the problem;")
    print("                                      informed detector moves ahead of budget fix.")
    print("  lateDrop >> 0                    -> ITERATION limited; just run longer.")


if __name__ == "__main__":
    main(sys.argv[1:])
