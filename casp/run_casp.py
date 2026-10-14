"""
casp/run_casp.py -- CASP masking-threshold experiment on ONE clip.

    python run_casp.py --index 0              # clip from casp/clips2.json
    python run_casp.py --synth --quick        # local smoke test, no data needed

Steps: calibrate D_JND -> probe every band x block -> solve tolerance maps for
several audibility targets -> verify with whole-signal runs against AWARE's
fixed tolerance -> figures. Everything lands in results/<RUN>/<clip_id>/.
No AWARE embedding happens here; perturbations are random-sign stand-ins with
AWARE's magnitude shape (same convention as masking Phase 1).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "masking"))

import threshold as TH                      # noqa: E402
import validate as VAL                      # noqa: E402
from casp_front import CASP, spl_scale      # noqa: E402

AWARE_REF_TOL = 6.0
RUN = os.environ.get("CASP_RUN", "2026-10-14_casp-threshold")
RESULTS = os.path.join(ROOT, "results", RUN)
CLIPS2 = os.path.join(HERE, "clips2.json")


def synth(sr=16000, dur=4.0, seed=1):
    """Speech-like test signal: harmonic stack, syllable-rate AM, two pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(h * ph) / h for h in range(1, 30))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    gate = np.ones_like(t)
    gate[(t > 1.3) & (t < 1.9)] = 0.01
    gate[t > 3.5] = 0.01
    x = x * env * gate + 0.002 * rng.standard_normal(len(t))
    return (x / np.max(np.abs(x))).astype(np.float32)


def quality_safe(ref, deg):
    try:
        import common as C
        return C.quality(ref, deg)
    except Exception as e:                   # pesq/stoi missing locally
        return dict(pesq=None, stoi=None, snr_db=float(10 * np.log10(
            np.sum(ref ** 2) / (np.sum((deg - ref) ** 2) + 1e-12))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--level", type=float, default=65.0, help="dB SPL of clip RMS")
    ap.add_argument("--max-s", type=float, default=5.0)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--block", type=int, default=16, help="STFT frames per block (16 = 256 ms)")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--jnd-db", type=float, default=VAL.JND_DB)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    SR = TH.SR
    if a.synth:
        cid, raw = "synth", synth()
    else:
        import common as C
        clip = json.load(open(CLIPS2))["clips"][a.index]
        cid = clip["clip_id"]
        raw = C.load_16k(clip["path"])
        print(f"[{cid}] {clip.get('utt')}  role={clip.get('role')}")
    raw = raw[: int(a.max_s * SR)]
    if a.quick:
        raw = raw[: 2 * SR]
    x_n = (raw / max(np.max(np.abs(raw)), 1e-9)).astype(np.float32)   # max|x| = 1 (AWARE grid units)
    out = a.out or os.path.join(RESULTS, cid + ("_level%d" % a.level if a.level != 65.0 else ""))
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    casp = CASP(fs=SR, threads=a.threads)
    d_jnd = VAL.calibrate(casp, a.jnd_db)
    print(f"channels {len(casp.fc)}  D_JND({a.jnd_db} dB @60 dB SPL) = {d_jnd:.4f}  "
          f"[{time.time() - t0:.0f}s]")

    x_pa, gain = spl_scale(x_n, a.level)
    n_bands = 4 if a.quick else a.bands
    etas = TH.ETAS
    g = TH.probe_grid(casp, x_pa, n_bands=n_bands, block=a.block, etas=etas, log=print)
    print(f"probing done [{time.time() - t0:.0f}s]")

    X = TH.psy.stft(x_n)
    M = np.abs(X)
    F, U = X.shape
    signs = TH.signs_like(X.shape, 0)
    bands, blocks = [tuple(b) for b in g["bands"]], [tuple(b) for b in g["blocks"]]

    maps = {}
    for T in a.targets:
        d_t = T * d_jnd / np.sqrt(n_bands)
        eta_kb, flag = TH.solve_eta(g["D"], g["etas"], d_t)
        maps[T] = dict(eta_kb=eta_kb, flag=flag, eta_fu=TH.eta_map(eta_kb, bands, blocks, (F, U)))
    np.savez_compressed(
        os.path.join(out, "probe.npz"), D=g["D"], etas=g["etas"], bands=g["bands"], hz=g["hz"],
        blocks=g["blocks"], d_jnd=d_jnd, level=a.level, targets=np.array(a.targets),
        **{f"eta_kb_{T}": maps[T]["eta_kb"] for T in a.targets},
        **{f"flag_{T}": maps[T]["flag"] for T in a.targets})

    # ---- verification: whole-signal runs, same random signs for every map ----
    aware_tols = [3.0, 6.0, 9.0]
    inband = np.zeros((F, U), np.float32)
    inband[bands[0][0]:bands[-1][1], :] = 1.0
    specs = [(f"AWARE tol {t:g} dB", inband * 10 ** (-t / 20)) for t in aware_tols]
    specs += [(f"CASP target {T:g}", maps[T]["eta_fu"]) for T in a.targets]
    # iso-energy: CASP SHAPE (relative tolerance across band/time) rescaled to the
    # in-band energy of AWARE 6 dB. Independent of the absolute JND calibration, so
    # this is the fair "is CASP's shape better than AWARE's flat rule" comparison.
    E_aw = float(np.sum((inband * 10 ** (-AWARE_REF_TOL / 20) * M) ** 2))
    for T in a.targets:
        eta = maps[T]["eta_fu"]
        k = np.sqrt(E_aw / max(float(np.sum((eta * M) ** 2)), 1e-20))
        specs.append((f"CASP shape {T:g} @AWARE-6dB energy", np.minimum(eta * k, 1.0)))
    ys = [x_n + TH.perturbation(X, eta_fu, signs, len(x_n)) for _, eta_fu in specs]
    aud = TH.audibility_blocks(casp, x_pa, [y * gain for y in ys], blocks, d_jnd)
    print(f"verification CASP runs done [{time.time() - t0:.0f}s]")
    conds, A_by = [], {}
    for (name, eta_fu), y_n, (A_blk, A_tot) in zip(specs, ys, aud):
        q = quality_safe(x_n, y_n)
        used = np.abs(eta_fu) * M
        r = dict(name=name, A_total=A_tot, A_block_mean=float(A_blk.mean()),
                 A_block_max=float(A_blk.max()), frac_blocks_gt1=float((A_blk > 1).mean()),
                 pesq=q.get("pesq"), stoi=q.get("stoi"), snr_db=q.get("snr_db"),
                 energy_db=float(10 * np.log10(np.sum(used ** 2) + 1e-20)),
                 mean_tol_db=float(np.mean(TH.tol_db(eta_fu[eta_fu > 0]))))
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        conds.append(r)
        A_by[name] = A_blk
    json.dump(dict(clip=cid, level_db=a.level, d_jnd=d_jnd, jnd_db=a.jnd_db, n_bands=n_bands,
                   block_frames=a.block, conditions=conds, seconds=time.time() - t0),
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "audibility.npz"), blocks=np.array(blocks),
                        **{k.replace(" ", "_"): v for k, v in A_by.items()})

    import plots
    plots.make_all(out, cid, x_n, X, bands, blocks, g, maps, A_by, conds, d_jnd, a.level)
    print(f"done [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()
