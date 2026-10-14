"""
casp/ab_dial.py -- dial sweep: stock-AWARE tolerance vs CASP-shaped target, both
measured on the same ruler (CASP audibility units) and attacked with harder
attacks. One clip per task.

    AWARE family : flat eta = 10^(-tol/20), tol in {6, 9, 12, 15, 18} dB
    CASP  family : eta(band, block) solved from the stored probes at target T in
                   {1, 2, 4, 6, 8} audibility units. NO energy rescaling and NO
                   cap at 1 (that distorted the first A/B run); eta tops out at
                   the largest probe gain (0.25), reported as `frac_at_range_cap`.

Everything else is identical across the 9 embeds: same AWARE optimiser, same
frozen detector, same payload, same bounds code path (BudgetOverride, fixed
mode). 10 embeds per clip. For every embed we measure the audibility of the REAL optimised
watermark (not the random-sign probe) with the revised CASP, then detect after
7 attack conditions.

    python ab_dial.py --index 0            # sbatch array 0-4, clips_ab.json
    python ab_dial.py --mock               # plumbing test (fake AWARE, real CASP)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_embed_detect as AB                     # noqa: E402  (also loads threshold, masking.common)

TH, C, psy, SR = AB.TH, AB.C, AB.psy, AB.SR
RUN = os.environ.get("CASP_DIAL_RUN", "2026-10-14_casp-dial")
RESULTS = os.path.join(AB.ROOT, "results", RUN)

# CASP's family stops at eta = 0.25 (largest probe gain), so AWARE's dial runs from its
# default 6 dB down into that range; otherwise the equal-audibility comparison is empty.
AWARE_TOLS = [6.0, 9.0, 12.0, 15.0, 18.0]
CASP_TARGETS = [1.0, 2.0, 4.0, 6.0, 8.0]

ATTACKS = {
    "clean":   lambda y: y,
    "awgn_10": lambda y: C._awgn(y, 10),
    "awgn_5":  lambda y: C._awgn(y, 5),
    "lp_2500": lambda y: C._lowpass(y, 2500),
    "lp_2000": lambda y: C._lowpass(y, 2000),
    "mp3_16":  lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "16k"], "mp3"),
    "opus_12": lambda y: C._ffmpeg_roundtrip(y, ["-c:a", "libopus", "-b:a", "12k"], "opus"),
}


def load_probe(cid, y_n, threads):
    p = os.path.join(AB.CASP_RESULTS, cid, "probe.npz")
    if not os.path.exists(p):
        AB.casp_eta_map(cid, y_n, threads)       # probes and saves D, etas, bands, blocks, d_jnd
    z = np.load(p)
    return (z["D"], z["etas"], [tuple(b) for b in z["bands"]],
            [tuple(b) for b in z["blocks"]], float(z["d_jnd"]))


def mock_probe(shape, U):
    rng = np.random.RandomState(0)
    bands, _ = TH.band_bins(8)
    blocks = TH.block_frames(U, 16)
    etas = TH.ETAS
    base = rng.uniform(600, 2600, size=(len(blocks), len(bands), 1))
    D = base * (etas[None, None, :] / 0.25) ** 0.7
    return D.astype("float32"), etas, bands, blocks, 708.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-s", type=float, default=AB.MAX_S)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.mock:
        import run_casp
        cid, y = "mock", run_casp.synth()[: int(a.max_s * SR)]
        load, embed_watermark, detect_watermark = AB.mock_aware()
    else:
        clip = json.load(open(AB.CLIPS_AB))["clips"][a.index]
        cid = clip["clip_id"]
        y = C.load_16k(clip["path"])[: int(a.max_s * SR)]
    out = a.out or os.path.join(RESULTS, cid)
    os.makedirs(out, exist_ok=True)
    print(f"[{cid}] {len(y) / SR:.1f}s")
    t0 = time.time()

    y_n = (y / max(float(np.max(np.abs(y))), 1e-9)).astype("float32")
    X_host = psy.stft(y_n)
    host_M = np.abs(X_host)
    F, U = X_host.shape
    if a.mock:
        D, etas, bands, blocks, d_jnd = mock_probe(X_host.shape, U)
    else:
        D, etas, bands, blocks, d_jnd = load_probe(cid, y_n, a.threads)
    n_bands = len(bands)

    # ---- budgets ---------------------------------------------------------
    arms = []
    for tol in AWARE_TOLS:
        arms.append(dict(family="AWARE", dial=tol, name=f"AWARE tol {tol:g}",
                         eta_fu=np.full((F, U), 10 ** (-tol / 20), np.float32), at_cap=0.0))
    for T in CASP_TARGETS:
        eta_kb, flag = TH.solve_eta(D, etas, T * d_jnd / np.sqrt(n_bands))
        arms.append(dict(family="CASP", dial=T, name=f"CASP T {T:g}",
                         eta_fu=TH.eta_map(eta_kb, bands, blocks, (F, U)),
                         at_cap=float(np.mean(flag == 1))))

    if not a.mock:
        load, embed_watermark, detect_watermark = C.import_aware()
    embedder, detector = load(name=C.AWARE_MODEL)
    ov = AB.BudgetOverride(embedder)
    bits = np.random.RandomState(a.seed * 1000 + a.index).randint(0, 2, C.N_BITS)

    # ---- embed -----------------------------------------------------------
    for arm in arms:
        ov.enable(arm["eta_fu"], host_M, fixed=True)
        t1 = time.time()
        arm["z"] = np.asarray(embed_watermark(y, SR, bits, embedder), "float32")[: len(y)]
        arm["embed_s"] = time.time() - t1
        arm["budget"] = dict(ov.info)
        print(f"  {arm['name']:<14} embed {arm['embed_s']:.0f}s  eta_med {ov.info['eta_casp_median']:.3f}  "
              f"E/E(6dB) {ov.info['E_ratio']:.3f}  at_cap {arm['at_cap']:.2f}")
    ov.disable()

    # ---- quality + CASP audibility of the real watermark -----------------
    for arm in arms:
        q = C.quality(y, arm["z"])
        arm["quality"] = {k: q.get(k) for k in ("pesq", "stoi", "snr_db", "si_snr_db")}
    from casp_front import CASP, spl_scale
    casp = CASP(fs=SR, threads=a.threads)
    x_pa, gain = spl_scale(y_n, AB.LEVEL_DB)
    scale_n = 1.0 / max(float(np.max(np.abs(y))), 1e-9)
    # AWARE's output can be a few samples shorter than the input (iSTFT framing);
    # measure everything on the common prefix so host and marked signals line up.
    L = min([len(y)] + [len(arm["z"]) for arm in arms])
    if L < len(y):
        print(f"  note: shortest AWARE output is {len(y) - L} samples shorter than the input; "
              f"measuring audibility on the first {L}")
    x_pa_m = x_pa[:L]
    sig = [arm["z"][:L] * scale_n * gain for arm in arms]
    CH = 3                                           # signals per CASP batch (memory)
    for i in range(0, len(arms), CH):
        res = TH.audibility_blocks(casp, x_pa_m, sig[i:i + CH], blocks, d_jnd)
        for arm, (A_blk, A_tot) in zip(arms[i:i + CH], res):
            arm["A_mean"], arm["A_p90"], arm["A_max"] = float(A_blk.mean()), float(np.percentile(A_blk, 90)), float(A_blk.max())
            arm["A_blocks"] = A_blk
    for arm in arms:
        print(f"  {arm['name']:<14} PESQ {arm['quality']['pesq']}  A_mean {arm['A_mean']:.2f}  A_max {arm['A_max']:.2f}")
    print(f"  measured audibility [{time.time() - t0:.0f}s]")

    # ---- attacks + detection ---------------------------------------------
    rows = []
    for att, fn in ATTACKS.items():
        for arm in arms + [dict(family="none", dial=0.0, name="none", z=y)]:
            kind = "neg" if arm["family"] == "none" else "wm"
            try:
                za = np.asarray(fn(arm["z"]), "float32")
                dec, conf = detect_watermark(za, SR, detector)[:2]
                acc = C.bit_acc(dec, bits)
            except Exception as e:
                print(f"  {arm['name']}/{att}: FAILED {e}")
                acc, conf = float("nan"), float("nan")
            rows.append(dict(clip_id=cid, kind=kind, family=arm["family"], dial=arm["dial"],
                             attack=att, bit_acc=acc, conf=float(conf), detected=bool(conf >= AB.DETECT_CONF)))
        sub = [r for r in rows[-(len(arms) + 1):] if r["kind"] == "wm"]
        print("  " + f"{att:<8}" + " ".join(f"{r['bit_acc']:.2f}" for r in sub))
    pd.DataFrame(rows).to_csv(os.path.join(out, "dial_rows.csv"), index=False)

    meta = dict(clip_id=cid, payload="".join(map(str, bits)), d_jnd=d_jnd, n_bands=n_bands,
                level_db=AB.LEVEL_DB,
                arms=[dict(name=r["name"], family=r["family"], dial=r["dial"], quality=r["quality"],
                           A_mean=r["A_mean"], A_p90=r["A_p90"], A_max=r["A_max"],
                           frac_at_range_cap=r["at_cap"], embed_s=r["embed_s"], budget=r["budget"])
                      for r in arms])
    json.dump(meta, open(os.path.join(out, "dial_meta.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "audibility_blocks.npz"),
                        blocks=np.array(blocks), **{r["name"].replace(" ", "_"): r["A_blocks"] for r in arms})

    # spectrograms: AWARE 6 dB vs the CASP point closest to it in audibility
    ref = next(r for r in arms if r["family"] == "AWARE" and r["dial"] == 6.0)
    near = min((r for r in arms if r["family"] == "CASP"), key=lambda r: abs(r["A_mean"] - ref["A_mean"]))
    import soundfile as sf
    for nm, v in [("original", y), ("aware_tol6", ref["z"]), (near["name"].replace(" ", "_"), near["z"])]:
        sf.write(os.path.join(out, f"{nm}.wav"), np.clip(v, -1, 1), SR, subtype="PCM_16")
    Ls = min(len(y), len(ref["z"]), len(near["z"]))
    AB.spectrogram_figure(os.path.join(out, "spectrograms.png"), f"{cid}  (AWARE tol 6 vs {near['name']})",
                          y[:Ls], ref["z"][:Ls], near["z"][:Ls])
    print(f"done [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()
