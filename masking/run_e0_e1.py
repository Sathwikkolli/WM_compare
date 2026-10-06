"""
masking/run_e0_e1.py -- one Slurm array task = one clip.

E0   original AWARE baseline: embed a random 20-bit payload, measure quality
     (PESQ, STOI, SNR, SI-SNR, ODG if PEAQ is installed), embed runtime, and
     bit accuracy + confidence under the starter attacks.
E1a  budget usage: how much of AWARE's per-bin budget |delta| <= M*10^(-tol/20)
     the optimiser actually spends, overall, quiet vs loud bins, per frequency
     and per frame.
E1b  where the watermark is readable: slide 1 s windows over the watermarked
     audio (clean, mp3_64, awgn_25), decode each window on its own, and relate
     window bit accuracy to the window's loudness and speech content.
     (AWARE full_length carries ONE 20-bit message for the whole clip, so there
     is no native per-frame bit -- per-window decoding is the closest analogue
     of Wang et al.'s "which frames fail".)

Usage:
    python run_e0_e1.py --index 3          # or via SLURM_ARRAY_TASK_ID
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import common as C

WIN = C.SR + 128              # same chunk length AWARE's own segments mode uses
HOP = C.SR // 4               # 250 ms
E1B_CONDITIONS = ("clean", "mp3_64", "awgn_25")


def speech_fraction(seg):
    import webrtcvad
    vad = webrtcvad.Vad(3)
    pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16).tobytes()
    n = int(C.SR * 0.03) * 2
    frames = [pcm[i:i + n] for i in range(0, len(pcm) - n + 1, n)]
    return float(np.mean([vad.is_speech(f, C.SR) for f in frames])) if frames else 0.0


def band_db(seg, lo=1000, hi=4000):
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / C.SR)
    return float(10 * np.log10(spec[(f >= lo) & (f <= hi)].sum() / len(seg) + 1e-12))


def e1a_stats(cap, tol_db):
    init, fin, up = cap["initial"], cap["final"], cap["upper"]
    n_freq = len(cap["freq_indices"])
    delta = up - init                                   # = init * 10^(-tol/20)
    used = np.abs(fin - init)
    util = np.divide(used, delta, out=np.zeros_like(used), where=delta > 1e-12)
    U = util.reshape(n_freq, -1)                        # (freq_bin, frame)
    M = init.reshape(n_freq, -1)

    q25, q75 = np.percentile(init, [25, 75])
    quiet, loud = init <= q25, init >= q75
    sat = util > 0.95
    frame_db = 10 * np.log10(np.sum(M ** 2, axis=0) + 1e-12)
    frame_util = U.mean(axis=0)
    from scipy.stats import spearmanr
    rho = spearmanr(frame_db, frame_util).correlation if U.shape[1] > 2 else float("nan")

    stats = dict(
        n_coeffs=int(util.size), n_freq_bins=int(n_freq), n_frames=int(U.shape[1]),
        tolerance_db=tol_db,
        util_mean=float(util.mean()), util_median=float(np.median(util)),
        frac_util_gt_0p95=float(sat.mean()), frac_util_gt_0p5=float((util > 0.5).mean()),
        frac_util_lt_0p05=float((util < 0.05).mean()),
        quiet_util_mean=float(util[quiet].mean()), loud_util_mean=float(util[loud].mean()),
        quiet_frac_sat=float(sat[quiet].mean()), loud_frac_sat=float(sat[loud].mean()),
        sat_pushed_up=float(np.mean(fin[sat] > init[sat])) if sat.any() else float("nan"),
        spearman_frame_energy_vs_util=float(rho),
        rel_change_energy_db=float(10 * np.log10(np.sum((fin - init) ** 2) / (np.sum(init ** 2) + 1e-12))),
    )
    profiles = dict(util_per_freq=U.mean(axis=1), util_per_frame=frame_util,
                    frame_band_db=frame_db, sat_per_freq=(U > 0.95).mean(axis=1))
    return stats, profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    spec = json.load(open(C.CLIPS_JSON))
    clip = spec["clips"][a.index]
    cid = clip["clip_id"]
    os.makedirs(C.DATA, exist_ok=True)
    print(f"[{cid}] {clip['utt']}  ({clip['duration_s']} s)")

    load, embed_watermark, detect_watermark = C.import_aware()
    embedder, detector = load(name=C.AWARE_MODEL)
    cap = C.OptimizeCapture(embedder)

    y = C.load_16k(clip["path"])
    bits = np.random.RandomState(a.seed * 1000 + a.index).randint(0, 2, C.N_BITS)

    # ---------------- E0: embed + quality ----------------
    t0 = time.time()
    z = np.asarray(embed_watermark(y, C.SR, bits, embedder), dtype="float32")[:len(y)]
    embed_s = time.time() - t0
    C.write_wav(os.path.join(C.WORK, "orig", cid + ".wav"), y)
    C.write_wav(os.path.join(C.WORK, "wm", cid + ".wav"), z)

    q = C.quality(y, z)
    qrow = dict(clip_id=cid, utt=clip["utt"], duration_s=len(y) / C.SR,
                embed_seconds=round(embed_s, 2), num_iterations=embedder.num_iterations,
                tolerance_db=embedder.tolerance_db, payload="".join(map(str, bits)), **q)
    pd.DataFrame([qrow]).to_csv(os.path.join(C.DATA, f"e0_quality_{cid}.csv"), index=False)
    print(f"  embed {embed_s:.1f}s  PESQ {q['pesq']}  STOI {q['stoi']}  SNR {q['snr_db']:.2f}  ODG {q['odg']}")

    # ---------------- E0: attacks ----------------
    rows, attacked = [], {}
    for name, (fn, aligned) in C.ATTACKS.items():
        try:
            za = np.asarray(fn(z), dtype="float32")
            dec, conf = detect_watermark(za, C.SR, detector)[:2]
            acc = C.bit_acc(dec, bits)
            if aligned:
                attacked[name] = za[:len(z)]
        except Exception as e:
            print(f"  {name}: FAILED {e}")
            acc, conf = float("nan"), float("nan")
        rows.append(dict(clip_id=cid, attack=name, bit_acc=acc, ber=1 - acc, conf=float(conf)))
        print(f"  {name:<13s} bit_acc {acc:.3f}  conf {float(conf):.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(C.DATA, f"e0_attacks_{cid}.csv"), index=False)

    # ---------------- E1a: budget usage ----------------
    stats, prof = e1a_stats(cap.data, embedder.tolerance_db)
    stats["clip_id"] = cid
    json.dump(stats, open(os.path.join(C.DATA, f"e1a_{cid}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(C.DATA, f"e1a_{cid}.npz"),
                        initial=cap.data["initial"], final=cap.data["final"],
                        upper=cap.data["upper"], freq_indices=cap.data["freq_indices"], **prof)
    print(f"  E1a util mean {stats['util_mean']:.3f}  >0.95: {stats['frac_util_gt_0p95']:.3f}  "
          f"quiet {stats['quiet_util_mean']:.3f} / loud {stats['loud_util_mean']:.3f}")

    # ---------------- E1b: per-window readability ----------------
    wrows = []
    for cond in E1B_CONDITIONS:
        za = attacked.get(cond)
        if za is None:
            continue
        for s in range(0, len(za) - WIN + 1, HOP):
            seg_o, seg_w = y[s:s + WIN], za[s:s + WIN]
            try:
                dec, conf = detect_watermark(seg_w, C.SR, detector)[:2]
                acc = C.bit_acc(dec, bits)
            except Exception as e:
                print(f"  window {cond}@{s}: {e}")
                acc, conf = float("nan"), float("nan")
            wrows.append(dict(clip_id=cid, condition=cond, start_s=s / C.SR,
                              bit_acc=acc, conf=float(conf),
                              win_dbfs=C.dbfs(seg_o), win_band_db=band_db(seg_o),
                              speech_frac=speech_fraction(seg_o),
                              wm_change_db=C.dbfs(seg_w - seg_o) if cond == "clean" else None))
    pd.DataFrame(wrows).to_csv(os.path.join(C.DATA, f"e1b_{cid}.csv"), index=False)
    print(f"  E1b {len(wrows)} windows")
    print(f"[{cid}] done")


if __name__ == "__main__":
    main()
