"""
casp/sb_run.py -- single-bit watermark: four budget arms x strength grid x attacks, one clip.

    python sb_run.py --index 0            # clip from clips_ab.json (sbatch array 0-4)
    python sb_run.py --synth --quick      # local plumbing test
    python sb_run.py --wav some.wav       # any wav (fake CASP shape unless a probe exists)

Arms (all energy-matched at the same strength g; g = 1 equals AWARE's 6 dB energy):
    aware_shape  amplitude proportional to |X| (AWARE's self-masking shape)
    casp         CASP tolerance per (band, 256 ms block) x |X|   (target 1 unit)
    johnston     textbook masking threshold from masking/psy.py (Phase 1)
    flat_abs     same amplitude in every cell (control)

For each arm and g: embed one bit, quantise to 16-bit, quality vs the host, then
detect after 10 attack conditions with a key-only correlation detector whose null
is estimated from wrong keys on the same file. Unwatermarked hosts go through the
same attacks for false positives.
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
import sb_embed as SB                        # noqa: E402
import ab_embed_detect as AB                 # noqa: E402  (masking.common, clips, spectrogram)
import ab_dial as AD                         # noqa: E402  (probe loading)

TH, C, psy, SR = SB.TH, AB.C, SB.psy, SB.SR
RUN = os.environ.get("CASP_SB_RUN", "2026-10-14_casp-sb")
RESULTS = os.path.join(AB.ROOT, "results", RUN)

GRID = [round(2 ** (-k / 2), 5) for k in range(15)]          # 1 ... 0.0078 (0 to -42 dB energy)
ATTACKS = {
    "clean":   lambda y: y,
    "awgn_20": lambda y: C._awgn(y, 20),
    "awgn_10": lambda y: C._awgn(y, 10),
    "awgn_5":  lambda y: C._awgn(y, 5),
    "lp_3500": lambda y: C._lowpass(y, 3500),
    "lp_2500": lambda y: C._lowpass(y, 2500),
    "lp_2000": lambda y: C._lowpass(y, 2000),
    "mp3_32":  lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    "mp3_16":  lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "16k"], "mp3"),
    "opus_12": lambda y: C._ffmpeg_roundtrip(y, ["-c:a", "libopus", "-b:a", "12k"], "opus"),
}


def load_casp_eta(cid, y_n, M, mock, threads):
    """-> eta_fu (F,U) for target 1, natural-level g for several targets."""
    F, U = M.shape
    if mock:
        D, etas, bands, blocks, d_jnd = AD.mock_probe(M.shape, U)
    else:
        D, etas, bands, blocks, d_jnd = AD.load_probe(cid, y_n, threads)
    out, nat = None, {}
    E_ref = SB.reference_energy(M)
    for T in (1.0, 2.0, 4.0, 8.0):
        eta_kb, _ = TH.solve_eta(D, etas, T * d_jnd / np.sqrt(len(bands)))
        eta_fu = TH.eta_map(eta_kb, bands, blocks, (F, U))
        fi = SB.band_idx()
        nat[f"T{T:g}"] = float(np.sqrt(np.sum((eta_fu[fi] * M[fi]) ** 2) / E_ref))
        if T == 1.0:
            out = eta_fu
    return out, nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="also report the oracle-weighted statistic")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--max-s", type=float, default=AB.MAX_S)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mock = a.synth or a.wav is not None
    if a.synth:
        import run_casp
        cid, y = "synth", run_casp.synth()
    elif a.wav:
        import librosa
        cid = os.path.splitext(os.path.basename(a.wav))[0]
        y, _ = librosa.load(a.wav, sr=SR, mono=True)
    else:
        clip = json.load(open(AB.CLIPS_AB))["clips"][a.index]
        cid = clip["clip_id"]
        y = C.load_16k(clip["path"])
    y = y[: int((2.0 if a.quick else a.max_s) * SR)].astype("float32")
    y_n = (y / max(float(np.max(np.abs(y))), 1e-9)).astype("float32")
    n = len(y_n)
    out = a.out or os.path.join(RESULTS, cid)
    os.makedirs(out, exist_ok=True)
    print(f"[{cid}] {n / SR:.1f}s  grid {len(GRID)}  arms {SB.ARMS}")
    t0 = time.time()

    host = SB.to_pcm16(y_n)
    X = psy.stft(y_n)
    M = np.abs(X)
    E_ref = SB.reference_energy(M)
    eta_casp, nat_g = load_casp_eta(cid, y_n, M, mock, a.threads)
    print(f"  CASP natural level (g): {{{', '.join(f'{k}: {v:.3f}' for k, v in nat_g.items())}}}")

    key = 1000 + 17 * a.index + a.seed
    bit = int(np.random.RandomState(a.seed * 1000 + a.index).choice([-1, 1]))
    fi = SB.band_idx()
    attacks = ATTACKS if not a.quick else {k: ATTACKS[k] for k in ("clean", "awgn_10", "lp_2500", "mp3_16")}
    grid = GRID if not a.quick else GRID[::3]

    rows, arm_info = [], {}
    # ---- negatives: unwatermarked host, true key ---------------------------
    for att, fn in attacks.items():
        d = SB.detect(fn(host), key, n, a.n_null)
        rows.append(dict(clip_id=cid, kind="neg", arm="none", g=0.0, attack=att, pesq=np.nan,
                         stoi=np.nan, snr_db=np.nan, z_std=d["z_std"], detected=d["detected"], bit_ok=np.nan))
    wavs = {}
    for arm in SB.ARMS:
        W = SB.shape_map(arm, M, eta_casp)
        A, info = SB.amplitude_map(W, M, E_ref)
        arm_info[arm] = info
        print(f"  arm {arm:<12} E/E(6dB) at g=1: {info['E_ratio']:.3f}  capped {info['frac_capped']:.2f}")
        for g in grid:
            z = SB.to_pcm16(SB.embed(X, A, key, bit, g, n))
            q = C.quality(host, z)
            if abs(g - 0.125) < 1e-3 and arm in ("aware_shape", "casp"):
                wavs[arm] = z
            w = (g * A[fi] / np.maximum(M[fi], 1e-8)) if a.oracle else None
            for att, fn in attacks.items():
                sig = fn(z)
                d = SB.detect(sig, key, n, a.n_null)
                row = dict(clip_id=cid, kind="wm", arm=arm, g=g, attack=att, pesq=q.get("pesq"),
                           stoi=q.get("stoi"), snr_db=q.get("snr_db"), z_std=d["z_std"],
                           detected=d["detected"], bit_ok=bool(d["bit_hat"] == bit))
                if a.oracle:
                    do = SB.detect(sig, key, n, a.n_null, weights=w)
                    row.update(z_oracle=do["z_std"], detected_oracle=do["detected"])
                rows.append(row)
        sub = [r for r in rows if r["kind"] == "wm" and r["arm"] == arm and r["attack"] == "clean"]
        print("   clean z_std by g:", " ".join(f"{r['z_std']:.1f}" for r in sub), f" [{time.time() - t0:.0f}s]")

    pd.DataFrame(rows).to_csv(os.path.join(out, "sb_rows.csv"), index=False)
    json.dump(dict(clip_id=cid, key=key, bit=bit, E_ref=E_ref, z_detect=SB.Z_DETECT, grid=grid,
                   arms=arm_info, casp_natural_g=nat_g, n=n, fake_casp=bool(mock)),
              open(os.path.join(out, "sb_meta.json"), "w"), indent=1)

    if "aware_shape" in wavs and "casp" in wavs:
        import soundfile as sf
        for nm, v in [("original", host), ("aware_shape_g0125", wavs["aware_shape"]), ("casp_g0125", wavs["casp"])]:
            sf.write(os.path.join(out, f"{nm}.wav"), v, SR, subtype="PCM_16")
        AB.spectrogram_figure(os.path.join(out, "spectrograms.png"),
                              f"{cid}: aware_shape vs casp at g = 0.125 (-18 dB re AWARE-6dB energy)",
                              host, wavs["aware_shape"], wavs["casp"])
    print(f"done [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()
