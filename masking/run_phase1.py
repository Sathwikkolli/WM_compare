"""
masking/run_phase1.py -- Phase 1 (E2-E6) for one clip: compare the SHAPE of
AWARE's budget with a masking-threshold budget, without the AWARE embedder.

Stand-in for AWARE's watermark: E1a showed AWARE pushes ~90 % of 1-4 kHz bins to
exactly +-budget with ~50/50 signs, so the watermark is (almost) signs * budget.
Here the signs are random (seeded), applied to |X| with the phase kept, exactly
the way AWARE edits the spectrogram (psy.apply_change).

Shapes:
  aware  -- |X| * 10^(-6/20)                  (AWARE's rule, gain 1)
  mask   -- psy.mask_budget(|X|)              (Johnston-style threshold)
  flat   -- constant per bin                  (control: no shaping at all)

E2  budget maps: where and by how much the two rules disagree
E3  equal energy: same total change, which shape sounds better?
E4  equal quality: how much more energy does the mask shape carry at AWARE's PESQ?
E5  tonal vs noisy frames: E3 restricted to each frame type
E6  codec survival: how much of each shape's change is still there after MP3/Opus

Usage:
    python run_phase1.py --index 3      # or via SLURM_ARRAY_TASK_ID
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import common as C
import psy

TOL_DB = 6.0
CODECS = {
    "mp3_64":  (["-codec:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    "mp3_32":  (["-codec:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    "opus_32": (["-c:a", "libopus", "-b:a", "32k"], "opus"),
    "opus_16": (["-c:a", "libopus", "-b:a", "16k"], "opus"),
}
RUN = os.environ.get("PHASE1_RUN", "2026-10-06_masking-phase1")
DATA = os.path.join(C.ROOT, "results", RUN, "data")


def energy(d):
    return float(np.sum(d[psy.band_mask()] ** 2))


def db(r):
    return float(10 * np.log10(max(r, 1e-30)))


def pesq_of(ref, deg):
    from pesq import pesq
    try:
        return float(pesq(C.SR, ref, deg, "wb"))
    except Exception:
        return float("nan")


def equal_energy_gain(B, A, frames=None):
    bm = psy.band_mask()
    a, b = A[bm], B[bm]
    if frames is not None:
        a, b = a[:, frames], b[:, frames]
    return float(np.sqrt(np.sum(a ** 2) / max(np.sum(b ** 2), 1e-30)))


def e2_stats(M, A, B, alpha):
    bm = psy.band_mask()
    r = 20 * np.log10(np.maximum(B[bm], 1e-12) / np.maximum(A[bm], 1e-12))
    fe = 10 * np.log10(np.sum(M[bm] ** 2, axis=0) + 1e-12)
    lo, hi = np.percentile(fe, [33, 67])
    z = psy.bark(psy.freqs())
    a_bin = alpha[np.floor(z).astype(int)][bm]           # band tonality per bin
    tonal, noisy = a_bin >= 0.5, a_bin < 0.5
    return dict(
        ratio_db_median=float(np.median(r)), ratio_db_p10=float(np.percentile(r, 10)),
        ratio_db_p90=float(np.percentile(r, 90)), frac_mask_gt_aware=float((r > 0).mean()),
        energy_ratio_db=db(np.sum(B[bm] ** 2) / np.sum(A[bm] ** 2)),
        ratio_db_quiet_frames=float(np.median(r[:, fe <= lo])),
        ratio_db_loud_frames=float(np.median(r[:, fe >= hi])),
        frac_bins_tonal=float(tonal.mean()),
        ratio_db_tonal=float(np.median(r[tonal])) if tonal.any() else None,
        ratio_db_noisy=float(np.median(r[noisy])) if noisy.any() else None,
        # AWARE's change is ~ +-A (E1a), so A > B means AWARE's watermark sits above
        # the modelled masking threshold in that bin (noise-to-mask ratio > 0 dB)
        aware_above_mask=float((r < 0).mean()),
        aware_above_mask_tonal=float((r[tonal] < 0).mean()) if tonal.any() else None,
        aware_above_mask_noisy=float((r[noisy] < 0).mean()) if noisy.any() else None,
    )


def change_survival(x, y, codec):
    """How much of the embedded change (y - x, seen in |STFT|) survives a codec.
    Compares |X(codec(y))| - |X(codec(x))| with |X(y)| - |X(x)| in 1-4 kHz."""
    args, ext = CODECS[codec]
    bm = psy.band_mask()
    Mx, My = np.abs(psy.stft(x))[bm], np.abs(psy.stft(y))[bm]
    cx = C._ffmpeg_roundtrip(x, args, ext)
    cy = C._ffmpeg_roundtrip(y, args, ext)
    n = min(len(cx), len(cy), len(x))
    Mcx, Mcy = np.abs(psy.stft(cx[:n]))[bm], np.abs(psy.stft(cy[:n]))[bm]
    u = min(Mx.shape[1], Mcx.shape[1])
    d0, d1 = (My - Mx)[:, :u].ravel(), (Mcy - Mcx)[:, :u].ravel()
    nz = np.abs(d0) > 1e-9
    return dict(
        corr=float(np.corrcoef(d0, d1)[0, 1]),
        kept_db=db(np.sum(d1 ** 2) / max(np.sum(d0 ** 2), 1e-30)),
        sign_agree=float(np.mean(np.sign(d0[nz]) == np.sign(d1[nz]))),
        sign_agree_wtd=float(np.sum(np.abs(d0[nz]) * (np.sign(d0[nz]) == np.sign(d1[nz])))
                             / np.sum(np.abs(d0[nz]))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    a = ap.parse_args()
    clip = json.load(open(C.CLIPS_JSON))["clips"][a.index]
    cid = clip["clip_id"]
    os.makedirs(DATA, exist_ok=True)
    print(f"[{cid}] {clip['utt']}")

    raw = C.load_16k(clip["path"])
    peak = float(np.max(np.abs(raw)))
    x = raw / peak                                     # AWARE's WaveformNormalizer
    M = np.abs(psy.stft(x))
    A = psy.aware_budget(M, TOL_DB)
    B, parts = psy.mask_budget(M, return_parts=True)
    bm = psy.band_mask()
    Fl = np.full_like(M, np.sqrt(np.mean(A[bm] ** 2)))
    signs = np.sign(np.random.RandomState(1000 + a.index).randn(*M.shape))

    def emb(Bud, g, frames=None):
        y, d = psy.apply_change(x, Bud, signs, g, frames)
        return y, d

    def q(y):
        r = C.quality(x * peak, y * peak)
        return {k: r[k] for k in ("pesq", "stoi", "snr_db")}

    # ---------- E2 ----------
    s2 = e2_stats(M, A, B, parts["alpha"])
    s2["clip_id"] = cid
    json.dump(s2, open(os.path.join(DATA, f"e2_{cid}.json"), "w"), indent=2)
    if a.index < 3:
        np.savez_compressed(os.path.join(DATA, f"e2_maps_{cid}.npz"),
                            M=M[bm].astype("float32"), A=A[bm].astype("float32"),
                            B=B[bm].astype("float32"), freqs=psy.freqs()[bm])
    print(f"  E2 mask/aware median {s2['ratio_db_median']:+.1f} dB, energy {s2['energy_ratio_db']:+.1f} dB, "
          f"AWARE above mask in {s2['aware_above_mask']:.0%} of bins")

    # ---------- E3: equal energy ----------
    ya, da = emb(A, 1.0)
    gm = equal_energy_gain(B, A)
    gf = equal_energy_gain(Fl, A)
    ym, dm = emb(B, gm)
    yf, df = emb(Fl, gf)
    rows3 = []
    for shape, y, d, g in (("aware", ya, da, 1.0), ("mask", ym, dm, gm), ("flat", yf, df, gf)):
        rows3.append(dict(clip_id=cid, shape=shape, gain_db=20 * np.log10(g),
                          applied_energy=energy(d), **q(y)))
    pd.DataFrame(rows3).to_csv(os.path.join(DATA, f"e3_{cid}.csv"), index=False)
    print("  E3 " + "  ".join(f"{r['shape']} PESQ {r['pesq']:.3f}" for r in rows3))

    # ---------- E4: equal quality (mask scaled to AWARE's PESQ) ----------
    target = rows3[0]["pesq"]
    lo, hi = -20.0, 20.0                                # gain on B, dB
    for _ in range(12):
        mid = (lo + hi) / 2
        p = pesq_of(x * peak, emb(B, 10 ** (mid / 20))[0] * peak)
        lo, hi = (mid, hi) if p > target else (lo, mid)
    g4_db = (lo + hi) / 2
    y4, d4 = emb(B, 10 ** (g4_db / 20))
    e4 = dict(clip_id=cid, target_pesq=target, mask_gain_db=g4_db,
              energy_gain_db=db(energy(d4) / energy(da)), **{f"mask_{k}": v for k, v in q(y4).items()})
    pd.DataFrame([e4]).to_csv(os.path.join(DATA, f"e4_{cid}.csv"), index=False)
    print(f"  E4 at PESQ {target:.3f}: mask carries {e4['energy_gain_db']:+.2f} dB vs AWARE")

    # ---------- E5: tonal vs noisy frames ----------
    z = psy.bark(psy.freqs())
    bands = np.unique(np.floor(z[bm]).astype(int))
    a_frame = parts["alpha"][bands].mean(axis=0)
    fe = 10 * np.log10(np.sum(M[bm] ** 2, axis=0) + 1e-12)
    active = fe > fe.max() - 40                        # skip silence
    rows5 = []
    if active.sum() >= 6:
        t_lo, t_hi = np.percentile(a_frame[active], [33, 67])
        for kind, fr in (("tonal", active & (a_frame >= t_hi)), ("noisy", active & (a_frame <= t_lo))):
            g = equal_energy_gain(B, A, fr)
            for shape, Bud, gg in (("aware", A, 1.0), ("mask", B, g)):
                y, d = emb(Bud, gg, fr)
                rows5.append(dict(clip_id=cid, frames=kind, n_frames=int(fr.sum()),
                                  mean_alpha=float(a_frame[fr].mean()), shape=shape,
                                  applied_energy=energy(d), **q(y)))
    pd.DataFrame(rows5).to_csv(os.path.join(DATA, f"e5_{cid}.csv"), index=False)

    # ---------- E6: codec survival ----------
    rows6 = []
    for codec in CODECS:
        for shape, y in (("aware", ya), ("mask_eq_energy", ym), ("mask_eq_quality", y4), ("flat", yf)):
            try:
                r = change_survival(x, y, codec)
            except Exception as e:
                print(f"  E6 {codec}/{shape} failed: {e}")
                r = {}
            rows6.append(dict(clip_id=cid, codec=codec, shape=shape, **r))
    pd.DataFrame(rows6).to_csv(os.path.join(DATA, f"e6_{cid}.csv"), index=False)
    print("  E6 " + "  ".join(f"{r['codec']}/{r['shape']} corr {r.get('corr', float('nan')):.2f}"
                             for r in rows6 if r["shape"] in ("aware", "mask_eq_energy")))
    print(f"[{cid}] done")


if __name__ == "__main__":
    main()
