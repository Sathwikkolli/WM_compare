"""
casp/sb_multi.py -- multibit spread-spectrum watermark, payload sweep, vs real AWARE.

One clip per task (sbatch array over clips20.json):

    python sb_multi.py --make-clips            # login node, once -> clips20.json (20 clips)
    python sb_multi.py --index 0               # one clip
    python sb_multi.py --mock --quick          # local plumbing test (fake AWARE)

Systems per clip, ALL matched to AWARE 6 dB's own PESQ on that clip:
    AWARE6      real AWARE, 20 bits, stock 6 dB budget, its own detector
    CASP_K{1,4,8,20}   our scheme, CASP-shaped budget, K payload bits
    AWSHAPE_K20        our scheme, AWARE-shaped budget, 20 bits (control: isolates the shape)

Our scheme: superposition. Each cell carries sum_i b_i p_i / sqrt(K) (p_i = key
pattern for bit i), scaled by the arm's per-cell amplitude A and a strength g.
g is found per system by bisection so that PESQ(host, marked) = PESQ of AWARE 6 dB
(never below it; g capped at 2.0 = 4x AWARE-6 energy).

Detection: whitened log-magnitude correlated with each bit's pattern (z_i),
standardised by a per-file wrong-key null. Presence = chi-square test on
sum z_i^2 (K degrees of freedom), detected if p <= 1e-3. Bit i = sign(z_i).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sb_embed as SB                         # noqa: E402
import ab_embed_detect as AB                  # noqa: E402
from sb_run import load_casp_eta              # noqa: E402

TH, C, psy, SR = SB.TH, AB.C, SB.psy, SB.SR
RUN = os.environ.get("CASP_SBM_RUN", "2026-10-14_casp-sbm")
RESULTS = os.path.join(AB.ROOT, "results", RUN)
CLIPS20 = os.path.join(HERE, "clips20.json")
KS = [1, 4, 8, 20]
P_DETECT = 1e-3
G_LO, G_HI = 0.01, 2.0

ATTACKS = {
    "clean":     lambda y: y,
    "awgn_20":   lambda y: C._awgn(y, 20),
    "awgn_10":   lambda y: C._awgn(y, 10),
    "awgn_5":    lambda y: C._awgn(y, 5),
    "lp_3500":   lambda y: C._lowpass(y, 3500),
    "lp_2500":   lambda y: C._lowpass(y, 2500),
    "lp_2000":   lambda y: C._lowpass(y, 2000),
    "mp3_32":    lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    "mp3_16":    lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "16k"], "mp3"),
    "opus_12":   lambda y: C._ffmpeg_roundtrip(y, ["-c:a", "libopus", "-b:a", "12k"], "opus"),
    "requant_6": lambda y: C._requant(y, 6),
}


# --------------------------------------------------------------------------- #
#  clip list
# --------------------------------------------------------------------------- #
def make_clips(n=20):
    import select_clips as S
    spec = json.load(open(C.CLIPS_JSON))
    rows = []
    for c in spec["clips"]:
        if c["duration_s"] < 3.0:
            continue
        y = C.load_16k(c["path"])[: int(AB.MAX_S * SR)]
        rows.append({**c, **S.stats(y)})
    rows.sort(key=lambda r: (-r["speech_frac"], -r["level_db"]))
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, n)).astype(int))
    picked = [rows[i] for i in idx]
    json.dump(dict(source=spec.get("source"), note="sb_multi: spread over pause fraction", clips=picked),
              open(CLIPS20, "w"), indent=2)
    print(f"wrote {CLIPS20}: {len(picked)} clips (of {len(rows)} eligible)")
    for i, c in enumerate(picked):
        print(f"[{i:2d}] {c['clip_id']} {c['utt']} {c['duration_s']}s speech_frac {c['speech_frac']:.2f}")


# --------------------------------------------------------------------------- #
#  patterns, embed, detect
# --------------------------------------------------------------------------- #
class Bank:
    """Key patterns for the bits and a wrong-key null bank, built once per clip."""

    def __init__(self, shape, key, n_null=200):
        self.shape, self.key = shape, int(key)
        self.N = int(np.prod(shape))
        self._sig = {}
        self.null = np.stack([SB.pattern(self.key * 1000003 + 500000 + j, shape).ravel()
                              for j in range(n_null)]).astype(np.float32)

    def sig(self, K):
        if K not in self._sig:
            self._sig[K] = np.stack([SB.pattern(self.key * 1000 + i, self.shape) for i in range(K)])
        return self._sig[K]


def embed_multi(X, A, bank, bits_pm, g, n):
    fi = SB.band_idx()
    K = len(bits_pm)
    acc = (np.asarray(bits_pm, np.float32)[:, None, None] * bank.sig(K)).sum(axis=0) / np.sqrt(K)
    mag = np.abs(X)
    ph = X / np.maximum(mag, 1e-12)
    new = mag.copy()
    new[fi] = np.maximum(mag[fi] + g * A[fi] * acc, 0.0)
    return psy.istft(new * ph, n).astype(np.float32)


def detect_multi(sig, bank, K, n):
    r = SB.whiten(sig, n).ravel().astype(np.float32)
    sqN = np.sqrt(r.size)
    z = (bank.sig(K).reshape(K, -1) @ r) / sqN
    nz = (bank.null @ r) / sqN
    sd = float(nz.std() + 1e-12)
    zs = z / sd
    S = float(np.sum(zs ** 2))
    p = float(chi2.sf(S, K))
    return dict(z=zs, p=p, z_eq=float(norm.isf(max(p, 1e-300) / 2)), detected=bool(p <= P_DETECT),
                bits_hat=np.where(z >= 0, 1, -1))


# --------------------------------------------------------------------------- #
#  quality and gain search
# --------------------------------------------------------------------------- #
HAVE_PESQ = True
try:
    from pesq import pesq as _pesq                  # noqa: F401
except Exception:
    HAVE_PESQ = False


def qfun(ref, deg):
    m = min(len(ref), len(deg))
    ref, deg = ref[:m], deg[:m]
    if HAVE_PESQ:
        from pesq import pesq
        return float(pesq(SR, ref, deg, "wb"))
    e = deg - ref
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum(e ** 2) + 1e-12)))   # SNR fallback


def find_g(host, X, A, bank, bits_pm, n, q_target, iters=9):
    def q_of(g):
        z = SB.to_pcm16(embed_multi(X, A, bank, bits_pm, g, n))
        return qfun(host, z), z

    q_hi, z_hi = q_of(G_HI)
    if q_hi >= q_target:
        return G_HI, q_hi, z_hi, "clamped_high"
    q_lo, z_lo = q_of(G_LO)
    if q_lo < q_target:
        return G_LO, q_lo, z_lo, "clamped_low"
    a, b = np.log(G_LO), np.log(G_HI)
    best = (G_LO, q_lo, z_lo)
    for _ in range(iters):
        m = 0.5 * (a + b)
        q, z = q_of(float(np.exp(m)))
        if q >= q_target:
            a, best = m, (float(np.exp(m)), q, z)
        else:
            b = m
    return best + ("ok",)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-clips", action="store_true")
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--max-s", type=float, default=AB.MAX_S)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.make_clips:
        return make_clips()

    mock = a.mock or a.wav is not None
    if a.wav:
        import librosa
        cid = os.path.splitext(os.path.basename(a.wav))[0]
        y, _ = librosa.load(a.wav, sr=SR, mono=True)
    elif a.mock:
        import run_casp
        cid, y = "synth", run_casp.synth()
    else:
        clip = json.load(open(CLIPS20))["clips"][a.index]
        cid = clip["clip_id"]
        y = C.load_16k(clip["path"])
    y = y[: int((2.0 if a.quick else a.max_s) * SR)].astype("float32")
    y_n = (y / max(float(np.max(np.abs(y))), 1e-9)).astype("float32")
    n = len(y_n)
    out = a.out or os.path.join(RESULTS, cid)
    os.makedirs(out, exist_ok=True)
    print(f"[{cid}] {n / SR:.1f}s  quality axis: {'PESQ' if HAVE_PESQ else 'SNR (fallback)'}")
    t0 = time.time()

    host = SB.to_pcm16(y_n)
    X = psy.stft(y_n)
    M = np.abs(X)
    E_ref = SB.reference_energy(M)
    eta_casp, nat_g = load_casp_eta(cid, y_n, M, mock, a.threads)
    A_casp, info_c = SB.amplitude_map(SB.shape_map("casp", M, eta_casp), M, E_ref)
    A_aw, info_a = SB.amplitude_map(SB.shape_map("aware_shape", M, None), M, E_ref)
    print(f"  CASP map ready [{time.time() - t0:.0f}s]  natural g: "
          + ", ".join(f"{k}={v:.3f}" for k, v in nat_g.items()))

    key = 2000 + 31 * a.index + a.seed
    fi = SB.band_idx()
    bank = Bank((len(fi), X.shape[1]), key, a.n_null)
    rng = np.random.RandomState(a.seed * 1000 + a.index)
    bits01 = rng.randint(0, 2, C.N_BITS)
    bits_pm = (2 * bits01 - 1).astype(np.float32)

    # ---- AWARE (real, or the plumbing mock) -------------------------------
    if mock:
        load, embed_watermark, detect_watermark = AB.mock_aware()
    else:
        load, embed_watermark, detect_watermark = C.import_aware()
    embedder, detector = load(name=C.AWARE_MODEL)
    z_aw = np.asarray(embed_watermark(y_n, SR, bits01, embedder), "float32")
    z_aw = SB.to_pcm16(np.pad(z_aw, (0, max(0, n - len(z_aw))))[:n])
    q_aw = qfun(host, z_aw)
    print(f"  AWARE 6 dB quality {q_aw:.3f}  [{time.time() - t0:.0f}s]")

    # ---- our systems at AWARE's quality ------------------------------------
    systems = {"AWARE6": dict(sig=z_aw, g=np.nan, status="stock", K=20, kind="aware")}
    for K in KS:
        g, q, z, st = find_g(host, X, A_casp, bank, bits_pm[:K], n, q_aw)
        systems[f"CASP_K{K}"] = dict(sig=z, g=g, status=st, K=K, kind="ss", q=q)
    g, q, z, st = find_g(host, X, A_aw, bank, bits_pm, n, q_aw)
    systems["AWSHAPE_K20"] = dict(sig=z, g=g, status=st, K=20, kind="ss", q=q)
    for nm, s in systems.items():
        print(f"   {nm:<12} g={s['g']:.3f} ({s['status']})" if s["kind"] == "ss" else f"   {nm:<12} stock")
    print(f"  gains found [{time.time() - t0:.0f}s]")

    attacks = ATTACKS if not a.quick else {k: ATTACKS[k] for k in ("clean", "awgn_10", "lp_2500", "mp3_16")}
    rows = []
    for att, fn in attacks.items():
        for nm, s in systems.items():
            sig = np.asarray(fn(s["sig"]), "float32")
            if s["kind"] == "aware":
                dec, conf = detect_watermark(sig, SR, detector)[:2]
                rows.append(dict(clip_id=cid, kind="wm", system=nm, K=20, attack=att, detected=bool(conf >= 0.5),
                                 score=float(conf), bit_acc=C.bit_acc(dec, bits01), g=np.nan))
            else:
                d = detect_multi(sig, bank, s["K"], n)
                rows.append(dict(clip_id=cid, kind="wm", system=nm, K=s["K"], attack=att, detected=d["detected"],
                                 score=d["z_eq"], bit_acc=float(np.mean(d["bits_hat"] == bits_pm[:s["K"]])), g=s["g"]))
        # unwatermarked host, same attack
        sig = np.asarray(fn(host), "float32")
        dec, conf = detect_watermark(sig, SR, detector)[:2]
        rows.append(dict(clip_id=cid, kind="neg", system="AWARE6", K=20, attack=att, detected=bool(conf >= 0.5),
                         score=float(conf), bit_acc=C.bit_acc(dec, bits01), g=np.nan))
        for K in KS:
            d = detect_multi(sig, bank, K, n)
            rows.append(dict(clip_id=cid, kind="neg", system=f"CASP_K{K}", K=K, attack=att, detected=d["detected"],
                             score=d["z_eq"], bit_acc=float(np.mean(d["bits_hat"] == bits_pm[:K])), g=np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(out, "sbm_rows.csv"), index=False)

    qual = {}
    for nm, s in systems.items():
        q = C.quality(host, s["sig"])
        qual[nm] = {k: q.get(k) for k in ("pesq", "stoi", "snr_db")}
    json.dump(dict(clip_id=cid, key=int(key), payload="".join(map(str, bits01)), aware_q=q_aw,
                   quality_axis="pesq" if HAVE_PESQ else "snr_db", natural_g=nat_g,
                   systems={nm: dict(g=(None if np.isnan(s["g"]) else float(s["g"])), status=s["status"], K=s["K"],
                                     quality=qual[nm]) for nm, s in systems.items()},
                   energy_info=dict(casp=info_c, aware_shape=info_a)),
              open(os.path.join(out, "sbm_meta.json"), "w"), indent=1)

    import soundfile as sf
    for nm, v in [("original", host), ("aware6", z_aw), ("casp_k20", systems["CASP_K20"]["sig"])]:
        sf.write(os.path.join(out, f"{nm}.wav"), v, SR, subtype="PCM_16")
    AB.spectrogram_figure(os.path.join(out, "spectrograms.png"),
                          f"{cid}: AWARE 6 dB vs CASP multibit (20 bits) at equal quality",
                          host, z_aw, systems["CASP_K20"]["sig"])
    print(f"done [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()
