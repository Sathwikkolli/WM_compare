"""
casp/ab_embed_detect.py -- stock AWARE vs AWARE-with-CASP-budget, one clip.

Both arms use the SAME frozen AWARE detector and the SAME optimiser and loss.
Only the per-coefficient `bounds` handed to AWAREEmbedder._optimize differ:

    A  stock      : |delta| <= eta0 * |X|              (eta0 = 10^(-6/20), flat)
    B  AWARE+CASP : |delta| <= min(kappa*eta_casp(band,time), 1) * |X|

eta_casp is the per-(band, 256 ms block) tolerance solved from the revised CASP
model (casp/threshold.py, target 1.0). kappa is chosen so B's in-band energy of
the allowed change equals A's (iso-energy): this tests where the budget goes,
not how quiet it is. The AWARE package is not modified; the override wraps
_optimize the way masking/common.OptimizeCapture does.

    python ab_embed_detect.py --make-clips     # login node, once -> clips_ab.json
    python ab_embed_detect.py --index 0        # one clip (sbatch array 0-4)
    python ab_embed_detect.py --mock           # plumbing test, no AWARE needed
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "masking"))

import threshold as TH                         # noqa: E402  (also loads masking/psy)
import common as C                             # noqa: E402  (masking/common.py)

psy = TH.psy
SR = C.SR
RUN = os.environ.get("CASP_AB_RUN", "2026-10-14_casp-ab")
CASP_RUN = os.environ.get("CASP_RUN", "2026-10-14_casp-threshold")
RESULTS = os.path.join(ROOT, "results", RUN)
CASP_RESULTS = os.path.join(ROOT, "results", CASP_RUN)
CLIPS_AB = os.path.join(HERE, "clips_ab.json")
CLIPS2 = os.path.join(HERE, "clips2.json")
MAX_S = 5.0
LEVEL_DB = 65.0
DETECT_CONF = 0.5          # results/THRESHOLD_DECISION.md

ATTACKS = {
    "clean":   lambda y: y,
    "mp3_64":  lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    "mp3_32":  lambda y: C._ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    "awgn_20": lambda y: C._awgn(y, 20),
    "awgn_10": lambda y: C._awgn(y, 10),
    "lp_3500": lambda y: C._lowpass(y, 3500),
    "lp_2500": lambda y: C._lowpass(y, 2500),
}


# --------------------------------------------------------------------------- #
#  clip list
# --------------------------------------------------------------------------- #
def make_clips():
    """4 clips already analysed by run_casp + 1 more with a mid pause-fraction."""
    import select_clips as S
    base = json.load(open(CLIPS2))
    have = {c["clip_id"] for c in base["clips"]}
    spec = json.load(open(C.CLIPS_JSON))
    rows = []
    for c in spec["clips"]:
        if c["clip_id"] in have or c["duration_s"] < 3.0:
            continue
        y = C.load_16k(c["path"])[: int(MAX_S * SR)]
        rows.append({**c, **S.stats(y)})
    target = float(np.median([c.get("speech_frac", 0.8) for c in base["clips"]]))
    extra = min(rows, key=lambda r: abs(r["speech_frac"] - target))
    extra["role"] = "extra_mid"
    out = dict(source=base.get("source"), note="clips2 + 1 extra", clips=base["clips"] + [extra])
    json.dump(out, open(CLIPS_AB, "w"), indent=2)
    for i, c in enumerate(out["clips"]):
        print(f"[{i}] {c['clip_id']} {c['utt']}  {c['duration_s']}s  {c.get('role')}")
    print(f"wrote {CLIPS_AB}; submit: sbatch ab_casp.sbatch")


# --------------------------------------------------------------------------- #
#  CASP map for a clip
# --------------------------------------------------------------------------- #
def casp_eta_map(cid, x_n, threads, target=1.0):
    """(F,U) eta on AWARE's STFT grid. Reuses probe.npz from run_casp if present,
    otherwise probes now (same settings as run_casp defaults)."""
    X = psy.stft(x_n)
    F, U = X.shape
    p = os.path.join(CASP_RESULTS, cid, "probe.npz")
    if os.path.exists(p):
        z = np.load(p)
        eta_kb = z[f"eta_kb_{target}"]
        bands = [tuple(b) for b in z["bands"]]
        blocks = [tuple(b) for b in z["blocks"]]
        print(f"  CASP map: reused {p}")
    else:
        from casp_front import CASP, spl_scale
        import validate as VAL
        casp = CASP(fs=SR, threads=threads)
        d_jnd = VAL.calibrate(casp)
        x_pa, _ = spl_scale(x_n, LEVEL_DB)
        g = TH.probe_grid(casp, x_pa, n_bands=8, block=16, etas=TH.ETAS)
        d_t = target * d_jnd / np.sqrt(8)
        eta_kb, _ = TH.solve_eta(g["D"], g["etas"], d_t)
        bands = [tuple(b) for b in g["bands"]]
        blocks = [tuple(b) for b in g["blocks"]]
        out = os.path.join(CASP_RESULTS, cid)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "probe.npz"), D=g["D"], etas=g["etas"],
                            bands=g["bands"], hz=g["hz"], blocks=g["blocks"], d_jnd=d_jnd,
                            **{f"eta_kb_{target}": eta_kb})
        print(f"  CASP map: probed now (D_JND {d_jnd:.1f})")
    return TH.eta_map(eta_kb, bands, blocks, (F, U)), X


# --------------------------------------------------------------------------- #
#  budget override for AWAREEmbedder._optimize
# --------------------------------------------------------------------------- #
def align_frames(a, n):
    """(rows, U) -> (rows, n) by edge padding / cropping; refuses big mismatches."""
    U = a.shape[1]
    if abs(U - n) > 3:
        raise RuntimeError(f"frame count mismatch: CASP grid {U} vs AWARE {n}")
    if U < n:
        a = np.pad(a, ((0, 0), (0, n - U)), mode="edge")
    return a[:, :n]


class BudgetOverride:
    def __init__(self, embedder):
        self.orig = embedder._optimize
        embedder._optimize = self._wrapped
        self.eta_fu = None
        self.host_M = None
        self.fixed = False
        self.info = {}

    def enable(self, eta_fu, host_M, fixed=False):
        """fixed=False: iso-energy (kappa-scaled, capped at 1) as in the first A/B run.
        fixed=True : eta_fu is used exactly as given (clipped to [0,1]); no rescaling."""
        self.eta_fu, self.host_M, self.fixed = eta_fu, host_M, fixed

    def disable(self):
        self.eta_fu = None

    def _wrapped(self, initial_coeffs, stft_magnitude, watermark_pattern,
                 freq_indices, not_freq_indices, bounds, stft_phase):
        if self.eta_fu is None:
            return self.orig(initial_coeffs, stft_magnitude, watermark_pattern,
                             freq_indices, not_freq_indices, bounds, stft_phase)
        fi = np.asarray(freq_indices)
        init = initial_coeffs.detach().cpu().numpy().astype("float64").copy()
        nf = len(fi)
        nu = init.size // nf
        M = init.reshape(nf, nu)                                  # (freq bin, frame)
        lo = np.array([b[0] for b in bounds], dtype="float64").reshape(nf, nu)
        up = np.array([b[1] for b in bounds], dtype="float64").reshape(nf, nu)
        eta = align_frames(self.eta_fu[fi], nu)
        # the CASP map and AWARE's own STFT must be the same time axis
        prof_a = np.log(np.sqrt((M ** 2).mean(0)) + 1e-9)
        prof_c = np.log(np.sqrt((align_frames(self.host_M[fi], nu) ** 2).mean(0)) + 1e-9)
        corr = float(np.corrcoef(prof_a, prof_c)[0, 1])
        if corr < 0.9:
            raise RuntimeError(f"CASP map misaligned with AWARE STFT (profile corr {corr:.2f})")
        E_stock = float((((up - lo) / 2) ** 2).sum())
        kappa = 1.0
        if self.fixed:
            e = np.clip(eta, 0.0, 1.0)
        else:
            for _ in range(8):                                    # energy match, with the <=1 cap
                e = np.minimum(kappa * eta, 1.0)
                E = float(((e * M) ** 2).sum())
                kappa *= np.sqrt(E_stock / max(E, 1e-30))
            e = np.minimum(kappa * eta, 1.0)
        new_lo, new_up = M * (1 - e), M * (1 + e)
        self.info = dict(kappa=float(kappa), align_corr=corr, n_freq=nf, n_frames=nu,
                         eta_stock_median=float(np.median(((up - lo) / 2)[M > 0] / M[M > 0])),
                         eta_casp_median=float(np.median(e[M > 0])),
                         frac_capped=float(np.mean(e >= 1.0 - 1e-9)),
                         E_ratio=float(((e * M) ** 2).sum() / E_stock),
                         max_rel_diff_vs_stock=float(np.max(np.abs(M * (1 + e) - up) / np.maximum(M, 1e-9))))
        nb = [(float(a), float(b)) for a, b in zip(new_lo.ravel(), new_up.ravel())]
        return self.orig(initial_coeffs, stft_magnitude, watermark_pattern,
                         freq_indices, not_freq_indices, nb, stft_phase)


# --------------------------------------------------------------------------- #
#  mock AWARE (plumbing test only)
# --------------------------------------------------------------------------- #
def mock_aware():
    class Emb:
        tolerance_db = 6.0
        num_iterations = 0

        def _optimize(self, initial_coeffs, stft_magnitude, watermark_pattern,
                      freq_indices, not_freq_indices, bounds, stft_phase):
            import torch
            up = np.array([b[1] for b in bounds], "float32")
            lo = np.array([b[0] for b in bounds], "float32")
            s = np.random.RandomState(1).choice([0.0, 1.0], size=up.shape).astype("float32")
            return torch.from_numpy(lo + (up - lo) * s)

    class Det:
        pass

    emb, det = Emb(), Det()
    det.last_bits = None

    def embed(y, sr, bits, embedder):
        import torch
        X = psy.stft(y.astype("float32"))
        fi = np.where((psy.freqs() >= 1000) & (psy.freqs() <= 4000))[0]
        M = np.abs(X)[fi]
        nf, nu = M.shape
        b0 = 10 ** (-embedder.tolerance_db / 20)
        bounds = [(float(m * (1 - b0)), float(m * (1 + b0))) for m in M.ravel()]
        out = embedder._optimize(torch.from_numpy(M.ravel().astype("float32")), None, None,
                                 fi, None, bounds, None).numpy().reshape(nf, nu)
        Y = X.copy()
        Y[fi] = X[fi] / np.maximum(np.abs(X[fi]), 1e-12) * out
        det.last_bits = np.asarray(bits)
        out = psy.istft(Y, len(y)).astype("float32")
        short = int(os.environ.get("CASP_MOCK_SHORT", "0"))   # test: output shorter than input
        return out[: len(out) - short]

    def detect(y, sr, detector):
        rng = np.random.RandomState(int(abs(y[:100]).sum() * 1e6) % 2 ** 31)
        return detector.last_bits.copy(), float(rng.uniform(0.6, 1.0))

    return (lambda name=None: (emb, det)), embed, detect


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def spectrogram_figure(path, cid, y, z_a, z_c, band=(1000, 4000), fmax=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = psy.freqs()
    X, A, B = (psy.stft(v.astype("float32")) for v in (y, z_a, z_c))
    t = np.arange(X.shape[1]) * psy.HOP / SR
    db = lambda S: 20 * np.log10(np.maximum(np.abs(S), 1e-7))
    ref = db(X)
    vmin, vmax = np.percentile(ref, 2), np.percentile(ref, 99.7)
    EA, EB = A - X, B - X
    ea, eb = db(EA), db(EB)
    dv = (np.percentile(ea[f <= fmax], 99.5))
    fig, ax = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey=True)
    k = f <= fmax

    def show(a, S, title, vmin_, vmax_, cmap="magma"):
        im = a.pcolormesh(t, f[k] / 1000, S[k], shading="auto", cmap=cmap, vmin=vmin_, vmax=vmax_)
        a.set_title(title, fontsize=10)
        for y_ in (band[0] / 1000, band[1] / 1000):
            a.axhline(y_, color="w", ls=":", lw=0.7)
        return im

    im = show(ax[0, 0], ref, "original", vmin, vmax)
    show(ax[0, 1], db(A), "AWARE (stock)", vmin, vmax)
    show(ax[0, 2], db(B), "AWARE + CASP budget", vmin, vmax)
    fig.colorbar(im, ax=ax[0, :], pad=0.01, label="|STFT| (dB)")
    im2 = show(ax[1, 0], ea, "watermark added: AWARE", dv - 60, dv)
    show(ax[1, 1], eb, "watermark added: AWARE + CASP", dv - 60, dv)
    fig.colorbar(im2, ax=ax[1, :2], pad=0.01, label="|change| (dB)")
    d = np.clip(eb - ea, -20, 20)
    quiet = ((ea < dv - 55) & (eb < dv - 55)) | (f[:, None] < band[0]) | (f[:, None] > band[1])
    d = np.where(quiet, np.nan, d)                    # no change in either arm / outside the band
    im3 = show(ax[1, 2], d, "AWARE+CASP minus AWARE (dB)  red = CASP adds more", -20, 20, "RdBu_r")
    fig.colorbar(im3, ax=ax[1, 2], pad=0.01)
    for a in ax[1]:
        a.set_xlabel("time (s)")
    for a in ax[:, 0]:
        a.set_ylabel("kHz")
    fig.suptitle(f"{cid}: where each embedder puts its energy (dotted = 1-4 kHz embedding band)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--make-clips", action="store_true")
    ap.add_argument("--mock", action="store_true", help="fake AWARE + synthetic clip (local plumbing test)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.make_clips:
        return make_clips()

    if a.mock:
        import run_casp
        cid, y = "mock", run_casp.synth()[: int(MAX_S * SR)]
        load, embed_watermark, detect_watermark = mock_aware()
        label = "mock"
    else:
        clip = json.load(open(CLIPS_AB))["clips"][a.index]
        cid = clip["clip_id"]
        y = C.load_16k(clip["path"])[: int(MAX_S * SR)]
        label = clip.get("role")
    out = a.out or os.path.join(RESULTS, cid)
    os.makedirs(out, exist_ok=True)
    print(f"[{cid}] {label}  {len(y) / SR:.1f}s")

    y_n = (y / max(float(np.max(np.abs(y))), 1e-9)).astype("float32")
    t0 = time.time()
    eta_fu, X_host = (casp_eta_map(cid, y_n, a.threads) if not a.mock
                      else (np.clip(np.random.RandomState(0).uniform(0.2, 1.0, psy.stft(y_n).shape), 0, 1).astype("float32"),
                            psy.stft(y_n)))
    host_M = np.abs(X_host)
    print(f"  map ready [{time.time() - t0:.0f}s]")

    if not a.mock:
        load, embed_watermark, detect_watermark = C.import_aware()
    embedder, detector = load(name=C.AWARE_MODEL)
    ov = BudgetOverride(embedder)
    bits = np.random.RandomState(a.seed * 1000 + a.index).randint(0, 2, C.N_BITS)

    t1 = time.time()
    z_a = np.asarray(embed_watermark(y, SR, bits, embedder), "float32")[: len(y)]
    t_a = time.time() - t1
    ov.enable(eta_fu, host_M)
    t1 = time.time()
    z_c = np.asarray(embed_watermark(y, SR, bits, embedder), "float32")[: len(y)]
    t_c = time.time() - t1
    ov.disable()
    print(f"  embed AWARE {t_a:.0f}s  AWARE+CASP {t_c:.0f}s  budget: {ov.info}")

    arms = {"AWARE": z_a, "AWARE+CASP": z_c}
    qual = {}
    for name, z in arms.items():
        q = C.quality(y, z)
        qual[name] = {k: q.get(k) for k in ("pesq", "stoi", "snr_db", "si_snr_db")}
        print(f"  {name:<11} PESQ {q.get('pesq')}  STOI {q.get('stoi')}  SNR {q.get('snr_db'):.2f}")

    rows = []
    for att, fn in ATTACKS.items():
        for kind, arm, sig in [("wm", "AWARE", z_a), ("wm", "AWARE+CASP", z_c), ("neg", "none", y)]:
            try:
                za = np.asarray(fn(sig), "float32")
                dec, conf = detect_watermark(za, SR, detector)[:2]
                acc = C.bit_acc(dec, bits)
            except Exception as e:
                print(f"  {arm}/{att}: FAILED {e}")
                acc, conf = float("nan"), float("nan")
            rows.append(dict(clip_id=cid, kind=kind, arm=arm, attack=att, bit_acc=acc,
                             conf=float(conf), detected=bool(conf >= DETECT_CONF)))
        r = {x["arm"]: x for x in rows[-3:]}
        print(f"  {att:<8} AWARE {r['AWARE']['bit_acc']:.2f}/{r['AWARE']['conf']:.2f}   "
              f"+CASP {r['AWARE+CASP']['bit_acc']:.2f}/{r['AWARE+CASP']['conf']:.2f}   "
              f"clean-neg conf {r['none']['conf']:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "ab_rows.csv"), index=False)
    json.dump(dict(clip_id=cid, payload="".join(map(str, bits)), quality=qual,
                   budget=ov.info, embed_seconds=dict(AWARE=t_a, CASP=t_c),
                   tolerance_db=getattr(embedder, "tolerance_db", None)),
              open(os.path.join(out, "ab_meta.json"), "w"), indent=1)

    import soundfile as sf
    for name, v in [("original", y), ("aware", z_a), ("aware_casp", z_c)]:
        sf.write(os.path.join(out, f"{name}.wav"), np.clip(v, -1, 1), SR, subtype="PCM_16")
    spectrogram_figure(os.path.join(out, "spectrograms.png"), cid, y, z_a, z_c)
    print(f"done [{time.time() - t0:.0f}s] -> {out}")


if __name__ == "__main__":
    main()
