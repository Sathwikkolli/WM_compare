"""
casp/bm_core.py -- shared pieces for the spread-spectrum vs neural watermark benchmark.

Design (see the plan in the conversation / results/2026-10-14_casp-bench/README.md):

  * ONE master rate, 22.05 kHz, like cascade/cascade_lib.py. Each system resamples
    to its own native rate internally and returns audio at the master rate.
  * Every system embeds a RANDOM payload per clip (cascade_lib's adapters embed a
    fixed message; we reuse their loaded models, not their embed()).
  * Systems: AudioSeal (16 bit), AWARE (20 bit), Timbre (10 bit), SS (16 bit,
    our keyed spread-spectrum scheme; shape = AWARE-shape or CASP-shape).
  * Quality matching by RESIDUAL SCALING: z = host + s * (marked - host), s found
    by bisection so PESQ-wb(host, z) hits a target. Residual scaling is off-design
    for the neural systems, hence the "as shipped" operating point (s = 1) too.
  * Attacks: the 17 VoxWatermark families (cascade/vox_attacks.py) plus
    absolute-Hz filters, notch, bit-depth requantisation, resample round trip,
    pitch shift, head crop, gain, and (optional, guarded) DAC.

Nothing here edits the model packages.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASC = os.path.join(ROOT, "cascade")
for p in (HERE, CASC):
    if p not in sys.path:
        sys.path.insert(0, p)

SR = 22050            # master rate (attacks run here)
SR16 = 16000          # AudioSeal / AWARE / SS native rate, and PESQ rate
MAX_S = 5.0
PEAK = 0.9
CASP_RESULTS = os.environ.get(
    "CASP_RUN_DIR", os.path.join(ROOT, "results", "2026-10-14_casp-threshold"))


# --------------------------------------------------------------------------- #
#  audio helpers
# --------------------------------------------------------------------------- #
def resample(y, sr_from, sr_to):
    if sr_from == sr_to:
        return np.asarray(y, dtype=np.float32)
    import librosa
    return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr_from,
                            target_sr=sr_to).astype(np.float32)


def load_master(path, max_s=MAX_S):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y[: int(max_s * SR)].astype(np.float32)
    return (y / max(float(np.max(np.abs(y))), 1e-9) * PEAK).astype(np.float32)


def fit_len(y, n):
    y = np.asarray(y, dtype=np.float32)
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def pcm16(x):
    return (np.round(np.clip(x, -1, 1) * 32767) / 32767).astype(np.float32)


# --------------------------------------------------------------------------- #
#  quality
# --------------------------------------------------------------------------- #
try:
    from pesq import pesq as _pesq_fn
    HAVE_PESQ = True
except Exception:
    HAVE_PESQ = False


def _snr(ref, deg):
    n = min(len(ref), len(deg))
    e = deg[:n] - ref[:n]
    return float(10 * np.log10(np.sum(ref[:n] ** 2) / (np.sum(e ** 2) + 1e-12)))


def quality(host22, deg22, full=False):
    """PESQ-wb at 16 kHz (SNR fallback if pesq is missing). full=True adds STOI."""
    ref = resample(host22, SR, SR16)
    deg = resample(deg22, SR, SR16)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    out = {"snr_db": _snr(ref, deg)}
    out["pesq"] = float(_pesq_fn(SR16, ref, deg, "wb")) if HAVE_PESQ else out["snr_db"]
    if full:
        try:
            from pystoi import stoi
            out["stoi"] = float(stoi(ref, deg, SR16, extended=False))
        except Exception:
            out["stoi"] = float("nan")
    return out


def apply_scale(host, marked, s):
    n = min(len(host), len(marked))
    return pcm16(host[:n] + s * (marked[:n] - host[:n]))


def match_scale(host, marked, target, lo=0.05, hi=3.0, iters=10):
    """Bisection on log(s): largest s whose quality >= target (never below it).
    Returns (s, achieved_quality, status)."""
    def q(s):
        return quality(host, apply_scale(host, marked, s))["pesq"]

    q_hi = q(hi)
    if q_hi >= target:
        return hi, q_hi, "clamped_high"
    q_lo = q(lo)
    if q_lo < target:
        return lo, q_lo, "clamped_low"
    a, b = np.log(lo), np.log(hi)
    best = (lo, q_lo)
    for _ in range(iters):
        m = 0.5 * (a + b)
        s = float(np.exp(m))
        v = q(s)
        if v >= target:
            a, best = m, (s, v)
        else:
            b = m
    return best[0], best[1], "ok"


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def asv_clips(n=20):
    """The 20 ASVspoof clips already used in the CASP runs (casp/clips20.json)."""
    import json
    spec = json.load(open(os.path.join(HERE, "clips20.json")))
    return [(c["clip_id"], c["path"]) for c in spec["clips"][:n]]


def emilia_clips(n, seed=0, skip=0):
    import null_test as NT           # cascade/null_test.py
    paths = NT.emilia_paths(n + skip, min_dur=MAX_S, seed=seed)[skip:skip + n]
    return [(f"emilia{seed}_{i:03d}", p) for i, p in enumerate(paths)]


# --------------------------------------------------------------------------- #
#  SS building blocks (self-contained copy of the sb_multi scheme)
# --------------------------------------------------------------------------- #
class SSBank:
    def __init__(self, shape, key, n_null=200):
        import sb_embed as SB
        self.SB, self.shape, self.key = SB, shape, int(key)
        self.N = int(np.prod(shape))
        self._sig = {}
        self.null = np.stack([SB.pattern(self.key * 1000003 + 500000 + j, shape).ravel()
                              for j in range(n_null)]).astype(np.float32)

    def sig(self, K):
        if K not in self._sig:
            self._sig[K] = np.stack([self.SB.pattern(self.key * 1000 + i, self.shape)
                                     for i in range(K)])
        return self._sig[K]


def casp_eta(clip_id, shape_fu):
    """CASP tolerance map (F,U) for a clip from its probe.npz, or None."""
    p = os.path.join(CASP_RESULTS, clip_id, "probe.npz")
    if not os.path.exists(p):
        return None
    import threshold as TH
    z = np.load(p)
    bands = [tuple(b) for b in z["bands"]]
    blocks = [tuple(b) for b in z["blocks"]]
    eta_kb, _ = TH.solve_eta(z["D"], z["etas"], 1.0 * float(z["d_jnd"]) / np.sqrt(len(bands)))
    return TH.eta_map(eta_kb, bands, blocks, shape_fu)


# --------------------------------------------------------------------------- #
#  systems.  embed(y22, bits01) -> marked22 (nominal strength)
#            detect(y22) -> (native_score or nan, bits01 array)
# --------------------------------------------------------------------------- #
class AudioSealSys:
    name, nbits, native_sr, has_score = "AudioSeal", 16, SR16, True

    def __init__(self):
        import cascade_lib as cl
        self.a = cl.get_adapter("audioseal")
        self.torch = self.a._torch

    def embed(self, y22, bits, **kw):
        t = self.torch
        x = t.from_numpy(resample(y22, SR, SR16)).unsqueeze(0).unsqueeze(0)
        msg = t.tensor([[int(b) for b in bits]], dtype=t.int32)
        with t.no_grad():
            wm = self.a._gen(x, sample_rate=SR16, message=msg, alpha=1.0)
        return resample(wm.squeeze().cpu().numpy(), SR16, SR)

    def detect(self, y22, **kw):
        t = self.torch
        x = t.from_numpy(resample(y22, SR, SR16)).unsqueeze(0).unsqueeze(0)
        with t.no_grad():
            prob, msg = self.a._det.detect_watermark(x, sample_rate=SR16)
        return float(prob), (msg.squeeze() > 0.5).int().cpu().numpy().ravel()[:16]


class AwareSys:
    name, nbits, native_sr, has_score = "AWARE", 20, SR16, True

    def __init__(self):
        import cascade_lib as cl
        self.a = cl.get_adapter("aware")

    def embed(self, y22, bits, **kw):
        from aware.service import embed_watermark
        y16 = resample(y22, SR, SR16)
        out = embed_watermark(y16, SR16, np.asarray(bits, dtype=np.int64), self.a._emb)
        return resample(fit_len(np.asarray(out, dtype=np.float32), len(y16)), SR16, SR)

    def detect(self, y22, **kw):
        from aware.service import detect_watermark
        pat, conf = detect_watermark(resample(y22, SR, SR16), SR16, self.a._det)
        return float(conf), np.asarray(pat).astype(int).ravel()[:20]


class TimbreSys:
    name, nbits, native_sr, has_score = "Timbre", 10, SR, False

    def __init__(self):
        import cascade_lib as cl
        self.cl = cl
        self.a = cl.get_adapter("timbre")
        self.torch = self.a._torch
        self.nbits = self.a.NB

    def embed(self, y22, bits, **kw):
        t, a = self.torch, self.a
        x = t.from_numpy(np.asarray(y22, np.float32)).unsqueeze(0).unsqueeze(0).to(a._dev)
        msg = t.from_numpy(np.array([[list(int(b) for b in bits)]])).float().to(a._dev) * 2 - 1
        with self.cl.chdir(self.cl.TIMBRE_REPO), t.no_grad():
            enc, _ = a._enc.test_forward(x, msg)
        return enc.squeeze().detach().cpu().numpy().astype(np.float32)

    def detect(self, y22, **kw):
        t, a = self.torch, self.a
        x = t.from_numpy(np.asarray(y22, np.float32)).unsqueeze(0).unsqueeze(0).to(a._dev)
        with self.cl.chdir(self.cl.TIMBRE_REPO), t.no_grad():
            dec = a._dec.test_forward(x)
        v = dec.squeeze().detach().cpu().numpy().ravel()[: a.NB]
        return float("nan"), (v >= 0).astype(int)


class SSSys:
    """Keyed spread spectrum: K bits superposed, STFT-magnitude domain, 1-4 kHz.
    Call prepare() per clip before embed()/detect()."""
    has_score = True
    native_sr = SR16

    def __init__(self, shape="aware_shape", K=16, n_null=200):
        import sb_embed as SB
        self.SB, self.shape_name, self.nbits, self.n_null = SB, shape, K, n_null
        self.name = {"aware_shape": "SS", "casp": "SS-CASP", "flat_abs": "SS-flat"}[shape]
        self.fi = SB.band_idx()

    def prepare(self, host22, key, clip_id=None):
        y16 = resample(host22, SR, SR16)
        self.n16 = len(y16)
        U = self.SB.psy.stft(y16).shape[1]
        self.bank = SSBank((len(self.fi), U), key, self.n_null)
        self.key = key
        self.clip_id = clip_id
        if self.shape_name == "casp":
            eta = casp_eta(clip_id, (self.SB.psy.stft(y16).shape))
            if eta is None:
                return False
            self.eta = eta
        return True

    def embed(self, y22, bits, **kw):
        SB = self.SB
        y16 = resample(y22, SR, SR16)
        X = SB.psy.stft(y16)
        M = np.abs(X)
        W = SB.shape_map(self.shape_name, M, getattr(self, "eta", None))
        A, _ = SB.amplitude_map(W, M, SB.reference_energy(M))
        bits_pm = (2 * np.asarray(bits, np.float32) - 1)
        K = len(bits_pm)
        acc = (bits_pm[:, None, None] * self.bank.sig(K)).sum(axis=0) / np.sqrt(K)
        new = M.copy()
        new[self.fi] = np.maximum(M[self.fi] + A[self.fi] * acc, 0.0)
        ph = X / np.maximum(M, 1e-12)
        m16 = SB.psy.istft(new * ph, len(y16)).astype(np.float32)
        return resample(m16, SR16, SR)

    def detect(self, y22, search=False, **kw):
        """search=True: try head-crop offsets 0 .. 2.5 s in 64-sample steps (16 kHz)
        and keep the best; the p-value is Bonferroni-corrected for the number of
        offsets, so the null stays calibrated. Used only for head-crop attacks."""
        from scipy.stats import chi2, norm
        y16 = resample(y22, SR, SR16)
        K = self.nbits
        shifts = list(range(0, int(2.5 * SR16) + 1, 64)) if search else [0]
        bank = self.bank.sig(K).reshape(K, -1)
        best = None
        for sh in shifts:
            yy = np.concatenate([np.zeros(sh, np.float32), y16]) if sh else y16
            r = self.SB.whiten(yy, self.n16).ravel().astype(np.float32)
            sq = np.sqrt(r.size)
            z = (bank @ r) / sq
            sd = float((self.bank.null @ r / sq).std() + 1e-12)
            S = float(np.sum((z / sd) ** 2))
            if best is None or S > best[0]:
                best = (S, z)
        S, z = best
        p = min(1.0, float(chi2.sf(S, K)) * len(shifts))
        self.last_p = p
        return float(norm.isf(max(p, 1e-300) / 2)), (z >= 0).astype(int)


# --------------------------------------------------------------------------- #
#  attacks (applied at the master rate)
# --------------------------------------------------------------------------- #
_REVERSE = {"opus", "encodec", "quantization", "lowpass", "mp3", "aac", "requant", "resample", "dac"}
_EXTRA = {
    "lp_hz":     [(f"{c}Hz", c) for c in (4000, 3500, 3000, 2500, 2000, 1500, 1000)],
    "hp_hz":     [(f"{c}Hz", c) for c in (100, 200, 300, 500)],
    "notch":     [(f"{c}Hz", c) for c in (500, 1000, 1500, 2000, 2500, 3000, 3500)],
    "requant":   [(f"{b}bit", b) for b in (4, 6, 8, 12, 16)],
    "resample":  [(f"{r}", r) for r in (4000, 8000, 11025, 16000)],
    "pitch":     [(f"{s:+d}", s) for s in (-2, -1, 1, 2)],
    "crop_head": [(f"{t}s", t) for t in (0.5, 1.0, 2.0)],
    "gain":      [(f"x{g}", g) for g in (0.25, 4.0)],
    "dac":       [(f"{q}q", q) for q in (2, 4, 8)],
}
_DAC = {}


def _sos(btype, fc, order=8):
    from scipy.signal import butter
    return butter(order, fc, btype=btype, fs=SR, output="sos")


def apply_extra(fam, param, y):
    from scipy.signal import sosfiltfilt, iirnotch, filtfilt
    if fam == "lp_hz":
        return sosfiltfilt(_sos("low", param), y).astype(np.float32)
    if fam == "hp_hz":
        return sosfiltfilt(_sos("high", param), y).astype(np.float32)
    if fam == "notch":
        b, a = iirnotch(param, param / 200.0, fs=SR)
        return filtfilt(b, a, y).astype(np.float32)
    if fam == "requant":
        L = 2 ** (param - 1)
        return (np.round(y * L) / L).astype(np.float32)
    if fam == "resample":
        return fit_len(resample(resample(y, SR, param), param, SR), len(y))
    if fam == "pitch":
        import librosa
        return fit_len(librosa.effects.pitch_shift(y, sr=SR, n_steps=param), len(y))
    if fam == "crop_head":
        return y[int(param * SR):].astype(np.float32)
    if fam == "gain":
        return np.clip(y * param, -1, 1).astype(np.float32)
    if fam == "dac":                                   # guarded, API not verified here
        try:
            import torch
            import dac
            if "m" not in _DAC:
                _DAC["m"] = dac.DAC.load(dac.utils.download(model_type="44khz")).eval()
            x = torch.from_numpy(resample(y, SR, 44100)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                z, *_ = _DAC["m"].encode(x, n_quantizers=param)
                out = _DAC["m"].decode(z).squeeze().numpy()
            return fit_len(resample(out, 44100, SR), len(y))
        except Exception:
            return None
    raise KeyError(fam)


def attack_configs(families=None, skip=("dac",)):
    """List of dict(family, label, param, sev). sev = 0 mildest ... n-1 strongest
    within a family (used for 'breaking point')."""
    import vox_attacks as V
    cfgs = []
    for fam, grid in V.VOX_GRID.items():
        items = [(lab, par) for lab, par in grid]
        if fam == "time_stretch":
            order = sorted(range(len(items)), key=lambda i: abs(float(items[i][1]) - 1.0))
        elif fam in _REVERSE:
            order = list(range(len(items)))[::-1]
        else:
            order = list(range(len(items)))
        for rank, i in enumerate(order):
            cfgs.append(dict(family=f"vox_{fam}", label=items[i][0], param=items[i][1], sev=rank, src="vox"))
    for fam, grid in _EXTRA.items():
        if fam == "pitch":
            order = sorted(range(len(grid)), key=lambda i: abs(grid[i][1]))
        elif fam in _REVERSE:
            order = list(range(len(grid)))[::-1]
        else:
            order = list(range(len(grid)))
        for rank, i in enumerate(order):
            cfgs.append(dict(family=fam, label=grid[i][0], param=grid[i][1], sev=rank, src="extra"))
    # libopus rejects 496 kbps in ffmpeg (selftest: no rows); 256k is the practical top.
    cfgs = [c for c in cfgs if not (c["family"] == "vox_opus" and float(c["param"]) > 300)]
    if families:
        cfgs = [c for c in cfgs if c["family"] in families]
    return [c for c in cfgs if c["family"] not in skip] if not families else cfgs


def run_attack(cfg, y22):
    """Returns attacked audio at the master rate, or None if unavailable."""
    if cfg["src"] == "vox":
        import vox_attacks as V
        return V.apply(cfg["family"].replace("vox_", "", 1), cfg["param"], y22, SR)
    return apply_extra(cfg["family"], cfg["param"], y22)


# --------------------------------------------------------------------------- #
#  decision rules
# --------------------------------------------------------------------------- #
def binom_threshold(n, fpr=0.011):
    """Smallest k with P(Binomial(n, 0.5) >= k) <= fpr. The default 1.1% is "about
    1%": it gives 13/16, 16/20 and 9/10 (an exact 1% would force 10/10 for
    Timbre's 10 bits, which is a different, far stricter test)."""
    from scipy.stats import binom
    for k in range(n + 1):
        if binom.sf(k - 1, n, 0.5) <= fpr:
            return k
    return n + 1


def binom_tail(n, k):
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, 0.5))
