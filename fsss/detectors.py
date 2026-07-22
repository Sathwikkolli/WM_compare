"""
fsss/detectors.py -- pluggable salient-point / anchor detectors for Experiment C.

Every detector shares one interface so the repeatability harness can swap them:

    detect(x, sr, target_rate=3.5) -> np.ndarray[int]   # sorted sample indices

For a FAIR comparison, all detectors emit the SAME number of points: the top-N by
their own confidence, where N = round(target_rate * duration). This matches the
thesis use case (a fixed budget of anchor slots for the key-driven hop schedule),
so hit-rate differences reflect anchor QUALITY, not anchor density. Minimum spacing
between anchors is MIN_SEP_MS for every detector.

Each detector raises ImportError (or similar) if its dependency is missing; the
harness treats that detector as unavailable and skips it, so a missing package
never blocks the run.

Detectors:
  fsss         energy-ratio salient points (Malik 2004) -- the baseline
  librosa_flux spectral-flux onset strength
  madmom_cnn   madmom pretrained CNN onset activation
  madmom_rnn   madmom pretrained RNN onset activation
  ssl_wavlm    WavLM embedding novelty (frame-to-frame representation change)
"""

import numpy as np

MIN_SEP_MS = 120.0
__all__ = ["DETECTORS", "detect"]


# --------------------------------------------------------------------------- #
#  shared helpers
# --------------------------------------------------------------------------- #
def _n_target(x, sr, target_rate):
    return max(1, int(round(target_rate * len(x) / float(sr))))


def _pick_top_n(strength, frame_hop, sr, n_target, offset=0, return_strength=False):
    """Greedy strongest-first peak pick from a per-frame strength curve.

    strength   : 1-D array over frames
    frame_hop  : samples per frame (frame f -> sample f*frame_hop + offset)
    Returns sorted int sample indices, min-separated by MIN_SEP_MS, capped at n_target.
    If return_strength, also returns the strength at each picked point (same order).
    """
    strength = np.asarray(strength, dtype=np.float64)
    if strength.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return (empty, np.empty(0)) if return_strength else empty
    min_sep = max(1, int(round(MIN_SEP_MS * sr / 1000.0)))
    order = np.argsort(-strength)
    picked = []          # (sample, strength)
    for f in order:
        s = int(f) * frame_hop + offset
        if all(abs(s - p[0]) >= min_sep for p in picked):
            picked.append((s, float(strength[f])))
            if len(picked) >= n_target:
                break
    picked.sort(key=lambda t: t[0])
    idx = np.array([p[0] for p in picked], dtype=np.int64)
    if return_strength:
        return idx, np.array([p[1] for p in picked], dtype=np.float64)
    return idx


# --------------------------------------------------------------------------- #
#  fsss -- energy-ratio salient points (baseline)
# --------------------------------------------------------------------------- #
def detect_fsss(x, sr, target_rate=3.5):
    try:
        from fsss.salient import find_salient_points
    except ImportError:
        from salient import find_salient_points
    return find_salient_points(x, sr, target_rate=target_rate, merge_ms=MIN_SEP_MS)


# --------------------------------------------------------------------------- #
#  librosa_flux -- spectral-flux onset strength
# --------------------------------------------------------------------------- #
def _librosa_topk(x, sr, k):
    import librosa
    hop = 256
    oenv = librosa.onset.onset_strength(y=np.asarray(x, dtype=np.float32), sr=sr,
                                        hop_length=hop)
    return _pick_top_n(oenv, hop, sr, k, return_strength=True)


def detect_librosa_flux(x, sr, target_rate=3.5):
    idx, _ = _librosa_topk(x, sr, _n_target(x, sr, target_rate))
    return idx


# --------------------------------------------------------------------------- #
#  fusion -- consensus-weighted union of fsss + librosa_flux
# --------------------------------------------------------------------------- #
def _fsss_topk(x, sr, k):
    """Top-k FSSS salient points with Er strengths (rate chosen to yield ~k)."""
    try:
        from fsss.salient import find_salient_points
    except ImportError:
        from salient import find_salient_points
    dur = max(len(x) / float(sr), 1e-6)
    rate = k / dur                                   # find_salient_points caps at rate*dur
    return find_salient_points(x, sr, target_rate=rate, merge_ms=MIN_SEP_MS,
                               return_strength=True)


def _norm01(s):
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.ones_like(s)


def detect_fusion(x, sr, target_rate=3.5):
    """Consensus-weighted fusion of fsss + librosa_flux.

    Each detector proposes 2N candidates; strengths are normalized to [0,1] and
    pooled. Candidates within MIN_SEP_MS are clustered and their strengths SUMMED,
    so a point BOTH detectors find scores ~2x (consensus floats to the top) while
    strong single-detector points still fill the remaining slots (coverage). The
    top-N clusters by summed score are returned.
    """
    n = _n_target(x, sr, target_rate)
    k = 2 * n
    idx_f, s_f = _fsss_topk(x, sr, k)
    idx_l, s_l = _librosa_topk(x, sr, k)
    s_f, s_l = _norm01(s_f), _norm01(s_l)

    pool = list(zip([int(i) for i in idx_f], [float(v) for v in s_f])) \
        + list(zip([int(i) for i in idx_l], [float(v) for v in s_l]))
    pool.sort(key=lambda t: -t[1])                   # strongest first

    min_sep = max(1, int(round(MIN_SEP_MS * sr / 1000.0)))
    clusters = []                                    # [best_pos, summed_score]
    for pos, s in pool:
        hit = None
        for c in clusters:
            if abs(pos - c[0]) < min_sep:
                hit = c
                break
        if hit is None:
            clusters.append([pos, s])                # pool is sorted, so pos is the peak
        else:
            hit[1] += s                              # proximity/agreement boost
    clusters.sort(key=lambda c: -c[1])
    picked = sorted(int(c[0]) for c in clusters[:n])
    return np.array(picked, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  madmom_cnn / madmom_rnn -- pretrained neural onset activations
# --------------------------------------------------------------------------- #
def _madmom_activation(x, sr, which):
    from madmom.audio.signal import Signal
    if which == "cnn":
        from madmom.features.onsets import CNNOnsetProcessor
        proc = CNNOnsetProcessor()
    else:
        from madmom.features.onsets import RNNOnsetProcessor
        proc = RNNOnsetProcessor()
    sig = Signal(np.asarray(x, dtype=np.float32), sample_rate=sr, num_channels=1)
    act = np.asarray(proc(sig), dtype=np.float64)   # activation at 100 fps
    return act


def detect_madmom_cnn(x, sr, target_rate=3.5):
    act = _madmom_activation(x, sr, "cnn")
    hop = int(round(sr / 100.0))                    # madmom activations are 100 fps
    return _pick_top_n(act, hop, sr, _n_target(x, sr, target_rate))


def detect_madmom_rnn(x, sr, target_rate=3.5):
    act = _madmom_activation(x, sr, "rnn")
    hop = int(round(sr / 100.0))
    return _pick_top_n(act, hop, sr, _n_target(x, sr, target_rate))


# --------------------------------------------------------------------------- #
#  ssl_wavlm -- WavLM embedding novelty
# --------------------------------------------------------------------------- #
_WAVLM = {}


def _wavlm_model():
    if "model" not in _WAVLM:
        import torch
        from transformers import WavLMModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        _WAVLM["model"] = model
        _WAVLM["device"] = device
    return _WAVLM["model"], _WAVLM["device"]


def detect_ssl_wavlm(x, sr, target_rate=3.5):
    import torch
    model, device = _wavlm_model()
    xf = np.asarray(x, dtype=np.float32)
    xf = xf - xf.mean()
    denom = xf.std() + 1e-8
    xf = xf / denom                                 # zero-mean / unit-var (WavLM norm)
    with torch.no_grad():
        inp = torch.from_numpy(xf).float().unsqueeze(0).to(device)
        hs = model(inp).last_hidden_state[0]        # (T, D), WavLM stride 320 (20 ms)
        hs = torch.nn.functional.normalize(hs, dim=-1)
        cos = (hs[1:] * hs[:-1]).sum(dim=-1)        # cosine sim of consecutive frames
        novelty = (1.0 - cos).cpu().numpy()         # high = big representation change
    novelty = np.concatenate([[0.0], novelty])      # align length to T
    hop = 320                                       # WavLM frame stride at 16 kHz
    return _pick_top_n(novelty, hop, sr, _n_target(x, sr, target_rate))


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #
DETECTORS = {
    "fsss": detect_fsss,
    "librosa_flux": detect_librosa_flux,
    "fusion": detect_fusion,
    "madmom_cnn": detect_madmom_cnn,
    "madmom_rnn": detect_madmom_rnn,
    "ssl_wavlm": detect_ssl_wavlm,
}


def detect(name, x, sr, target_rate=3.5):
    return DETECTORS[name](x, sr, target_rate=target_rate)


if __name__ == "__main__":
    # quick availability probe on 2 s of noise
    import sys
    sr = 16000
    y = (np.random.default_rng(0).standard_normal(2 * sr) * 0.1).astype("float32")
    for name, fn in DETECTORS.items():
        try:
            pts = fn(y, sr)
            print(f"{name:14s} OK   ({len(pts)} pts)")
        except Exception as e:
            print(f"{name:14s} SKIP ({type(e).__name__}: {e})")
