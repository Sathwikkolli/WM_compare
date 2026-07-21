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


def _pick_top_n(strength, frame_hop, sr, n_target, offset=0):
    """Greedy strongest-first peak pick from a per-frame strength curve.

    strength   : 1-D array over frames
    frame_hop  : samples per frame (frame f -> sample f*frame_hop + offset)
    Returns sorted int sample indices, min-separated by MIN_SEP_MS, capped at n_target.
    """
    strength = np.asarray(strength, dtype=np.float64)
    if strength.size == 0:
        return np.empty(0, dtype=np.int64)
    min_sep = max(1, int(round(MIN_SEP_MS * sr / 1000.0)))
    order = np.argsort(-strength)
    picked = []
    for f in order:
        s = int(f) * frame_hop + offset
        if all(abs(s - p) >= min_sep for p in picked):
            picked.append(int(s))
            if len(picked) >= n_target:
                break
    return np.array(sorted(picked), dtype=np.int64)


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
def detect_librosa_flux(x, sr, target_rate=3.5):
    import librosa
    hop = 256
    oenv = librosa.onset.onset_strength(y=np.asarray(x, dtype=np.float32), sr=sr,
                                        hop_length=hop)
    return _pick_top_n(oenv, hop, sr, _n_target(x, sr, target_rate))


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
