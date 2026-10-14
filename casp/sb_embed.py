"""
casp/sb_embed.py -- single-bit spread-spectrum watermark with a pluggable budget.

Embedding (STFT magnitude, phase kept; AWARE's grid: n_fft 1024, hop 256, 1-4 kHz):

    |Y[f,t]| = max( |X[f,t]| + bit * p[f,t] * g * A[f,t] , 0 )

    p  : secret +-1 pattern from the key, one value per band cell
    A  : per-cell amplitude shape of a BUDGET ARM, scaled so that at g = 1 the
         total in-band energy of the change equals AWARE's 6 dB rule
         (sum (0.5|X|)^2), capped at A <= |X|. So all arms are energy-matched at
         the same g and g is a common "how loud" dial.
    g  : strength, 1 = AWARE-6dB energy, 0.1 = -20 dB.

Detection (blind, key only):

    r = log|Y'| - smooth(log|Y'|)         (whitening: removes the host's shape)
    z = sum(p * r) / sqrt(N)  (r standardised)
    null: the same statistic with 200 WRONG keys on the SAME file -> z_std = z / sd(null)
    detected if |z_std| >= 3.3 (two-sided p ~ 1e-3), bit = sign(z).

Per-file wrong-key nulls absorb whatever an attack does to the variance, and the
STFT overlap correlation, so no fixed threshold is assumed.
"""
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threshold as TH                          # noqa: E402  (loads masking/psy)

psy = TH.psy
SR = psy.SR
ARMS = ["aware_shape", "casp", "johnston", "flat_abs"]
Z_DETECT = 3.3
AWARE_ETA = 10 ** (-6.0 / 20)                   # 0.5


def band_idx():
    f = psy.freqs()
    return np.where((f >= psy.BAND[0]) & (f <= psy.BAND[1]))[0]


def pattern(key, shape):
    return np.random.default_rng(int(key)).choice([-1.0, 1.0], size=shape).astype(np.float32)


# --------------------------------------------------------------------------- #
#  budget arms
# --------------------------------------------------------------------------- #
def shape_map(arm, M, eta_casp_fu=None):
    """Relative per-cell amplitude shape W (F, U), zero outside the band."""
    fi = band_idx()
    W = np.zeros_like(M, dtype=np.float64)
    if arm == "aware_shape":                     # constant ratio to the bin's own level
        W[fi] = M[fi]
    elif arm == "casp":                          # CASP tolerance per (band, block) x |X|
        W[fi] = eta_casp_fu[fi] * M[fi]
    elif arm == "johnston":                      # textbook masking threshold (Phase 1)
        W[fi] = psy.mask_budget(M)[fi]
    elif arm == "flat_abs":                      # control: same amplitude in every cell
        W[fi] = 1.0
    else:
        raise ValueError(arm)
    return W


def reference_energy(M):
    fi = band_idx()
    return float(np.sum((AWARE_ETA * M[fi]) ** 2))


def amplitude_map(W, M, E_ref, iters=40):
    """Scale W so the change's energy is E_ref, never exceeding |X| in any cell."""
    s = np.sqrt(E_ref / max(float(np.sum(W ** 2)), 1e-30))
    for _ in range(iters):
        A = np.minimum(s * W, M)
        E = float(np.sum(A ** 2))
        if abs(E / E_ref - 1.0) < 1e-3:
            break
        s *= np.sqrt(E_ref / max(E, 1e-30))
    A = np.minimum(s * W, M)
    E = float(np.sum(A ** 2))
    live = W > 0
    info = dict(E_ratio=E / E_ref, frac_capped=float(np.mean((s * W >= M)[live])) if live.any() else 0.0)
    return A.astype(np.float32), info


# --------------------------------------------------------------------------- #
#  embed / detect
# --------------------------------------------------------------------------- #
def embed(X, A, key, bit, g, n):
    """X: complex STFT of the host (F, U). Returns the marked waveform (n,)."""
    fi = band_idx()
    mag = np.abs(X)
    ph = X / np.maximum(mag, 1e-12)
    p = pattern(key, (len(fi), X.shape[1]))
    new = mag.copy()
    new[fi] = np.maximum(mag[fi] + bit * p * g * A[fi], 0.0)
    return psy.istft(new * ph, n).astype(np.float32)


def whiten(sig, n, smooth=(7, 7)):
    s = np.asarray(sig, np.float32)
    s = s[:n] if len(s) >= n else np.pad(s, (0, n - len(s)))
    fi = band_idx()
    L = np.log(np.abs(psy.stft(s))[fi] + 1e-8)
    r = L - uniform_filter(L, size=smooth, mode="reflect")
    return r / (r.std() + 1e-12)


def detect(sig, key, n, n_null=200, weights=None):
    """Returns dict(z, z_std, null_sd, bit_hat, detected). `weights` (same shape
    as r) switches on an oracle-weighted statistic that needs the embedder's map;
    it is reported separately and is not part of the blind detector."""
    r = whiten(sig, n)
    N = r.size

    def stat(k):
        p = pattern(k, r.shape)
        if weights is None:
            return float((p * r).sum() / np.sqrt(N))
        return float((weights * p * r).sum() / np.sqrt((weights ** 2).sum()))

    z = stat(key)
    nulls = np.array([stat(key + 1 + k) for k in range(n_null)])
    sd = float(nulls.std() + 1e-12)
    zs = z / sd
    return dict(z=z, z_std=zs, null_sd=sd, bit_hat=1 if z >= 0 else -1,
                detected=bool(abs(zs) >= Z_DETECT))


def to_pcm16(x):
    return (np.round(np.clip(x, -1, 1) * 32767) / 32767).astype(np.float32)
