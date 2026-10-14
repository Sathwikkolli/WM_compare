"""
casp/threshold.py -- CASP-derived tolerance map for AWARE's STFT grid.

Idea (see casp/README.md): AWARE changes every bin by at most eta*|X|, with one
fixed eta (6 dB). Here eta becomes a function of (band, time block), chosen so
that the revised CASP model finds the change "just noticeable":

    D(eta) = || IR(x + eta * e_block) - IR(x) ||_2        (internal representation)

`e_block` is an AWARE-shaped, random-sign perturbation confined to one
band x time block (|delta| = |X| inside the block). For a template equal to the
difference and constant internal noise sigma, d' = D / sigma, so we calibrate
D_JND once from the paper's own fitting condition (1 kHz tone, 60 dB SPL,
intensity increment) and solve D(eta) = target * D_JND / sqrt(n_bands).
The sqrt(n_bands) is an equal-share rule: a real watermark perturbs all bands
at once and independent contributions add in quadrature. `verify` checks it.

STFT grid = AWARE's (n_fft 1024, hop 256, Hann, 16 kHz, 1-4 kHz band), taken
from masking/psy.py so both experiments share one definition.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "masking"))
import psy                                   # noqa: E402  (AWARE's STFT grid + budgets)

SR = psy.SR
BAND = psy.BAND
HOP = psy.HOP
# probe gains: tol 54, 40, 26, 12 dB. Measured on a synthetic clip, D(eta) is linear below
# eta~0.01 and saturates above ~0.06, so the grid must reach down to ~0.002 to bracket
# one JND-unit (the first version started at 0.06 and every cell hit the floor).
ETAS = np.array([0.002, 0.01, 0.05, 0.25])


# --------------------------------------------------------------------------- #
#  geometry: bands x blocks
# --------------------------------------------------------------------------- #
def erb_rate(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def erb_rate_inv(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def band_bins(n_bands=12, band=BAND):
    """Split the embedding band into `n_bands` ERB-equal slices.
    Returns list of (lo_bin, hi_bin_exclusive) and the Hz edges."""
    f = psy.freqs()
    e = np.linspace(erb_rate(band[0]), erb_rate(band[1]), n_bands + 1)
    hz = erb_rate_inv(e)
    inband = np.where((f >= band[0]) & (f <= band[1]))[0]
    edges = np.searchsorted(f, hz)
    edges[0], edges[-1] = inband[0], inband[-1] + 1
    edges = np.maximum.accumulate(edges)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_bands)], hz


def block_frames(n_frames, block=8):
    return [(s, min(s + block, n_frames)) for s in range(0, n_frames, block)]


# --------------------------------------------------------------------------- #
#  perturbations
# --------------------------------------------------------------------------- #
def signs_like(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=shape).astype(np.float32)


def perturbation(X, eta, signs, length):
    """Time-domain change for an STFT-magnitude edit  |X| -> |X| + eta*signs*|X|
    with the original phase kept (what AWARE does). `eta` is (F,U) or scalar;
    zero entries leave the bin alone. Linear in eta, so probes = x + g*e."""
    M = np.abs(X)
    ph = X / np.maximum(M, 1e-12)
    D = (eta * signs * M) * ph
    return psy.istft(D, length).astype(np.float32)


def local_probe(X, signs, f0, f1, u0, u1, length):
    """e_block for (bins f0:f1, frames u0:u1) at eta = 1."""
    eta = np.zeros(X.shape, dtype=np.float32)
    eta[f0:f1, u0:u1] = 1.0
    return perturbation(X, eta, signs, length)


# --------------------------------------------------------------------------- #
#  probing
# --------------------------------------------------------------------------- #
def probe_grid(casp, x_pa, n_bands=12, block=8, etas=ETAS, seed=0,
               pre_s=0.30, post_s=0.10, chunk=36, log=print):
    """x_pa: host in Pascals (already level-calibrated), 16 kHz.
    Returns dict with D[k, b, j], block/band geometry. One host window and
    n_bands*len(etas) probe windows per block, batched through CASP."""
    import torch
    X = psy.stft(x_pa)
    F, U = X.shape
    bands, hz = band_bins(n_bands)
    blocks = block_frames(U, block)
    signs = signs_like(X.shape, seed)
    pre, post = int(pre_s * SR), int(post_s * SR)
    n = len(x_pa)
    xpad = np.concatenate([np.zeros(pre, np.float32), x_pa.astype(np.float32),
                           np.zeros(post, np.float32)])

    D = np.zeros((len(blocks), n_bands, len(etas)), np.float32)
    for k, (u0, u1) in enumerate(blocks):
        s0 = max(0, u0 * HOP - HOP // 2)                      # sample span the block touches
        s1 = min(n, u1 * HOP + HOP // 2 + psy.N_FFT // 2)
        w0, w1 = s0, s1 + pre + post                           # in padded coordinates
        host_w = xpad[w0:w1]
        wins = [host_w]
        for (f0, f1) in bands:
            e = local_probe(X, signs, f0, f1, u0, u1, n)
            e_w = np.concatenate([np.zeros(pre, np.float32), e, np.zeros(post, np.float32)])[w0:w1]
            for g in etas:
                wins.append(host_w + np.float32(g) * e_w)
        W = np.stack(wins)                                     # (1 + nb*G, T)
        outs = []
        for i in range(0, len(W), chunk):
            outs.append(casp.ir(W[i:i + chunk]))
        ir = torch.cat(outs, dim=0)
        d = casp.dist(ir[1:], ir[:1].expand(len(ir) - 1, -1, -1))
        D[k] = d.reshape(n_bands, len(etas)).numpy()
        if log and (k % 5 == 0 or k == len(blocks) - 1):
            log(f"  block {k + 1}/{len(blocks)}")
    return dict(D=D, etas=np.asarray(etas), bands=np.asarray(bands), hz=hz,
                blocks=np.asarray(blocks), n_frames=U)


# --------------------------------------------------------------------------- #
#  from D(eta) to a tolerance map
# --------------------------------------------------------------------------- #
def solve_eta(D, etas, d_target):
    """D: (K, nb, G) increasing-in-eta distances. Returns eta*(K, nb) and
    flags (K, nb): -1 sensitive (even smallest probe too audible -> clipped to
    etas[0]), +1 unconstrained (largest probe still inaudible -> etas[-1])."""
    K, nb, G = D.shape
    le = np.log(etas)
    eta = np.zeros((K, nb), np.float32)
    flag = np.zeros((K, nb), np.int8)
    for k in range(K):
        for b in range(nb):
            d = np.maximum.accumulate(np.maximum(D[k, b], 1e-9))
            if d_target <= d[0]:
                eta[k, b], flag[k, b] = etas[0], -1
            elif d_target >= d[-1]:
                eta[k, b], flag[k, b] = etas[-1], 1
            else:
                eta[k, b] = np.exp(np.interp(np.log(d_target), np.log(d), le))
    return eta, flag


def tol_db(eta):
    return -20 * np.log10(np.maximum(eta, 1e-9))


def eta_map(eta_kb, bands, blocks, shape, smooth=True):
    """(K, nb) -> (F, U) eta per STFT bin, zero outside the band. With
    `smooth`, linear interpolation in dB across block centres (per band)."""
    F, U = shape
    out = np.zeros(shape, np.float32)
    ctr = np.array([(u0 + u1 - 1) / 2 for u0, u1 in blocks])
    u = np.arange(U)
    for b, (f0, f1) in enumerate(bands):
        t = tol_db(eta_kb[:, b])
        if smooth and len(ctr) > 1:
            tb = np.interp(u, ctr, t)
        else:
            tb = np.zeros(U)
            for k, (u0, u1) in enumerate(blocks):
                tb[u0:u1] = t[k]
        out[f0:f1, :] = (10 ** (-tb / 20))[None, :]
    return out


def casp_budget(eta_fu, M):
    """Allowed |delta| per bin, AWARE units: eta(band,time) * |X|."""
    return eta_fu * M


# --------------------------------------------------------------------------- #
#  full-signal audibility check
# --------------------------------------------------------------------------- #
def audibility_blocks(casp, x_pa, ys_pa, blocks, d_jnd):
    """Run the host and each perturbed signal in `ys_pa` (list) through CASP over
    the WHOLE signal, batched. Returns, per signal, A[k] = ||dIR over block k|| /
    D_JND (norm over the block's time span, all channels and modulation filters)
    and the whole-signal total."""
    import torch
    W = np.stack([x_pa] + list(ys_pa)).astype(np.float32)
    ir = casp.ir(W)                                           # (1+n, K, T)
    d = ir[1:] - ir[:1]
    scale = np.sqrt(4000.0 / casp.fs_ir)                      # same normalisation as casp.dist
    ratio = casp.fs_ir / SR
    out = []
    for i in range(d.shape[0]):
        A = np.zeros(len(blocks), np.float32)
        for k, (u0, u1) in enumerate(blocks):
            t0 = int(max(0, (u0 * HOP - HOP // 2)) * ratio)
            t1 = int(min(len(x_pa), u1 * HOP + HOP // 2 + psy.N_FFT // 2) * ratio) + 1
            A[k] = float(d[i, :, t0:t1].pow(2).sum().sqrt()) * scale / d_jnd
        out.append((A, float(d[i].pow(2).sum().sqrt()) * scale / d_jnd))
    return out
