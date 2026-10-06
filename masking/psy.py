"""
masking/psy.py -- psychoacoustic masking threshold on AWARE's STFT grid, plus
AWARE's own budget for comparison.

The model is Johnston's (IEEE JSAC 1988) critical-band model, the textbook
ancestor of MPEG psychoacoustic model 1, computed per STFT frame:

    1. power spectrum -> dB SPL      (full-scale sine = FS_DB, default 96 dB)
    2. energy per critical band       (1-Bark bands)
    3. spread across bands            (Schroeder spreading function)
    4. tonality offset                (per-band spectral flatness: tone 14.5+b dB, noise 5.5 dB)
    5. renormalise the spreading gain, divide band threshold over its bins
    6. temporal masking               (post: slow decay over frames, pre: one frame)
    7. absolute threshold of hearing  (Terhardt) as a floor

Each step after 3 can be switched off (`components`), which is what E7 needs.

Everything is expressed as an AMPLITUDE budget in the same units as AWARE's
STFT magnitude |X| (waveform normalised to max |x| = 1, torch.stft defaults),
so `mask_budget` can replace AWARE's `|X| * 10^(-tol/20)` directly.
"""
import numpy as np

SR = 16000
N_FFT = 1024
HOP = 256
BAND = (1000, 4000)                # AWARE's embedding band
FS_DB = 96.0                       # dB SPL of a full-scale sine (common convention)
SINE_PEAK = N_FFT / 4              # |X| peak of a full-scale sine, periodic Hann

ALL = ("spread", "tonality", "post", "pre", "ath")
DEFAULTS = dict(post_db_per_s=300.0,   # ~60 dB fade over ~200 ms
                pre_db_per_frame=20.0,  # ~1 hop (16 ms) of pre-masking
                sfm_max_db=-30.0)   # per-band flatness; -30 dB counts as fully tonal


def stft(x):
    import librosa
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                        window="hann", center=True, pad_mode="reflect")


def istft(X, length):
    import librosa
    return librosa.istft(X, hop_length=HOP, win_length=N_FFT, n_fft=N_FFT,
                         window="hann", center=True, length=length)


def freqs():
    return np.fft.rfftfreq(N_FFT, 1 / SR)


def band_mask():
    f = freqs()
    return (f >= BAND[0]) & (f <= BAND[1])


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath_db(f):
    k = np.maximum(f, 20.0) / 1000
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


def db_to_amp(db_spl, fs_db=FS_DB):
    """dB SPL per bin -> |X| amplitude in AWARE's units."""
    return SINE_PEAK * 10 ** ((db_spl - fs_db) / 20)


def amp_to_db(a, fs_db=FS_DB):
    return 20 * np.log10(np.maximum(a, 1e-12) / SINE_PEAK) + fs_db


def schroeder(dz):
    """Spreading function in dB; dz = z_maskee - z_masker (Bark)."""
    d = dz + 0.474
    return 15.81 + 7.5 * d - 17.5 * np.sqrt(1 + d ** 2)


def aware_budget(M, tol_db=6.0):
    return M * 10 ** (-tol_db / 20)


def mask_budget(M, components=ALL, fs_db=FS_DB, margin_db=0.0, return_parts=False, **kw):
    """M: |STFT| (F x U) of the max-normalised waveform. Returns the allowed |delta|
    per bin (F x U), same units as M. `margin_db` lowers the threshold (safety)."""
    p = {**DEFAULTS, **kw}
    F, U = M.shape
    f = freqs()
    z = bark(f)
    P = M.astype("float64") ** 2

    # 2. critical bands: integer Bark index per bin
    bidx = np.floor(z).astype(int)
    nb = bidx.max() + 1
    onehot = np.zeros((nb, F))
    onehot[bidx, np.arange(F)] = 1.0
    nbins = onehot.sum(axis=1)
    E = onehot @ P                                         # (nb, U) band energy

    # 3. spreading across bands
    if "spread" in components:
        zc = np.array([z[bidx == b].mean() if nbins[b] else b + 0.5 for b in range(nb)])
        S = 10 ** (schroeder(zc[:, None] - zc[None, :]) / 10)   # [maskee, masker]
        C = S @ E
        gain = S @ np.ones(nb)                             # what a flat spectrum gains
        C = C / gain[:, None]
    else:
        C = E.copy()

    # 4. tonality offset per band and frame: spectral flatness of the bins inside
    #    each critical band (whole-frame flatness calls every speech frame a tone,
    #    because the near-empty bins above 4 kHz sink the geometric mean)
    if "tonality" in components:
        logP = np.log(P + 1e-20)
        gm = np.exp((onehot @ logP) / np.maximum(nbins, 1)[:, None])
        am = E / np.maximum(nbins, 1)[:, None] + 1e-20
        sfm_db = 10 * np.log10(gm / am + 1e-20)
        alpha = np.clip(sfm_db / p["sfm_max_db"], 0, 1)    # (nb, U): 1 = tonal, 0 = noise
        alpha[nbins < 3] = 0.5                             # too few bins to judge
    else:
        alpha = np.zeros((nb, U))
    b = np.arange(nb)[:, None]
    offset_db = alpha * (14.5 + b) + (1 - alpha) * 5.5
    T_band = C * 10 ** (-offset_db / 10)

    # 5. back to bins: band threshold shared evenly by the band's bins
    T = (onehot.T @ (T_band / np.maximum(nbins, 1)[:, None]))   # (F, U) power
    T_db = 10 * np.log10(T + 1e-30) - 20 * np.log10(SINE_PEAK) + fs_db

    # 6. temporal masking, in dB along frames
    if "post" in components:
        step = p["post_db_per_s"] * HOP / SR
        for u in range(1, U):
            T_db[:, u] = np.maximum(T_db[:, u], T_db[:, u - 1] - step)
    if "pre" in components:
        for u in range(U - 2, -1, -1):
            T_db[:, u] = np.maximum(T_db[:, u], T_db[:, u + 1] - p["pre_db_per_frame"])

    # 7. absolute threshold of hearing
    if "ath" in components:
        T_db = np.maximum(T_db, ath_db(f)[:, None])

    B = db_to_amp(T_db - margin_db, fs_db)
    if return_parts:
        return B, dict(alpha=alpha, T_db=T_db)
    return B


def apply_change(x, B, signs, gain=1.0, frames=None):
    """AWARE-style embedding stand-in: |X| + gain*signs*B in the 1-4 kHz band
    (clamped at 0, as AWARE's lower bound is max(0, M - delta)), original phase,
    inverse STFT. `frames` (bool, U) restricts the change to some frames.
    x must be max-normalised. Returns (y, applied_delta)."""
    X = stft(x)
    M, ph = np.abs(X), np.angle(X)
    D = np.zeros_like(M)
    bm = band_mask()
    D[bm] = gain * signs[bm] * B[bm]
    if frames is not None:
        D[:, ~frames] = 0.0
    M2 = np.maximum(M + D, 0.0)
    y = istft(M2 * np.exp(1j * ph), len(x)).astype("float32")
    return y, M2 - M
