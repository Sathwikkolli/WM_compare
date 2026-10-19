"""Blind multibit detector for the embed.py watermark (no original audio needed).

For every block the detector:
  1. re-estimates the masking threshold from the RECEIVED audio (the watermark
     sits below the mask, so it barely changes it);
  2. whitens the block's spectrum by that threshold (host speech is coloured
     like the mask, the watermark is mask-shaped: dividing by the mask makes
     the host roughly white and turns the watermark back into the plain PN);
  3. correlates with the block's PN spectrum, regenerated from the key.
Block scores are summed per payload bit; the sign of each sum is the bit.

Score: each bit's sum is self-normalised, t_i = z_i / sqrt(sum r_b^2), which is
roughly N(0, 1) without a watermark. S = sum t_i^2 ~ chi2(K) gives a p-value.
"""

import numpy as np

from embed import FRAME_LEN, SR, block_starts, pn_block
from psychoacoustic import masking_threshold, spl_offset_db


def block_scores(received: np.ndarray, key: int, fs: int = SR) -> np.ndarray:
    """Whitened correlation of every block with its PN; one float per block."""
    y = np.asarray(received, dtype=float)
    n = FRAME_LEN
    offset = spl_offset_db(n)
    scores = []
    for b, start in enumerate(block_starts(len(y))):
        frame = y[start:start + n]
        amp = 10 ** ((masking_threshold(frame, fs) - offset) / 20)
        weight = 1.0 / amp
        weight[0] = 0.0                                   # embedder put nothing at DC
        Y = np.fft.rfft(frame) / n
        P = np.fft.rfft(pn_block(key, b)) / np.sqrt(n)
        scores.append(np.sum(np.real(Y * np.conj(P)) * weight))
    return np.asarray(scores)


def decode(received: np.ndarray, key: int, n_bits: int, fs: int = SR):
    """Recover the payload.

    Returns:
        bits:    int array (0/1) of length n_bits.
        t:       per-bit statistic, ~N(0, 1) when there is no watermark.
        p_value: chance of a chi2(n_bits) value this large without a watermark.
    """
    from scipy.stats import chi2

    r = block_scores(received, key, fs)
    if len(r) == 0:
        return np.zeros(n_bits, dtype=int), np.zeros(n_bits), 1.0
    idx = np.arange(len(r)) % n_bits
    z = np.bincount(idx, weights=r, minlength=n_bits)
    energy = np.bincount(idx, weights=r ** 2, minlength=n_bits)
    t = z / np.sqrt(energy + 1e-300)
    p_value = float(chi2.sf(np.sum(t ** 2), n_bits))
    return (z > 0).astype(int), t, p_value
