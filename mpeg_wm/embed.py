"""Watermark embedding (Swanson et al.): secret noise, shaped to hide under the mask.

Multibit, blind-detectable variant:
  * the clip is cut into 512-sample blocks with 50% overlap (hop 256);
  * block b gets its own PN sequence from (key, b), shaped to that block's
    MPEG model-1 masking threshold, faded by the temporal masking envelope,
    and multiplied by +1 / -1 for payload bit (b mod n_bits);
  * blocks are overlap-added with a periodic Hann window (sums to 1 at 50% overlap).
"""

import numpy as np

from psychoacoustic import FRAME_LEN, masking_threshold, spl_offset_db, temporal_envelope

SR = 16000
HOP = FRAME_LEN // 2


def pn_block(key: int, block_index: int, n: int = FRAME_LEN) -> np.ndarray:
    """Pseudo-random +-1 noise for one block, reproducible from (key, block_index).

    Anyone with the same key regenerates exactly the same sequence; anyone
    without it sees white noise.
    """
    rng = np.random.default_rng([key, block_index])
    return rng.choice([-1.0, 1.0], size=n)


def shape_to_mask(pn: np.ndarray, threshold_db: np.ndarray,
                  strength_db: float = 0.0) -> np.ndarray:
    """Give white PN noise the spectrum of the masking threshold.

    Args:
        pn: white noise block, length N (from pn_block).
        threshold_db: masking threshold for this block, length N//2 + 1, dB SPL.
        strength_db: shift the whole watermark up (+) or down (-) relative to
            the threshold. Negative = safety margin, positive = more robust.

    Returns:
        Time-domain watermark block, length N, whose per-bin level follows
        threshold_db (on the same dB SPL scale as power_spectrum_db).
    """
    n = len(pn)

    # Undo the dB SPL calibration: threshold (dB SPL) -> target |X(k)| / N.
    target_amp = 10 ** ((threshold_db + strength_db - spl_offset_db(n)) / 20)
    target_amp[0] = 0.0   # no DC offset: it is not sound and just eats headroom

    # FFT of +-1 noise has random phases and |P(k)| ~ sqrt(N) on average.
    # Dividing by sqrt(N) gives each bin unit average magnitude, so after the
    # multiply below each bin's magnitude follows target_amp, phase stays random.
    spectrum = np.fft.rfft(pn) / np.sqrt(n)

    # Back to time. The * n undoes the /N convention used in power_spectrum_db.
    return np.fft.irfft(spectrum * target_amp * n, n)


def block_starts(n_samples: int, n: int = FRAME_LEN, hop: int = HOP) -> range:
    """Start sample of every full block. Shared by embedder and detector."""
    return range(0, n_samples - n + 1, hop)


def embed_watermark(host: np.ndarray, bits, key: int, fs: int = SR,
                    strength_db: float = 0.0) -> np.ndarray:
    """Embed a multibit payload into a mono clip.

    Args:
        host: mono float clip in [-1, 1] at fs (16 kHz expected).
        bits: payload, sequence of 0/1. Bit i rides on blocks i, i+K, i+2K, ...
        key: secret integer; the detector needs the same one.
        strength_db: watermark level relative to the masking threshold.

    Returns:
        Watermarked clip, same length as host, float32.
    """
    host = np.asarray(host, dtype=float)
    signs = 2.0 * np.asarray(bits, dtype=float) - 1.0       # 0/1 -> -1/+1
    env = temporal_envelope(host, fs)
    synth = np.hanning(FRAME_LEN + 1)[:-1]                  # periodic Hann: OLA sums to 1
    wm = np.zeros_like(host)

    for b, start in enumerate(block_starts(len(host))):
        seg = slice(start, start + FRAME_LEN)
        threshold = masking_threshold(host[seg], fs)
        w = shape_to_mask(pn_block(key, b), threshold, strength_db)

        # Temporal masking: fade the block where the audio is quiet inside it
        # (the frequency mask describes the block's loudest part).
        e = env[seg]
        w *= synth * (e / max(e.max(), 1e-9))

        wm[seg] += signs[b % len(signs)] * w

    return (host + wm).astype(np.float32)
