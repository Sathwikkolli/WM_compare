"""Sanity tests for the MPEG-masking watermark. No model weights or data needed.

    python test_mpeg.py          (or: python -m pytest test_mpeg.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psychoacoustic as P   # noqa: E402
import embed as E            # noqa: E402
import detect as D           # noqa: E402

FS = 16000


def speechlike(seconds=4.0, seed=0):
    """Voiced harmonics with a slow syllable envelope, pauses and some breath noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    f0 = 120 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / FS
    voiced = sum(np.sin(h * phase) / h for h in range(1, 30) if h * 120 < 7000)
    syll = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 2
    y = voiced * syll + 0.02 * rng.standard_normal(len(t)) * syll
    y[int(1.5 * FS):int(1.9 * FS)] = 0.0                       # a pause
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


def test_full_scale_sine_reads_96_db():
    n = P.FRAME_LEN
    x = np.cos(2 * np.pi * 32 * np.arange(n) / n)              # exactly on bin 32
    assert abs(P.power_spectrum_db(x)[32] - 96.0) < 1e-6


def test_sine_is_a_tonal_masker():
    n = P.FRAME_LEN
    x = 0.5 * np.cos(2 * np.pi * 1000 * np.arange(n) / FS)
    bins, _ = P.find_tonal_maskers(P.power_spectrum_db(x), FS)
    assert 32 in bins


def test_threshold_is_ath_in_silence_and_rises_under_a_tone():
    n = P.FRAME_LEN
    freqs = np.arange(n // 2 + 1) * FS / n
    quiet = P.masking_threshold(np.zeros(n), FS)
    assert np.allclose(quiet, P.absolute_threshold_db(freqs), atol=1e-6)
    tone = P.masking_threshold(0.5 * np.cos(2 * np.pi * 1000 * np.arange(n) / FS), FS)
    assert tone[32] > quiet[32] + 40                           # strong masking at the tone
    assert tone[40] - tone[24] > 0                             # upward spread > downward


def test_roundtrip_recovers_bits_and_is_quiet():
    host = speechlike()
    bits = np.random.default_rng(1).integers(0, 2, 16)
    marked = E.embed_watermark(host, bits, key=1234)
    snr = 10 * np.log10(np.sum(host ** 2) / np.sum((marked - host) ** 2))
    got, t, p = D.decode(marked, key=1234, n_bits=16)
    print(f"  roundtrip: SNR {snr:.1f} dB, bit acc {np.mean(got == bits):.2f}, p {p:.1e}")
    assert np.array_equal(got, bits)
    assert p < 1e-6
    assert 10 < snr < 45


def test_no_watermark_or_wrong_key_gives_no_detection():
    host = speechlike(seed=2)
    bits = np.random.default_rng(3).integers(0, 2, 16)
    marked = E.embed_watermark(host, bits, key=1234)
    _, _, p_clean = D.decode(host, key=1234, n_bits=16)
    _, _, p_wrong = D.decode(marked, key=999, n_bits=16)
    print(f"  null: unmarked p {p_clean:.2f}, wrong key p {p_wrong:.2f}")
    assert p_clean > 1e-3 and p_wrong > 1e-3


def test_survives_mild_noise():
    host = speechlike(seed=4)
    bits = np.random.default_rng(5).integers(0, 2, 16)
    marked = E.embed_watermark(host, bits, key=77)
    noise = np.random.default_rng(6).standard_normal(len(host))
    noisy = marked + noise * np.sqrt(np.mean(host ** 2) / 10 ** (30 / 10))   # 30 dB SNR
    got, _, p = D.decode(noisy, key=77, n_bits=16)
    print(f"  30 dB noise: bit acc {np.mean(got == bits):.2f}, p {p:.1e}")
    assert np.mean(got == bits) >= 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
