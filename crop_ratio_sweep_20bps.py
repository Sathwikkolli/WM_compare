"""
Same crop-ratio sweep methodology as crop_ratio_sweep.py and
penalty_weight_ab_test.py, but using AWARE's already-built "AWARE(20bps)"
(segments) mode instead of the default "AWARE" (full_length) mode.

This mode chunks embedding into ~1-second segments and does sliding-window
majority-vote detection -- exactly the kind of built-in redundancy our
investigation pointed to as the fix for AWARE's flat-global-average
cropping weakness. We've been testing full_length this whole time; this
tells us how much of the problem is already solved by code that ships in
the repo.

Runs the SAME two tests we've already done for full_length, for a direct
comparison:
  1) crop sweep on the full-length track (mirrors crop_ratio_sweep.py)
  2) crop sweep on a short ~5s window (mirrors penalty_weight_ab_test.py)

Usage:
    python crop_ratio_sweep_20bps.py audio/family_now_16k.wav [start_sec] [duration_sec]
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import numpy as np
import librosa
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

N_BITS = 20
SEED = 42
LONG_KEEP_RATIOS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
SHORT_KEEP_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.1]


def bit_error_rate(a, b):
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] != b[:n]))


def safe_detect(detector, cropped, sr):
    # detect_watermark's "segments" branch has an early-exit path that
    # returns a 3-tuple (empty result) instead of the usual 2-tuple when
    # the cropped audio is too short (< ~1 second) for even one sliding
    # window. Handle both shapes defensively.
    result = detect_watermark(cropped, sr, detector)
    if len(result) == 2:
        pattern, confidence = result
    else:
        pattern, confidence = result[0], result[1]
    return pattern, confidence


def run_sweep(embedder, detector, signal, sr, watermark_bits, keep_ratios, label):
    print(f"\n=== {label} ===")
    print(f"Embedding into a {len(signal) / sr:.2f}s clip...")
    watermarked = embed_watermark(signal, sr, watermark_bits, embedder).astype("float32")

    print(f"{'keep_ratio':>10s} {'kept_sec':>10s} {'confidence':>11s} {'BER':>8s}")
    results = []
    for keep_ratio in keep_ratios:
        keep_len = int(round(keep_ratio * len(watermarked)))
        cropped = watermarked[:keep_len]
        pattern, confidence = safe_detect(detector, cropped, sr)
        ber = bit_error_rate(watermark_bits, pattern)
        results.append((keep_ratio, keep_len / sr, float(confidence), ber))
        print(f"{keep_ratio:10.2f} {keep_len / sr:10.2f} {confidence:11.4f} {ber:8.3f}")
    return results


def main():
    audio_path = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    embedder, detector = load(name="AWARE(20bps)")

    full_signal, sr = librosa.load(audio_path, sr=16000, mono=True)
    full_signal = full_signal.astype("float32")

    rng = np.random.RandomState(SEED)
    watermark_bits = rng.randint(0, 2, size=N_BITS).astype(np.int32)

    long_results = run_sweep(embedder, detector, full_signal, sr, watermark_bits,
                              LONG_KEEP_RATIOS, f"Full-length track ({len(full_signal)/sr:.1f}s)")

    start_sample = int(start_sec * sr)
    end_sample = start_sample + int(duration_sec * sr)
    short_signal = full_signal[start_sample:end_sample]
    short_results = run_sweep(embedder, detector, short_signal, sr, watermark_bits,
                               SHORT_KEEP_RATIOS,
                               f"Short window [{start_sec:.1f}s-{start_sec + duration_sec:.1f}s]")

    out_path = audio_path.replace(".wav", "_20bps_crop_sweep_results.npz")
    np.savez(out_path,
             long_keep_ratios=[r[0] for r in long_results],
             long_confidence=[r[2] for r in long_results],
             long_ber=[r[3] for r in long_results],
             short_keep_ratios=[r[0] for r in short_results],
             short_confidence=[r[2] for r in short_results],
             short_ber=[r[3] for r in short_results])
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
