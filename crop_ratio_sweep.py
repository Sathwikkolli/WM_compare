"""
Tests whether AWARE's real-world segment-removal (SR) degradation is
explained by simple sample-size dilution (BRH averages over fewer frames,
so the average is a noisier estimate of the same underlying signal) rather
than spatial concentration of evidence -- which the last two probes already
argued against (no loud/quiet asymmetry, no strong single-region time
concentration).

Embeds a watermark once, then truncates the SAME watermarked audio to
several lengths (pure truncation from the start -- no splicing -- to
isolate sample-size effects from splice-boundary artifacts) and measures
confidence/BER at each length.

Usage:
    python crop_ratio_sweep.py audio/family_now_16k.wav
"""
import sys
import os

# See probe_budget_utilization.py for why this is needed: the repo's
# uninitialized aware/ submodule checkout shadows the real, pip-installed
# aware package via Python's implicit namespace packages.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import numpy as np
import librosa
import soundfile as sf
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

N_BITS = 20
KEEP_RATIOS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
SEED = 42


def bit_error_rate(a, b):
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    n = min(len(a), len(b))
    return float(np.mean(a[:n] != b[:n]))


def main():
    audio_path = sys.argv[1]
    embedder, detector = load(name="AWARE")

    signal, sr = librosa.load(audio_path, sr=16000, mono=True)
    signal = signal.astype("float32")

    rng = np.random.RandomState(SEED)
    watermark_bits = rng.randint(0, 2, size=N_BITS).astype(np.int32)

    print(f"Embedding {N_BITS}-bit watermark into {audio_path} "
          f"({len(signal) / sr:.1f}s @ {sr}Hz)...")
    watermarked = embed_watermark(signal, sr, watermark_bits, embedder)
    watermarked = watermarked.astype("float32")

    wm_path = audio_path.replace(".wav", "_watermarked.wav")
    sf.write(wm_path, watermarked, sr)
    print(f"Saved watermarked audio to {wm_path}\n")

    print(f"{'keep_ratio':>10s} {'kept_sec':>10s} {'confidence':>11s} {'BER':>8s}")
    results = []
    for keep_ratio in KEEP_RATIOS:
        keep_len = int(round(keep_ratio * len(watermarked)))
        cropped = watermarked[:keep_len]

        pattern, confidence = detect_watermark(cropped, sr, detector)
        ber = bit_error_rate(watermark_bits, pattern)

        results.append((keep_ratio, keep_len / sr, float(confidence), ber))
        print(f"{keep_ratio:10.2f} {keep_len / sr:10.1f} {confidence:11.4f} {ber:8.3f}")

    out_path = audio_path.replace(".wav", "_crop_sweep_results.npz")
    np.savez(out_path,
             keep_ratios=[r[0] for r in results],
             kept_sec=[r[1] for r in results],
             confidence=[r[2] for r in results],
             ber=[r[3] for r in results])
    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    main()
