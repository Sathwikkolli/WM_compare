"""
A/B test: does simply increasing the push-loss confidence penalty
(spending more of AWARE's ~88%-unused perceptual safety budget on a
stronger signal) improve robustness on short, heavily-cropped clips --
without any structural/architecture change?

Trims the input audio down to a short window matching the scale of
AWARE's own benchmark (~5s LibriSpeech utterances), embeds the SAME
watermark bits with the default push-loss penalty_weight and with a
stronger one, then runs the same truncation crop sweep on both and prints
the resulting confidence/BER curves side by side.

Usage:
    python penalty_weight_ab_test.py audio/family_now_16k.wav [start_sec] [duration_sec]
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
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.embedding.losses import PushToExtremesLoss

N_BITS = 20
SEED = 42
KEEP_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.1]
PENALTY_WEIGHTS = [("baseline", 0.1), ("stronger", 0.5)]


def bit_error_rate(a, b):
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    n = min(len(a), len(b))
    return float(np.mean(a[:n] != b[:n]))


def run_sweep(embedder, detector, signal, sr, watermark_bits, penalty_weight, tag):
    # Override the loss instance in place -- same embedder, same frozen
    # random detector, only the confidence-push strength changes.
    embedder.loss = PushToExtremesLoss(penalty_weight=penalty_weight)

    print(f"\n=== {tag} (penalty_weight={penalty_weight}) ===")
    watermarked = embed_watermark(signal, sr, watermark_bits, embedder).astype("float32")

    print(f"{'keep_ratio':>10s} {'kept_sec':>10s} {'confidence':>11s} {'BER':>8s}")
    results = []
    for keep_ratio in KEEP_RATIOS:
        keep_len = int(round(keep_ratio * len(watermarked)))
        cropped = watermarked[:keep_len]
        pattern, confidence = detect_watermark(cropped, sr, detector)
        ber = bit_error_rate(watermark_bits, pattern)
        results.append((keep_ratio, keep_len / sr, float(confidence), ber))
        print(f"{keep_ratio:10.2f} {keep_len / sr:10.2f} {confidence:11.4f} {ber:8.3f}")
    return results


def main():
    audio_path = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    embedder, detector = load(name="AWARE")

    signal, sr = librosa.load(audio_path, sr=16000, mono=True)
    signal = signal.astype("float32")

    start_sample = int(start_sec * sr)
    end_sample = start_sample + int(duration_sec * sr)
    signal = signal[start_sample:end_sample]
    print(f"Testing on a {len(signal) / sr:.2f}s window "
          f"[{start_sec:.1f}s - {start_sec + duration_sec:.1f}s] of {audio_path} "
          f"(paper-benchmark scale, not the full clip)")

    rng = np.random.RandomState(SEED)
    watermark_bits = rng.randint(0, 2, size=N_BITS).astype(np.int32)

    all_results = {}
    for tag, pw in PENALTY_WEIGHTS:
        all_results[tag] = run_sweep(embedder, detector, signal, sr, watermark_bits, pw, tag)

    print("\n\n=== SIDE-BY-SIDE COMPARISON ===")
    tags = [t for t, _ in PENALTY_WEIGHTS]
    header = f"{'keep_ratio':>10s}"
    for tag, pw in PENALTY_WEIGHTS:
        header += f" | {tag + f'(pw={pw})':>16s} conf     BER"
    print(header)
    for i, keep_ratio in enumerate(KEEP_RATIOS):
        row = f"{keep_ratio:10.2f}"
        for tag in tags:
            _, _, conf, ber = all_results[tag][i]
            row += f" | {conf:20.4f} {ber:8.3f}"
        print(row)

    out_path = audio_path.replace(".wav", f"_penalty_ab_{int(start_sec)}s_results.npz")
    save_kwargs = {}
    for tag in tags:
        save_kwargs[f"{tag}_keep_ratios"] = [r[0] for r in all_results[tag]]
        save_kwargs[f"{tag}_confidence"] = [r[2] for r in all_results[tag]]
        save_kwargs[f"{tag}_ber"] = [r[3] for r in all_results[tag]]
    np.savez(out_path, **save_kwargs)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
