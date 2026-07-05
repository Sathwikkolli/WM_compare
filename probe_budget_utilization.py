"""
Measures how much of AWARE's per-bin safety budget actually gets used
during embedding, split by whether the original bin was quiet or loud.

Answers: does the optimizer already max out every bin (no free lunch for
crop-robustness fixes), or is there unused headroom concentrated in
low-magnitude (quiet) bins that a crop-augmented optimization could exploit
without touching the perceptual safety budget?

Usage:
    python probe_budget_utilization.py <path_to_16k_mono_wav>
"""
import sys
import numpy as np
import librosa
from aware.utils.models import load
from aware.service import embed_watermark

N_BITS = 20  # matches config_full_length.yaml output_length

captured = {}


def main():
    if len(sys.argv) < 2:
        print("Usage: python probe_budget_utilization.py <path_to_16k_mono_wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    embedder, detector = load(name="AWARE")

    # --- instrument _optimize without touching the library ---
    orig_optimize = embedder._optimize

    def patched_optimize(initial_coeffs, stft_magnitude, watermark_pattern,
                          freq_indices, not_freq_indices, bounds, stft_phase):
        result = orig_optimize(initial_coeffs, stft_magnitude, watermark_pattern,
                                freq_indices, not_freq_indices, bounds, stft_phase)
        captured["initial"] = initial_coeffs.detach().cpu().numpy().copy()
        captured["final"] = result.detach().cpu().numpy().copy()
        captured["bounds"] = bounds
        return result

    embedder._optimize = patched_optimize

    signal, sr = librosa.load(audio_path, sr=16000, mono=True)
    signal = signal.astype("float32")
    watermark_bits = np.random.randint(0, 2, size=N_BITS, dtype=np.int32)

    print(f"Embedding into {audio_path} ({len(signal) / sr:.1f}s @ {sr}Hz), "
          f"{embedder.num_iterations} iterations...")
    embed_watermark(signal, sr, watermark_bits, embedder)

    orig = captured["initial"]
    final = captured["final"]
    delta_used = np.abs(final - orig)
    delta_allowed = np.array([(u - l) / 2.0 for l, u in captured["bounds"]])

    utilization = np.divide(delta_used, delta_allowed,
                             out=np.zeros_like(delta_used), where=delta_allowed > 1e-12)

    q25, q75 = np.percentile(orig, [25, 75])
    quiet_mask = orig <= q25
    loud_mask = orig >= q75

    print(f"\nTotal coefficients: {len(orig)}")
    print(f"Quiet bucket (bottom 25% magnitude): mean utilization = {utilization[quiet_mask].mean():.3f}")
    print(f"Loud  bucket (top 25% magnitude):    mean utilization = {utilization[loud_mask].mean():.3f}")
    print(f"Overall mean utilization:            {utilization.mean():.3f}")
    print(f"Overall utilization std:             {utilization.std():.3f}")
    print(f"Fraction of coeffs with utilization > 0.95: {(utilization > 0.95).mean():.3f}")

    out_path = audio_path + ".utilization.npz"
    np.savez(out_path, orig=orig, final=final,
             delta_allowed=delta_allowed, utilization=utilization)
    print(f"\nSaved raw arrays to {out_path}")


if __name__ == "__main__":
    main()
