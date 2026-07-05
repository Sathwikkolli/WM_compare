"""
Follow-up analysis (no re-embedding needed): reshapes the flat utilization
array already saved by probe_budget_utilization.py back into
(frequency_bin, time_frame), to find WHERE the sparse ~5% high-utilization
coefficients actually concentrate -- along frequency, along time, or neither.

Usage:
    python analyze_utilization_2d.py audio/family_now_16k.wav.utilization.npz
"""
import sys
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_FFT = 1024
EMBEDDING_BANDS = (1000, 4000)


def main():
    npz_path = sys.argv[1]
    data = np.load(npz_path)
    utilization = data["utilization"]

    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    mask = (freqs >= EMBEDDING_BANDS[0]) & (freqs <= EMBEDDING_BANDS[1])
    n_freq_bins = int(mask.sum())
    n_frames = utilization.size // n_freq_bins
    assert n_freq_bins * n_frames == utilization.size, (
        f"shape mismatch: {n_freq_bins} x {n_frames} != {utilization.size} "
        "-- check SAMPLE_RATE/N_FFT/EMBEDDING_BANDS match the original run"
    )

    util_2d = utilization.reshape(n_freq_bins, n_frames)

    per_freq = util_2d.mean(axis=1)   # collapse time -> is the hot tail frequency-specific?
    per_time = util_2d.mean(axis=0)   # collapse frequency -> is the hot tail time-localized?

    print(f"Shape: {n_freq_bins} freq bins x {n_frames} time frames")

    print(f"\nPer-frequency-bin utilization: min={per_freq.min():.3f} "
          f"max={per_freq.max():.3f} std={per_freq.std():.3f}")
    top_freq_idx = np.argsort(per_freq)[-5:][::-1]
    print(f"Top 5 hottest frequency bins (index within band): {top_freq_idx.tolist()}")
    print(f"Their mean utilization: {[round(v, 3) for v in per_freq[top_freq_idx]]}")

    print(f"\nPer-time-frame utilization: min={per_time.min():.3f} "
          f"max={per_time.max():.3f} std={per_time.std():.3f}")
    decile_means = [per_time[i * n_frames // 10:(i + 1) * n_frames // 10].mean() for i in range(10)]
    print("Per-decile (time) mean utilization across the clip:")
    for i, m in enumerate(decile_means):
        print(f"  {i*10:3d}-{(i+1)*10:3d}%: {m:.4f}")

    out_path = npz_path.replace(".npz", ".2d_breakdown.npz")
    np.savez(out_path, per_freq=per_freq, per_time=per_time, util_2d=util_2d)
    print(f"\nSaved 2D breakdown to {out_path}")


if __name__ == "__main__":
    main()
