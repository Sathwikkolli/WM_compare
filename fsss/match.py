"""
fsss/match.py -- salient-point alignment + matching + repeatability metrics.

Pure (no I/O). Used by Experiment A (repeatability) and reused by Experiment C
(anchor-definition A/B). Given the salient points of a CLEAN clip and of an
ATTACKED version, it answers: how many clean points survive, how many spurious
points appear, and how far the survivors moved.

The one hard part is TIME ALIGNMENT, because several attacks shift/scale the
timeline (the watermark/anchors are intact, the clock just moved):
  * known warp  (e.g. time_stretch rate)  -> pass `scale`
  * unknown offset (codec padding, strip-silence) -> `align_lag` estimates it
    from an energy-envelope cross-correlation.

Sign convention (verified by the __main__ self-test): a returned `lag` means the
attacked signal is DELAYED by `lag` samples relative to clean, so a clean point
at index i is expected at (i / scale + lag) in the attacked timeline.
"""

import numpy as np

__all__ = ["energy_envelope", "align_lag", "match_points"]


def energy_envelope(x, sr, win_ms=25.0, hop_ms=5.0):
    """Short-time RMS envelope on a hop grid. Returns (env, hop_samples)."""
    x = np.asarray(x, dtype=np.float64)
    win = max(1, int(round(win_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    if x.size < win:
        return np.array([np.sqrt(np.mean(x * x) + 1e-12)]), hop
    n = 1 + (x.size - win) // hop
    env = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        env[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return env, hop


def align_lag(clean, attacked, sr, max_lag_ms=250.0, hop_ms=5.0):
    """Estimate the global time offset (in samples) of `attacked` vs `clean`.

    Cross-correlates the mean-removed energy envelopes and returns the lag, in
    samples, restricted to +/- max_lag_ms. Positive => attacked is delayed.
    """
    ec, hop = energy_envelope(clean, sr, hop_ms=hop_ms)
    ea, _ = energy_envelope(attacked, sr, hop_ms=hop_ms)
    ec = ec - ec.mean()
    ea = ea - ea.mean()
    if ec.size < 2 or ea.size < 2 or np.allclose(ec, 0) or np.allclose(ea, 0):
        return 0
    corr = np.correlate(ea, ec, mode="full")
    # lags[k] for np.correlate(ea, ec, 'full'): ea[n] ~ ec[n - lag]
    lags = np.arange(-(ec.size - 1), ea.size)
    max_frames = max(1, int(round(max_lag_ms / hop_ms)))
    m = (lags >= -max_frames) & (lags <= max_frames)
    if not np.any(m):
        return 0
    best_frame = lags[m][int(np.argmax(corr[m]))]
    return int(best_frame * hop)


def match_points(clean_idx, attacked_idx, sr, w_ms=20.0, scale=1.0, lag=0):
    """Match clean salient points to attacked ones within +/- w_ms.

    Each clean point is projected to its expected attacked position
    (clean_idx / scale + lag), then greedily matched to the nearest unused
    attacked point within the tolerance window.

    Returns a dict: hit_rate, false_alarm, median_jitter_ms, n_clean,
    n_attacked, n_matched.
    """
    clean_idx = np.asarray(clean_idx, dtype=np.float64)
    attacked_idx = np.asarray(attacked_idx, dtype=np.float64)
    n_clean, n_att = clean_idx.size, attacked_idx.size

    if n_clean == 0 or n_att == 0:
        return dict(
            hit_rate=(0.0 if n_clean else float("nan")),
            false_alarm=(1.0 if n_att else float("nan")),
            median_jitter_ms=float("nan"),
            n_clean=n_clean, n_attacked=n_att, n_matched=0,
        )

    predicted = clean_idx / scale + lag
    w = w_ms * sr / 1000.0
    used = np.zeros(n_att, dtype=bool)
    jitter = []
    n_match = 0
    for p in np.sort(predicted):
        d = np.abs(attacked_idx - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= w:
            used[j] = True
            n_match += 1
            jitter.append(attacked_idx[j] - p)

    return dict(
        hit_rate=n_match / n_clean,
        false_alarm=(n_att - n_match) / n_att,
        median_jitter_ms=(float(np.median(np.abs(jitter))) / sr * 1000.0) if jitter else float("nan"),
        n_clean=n_clean, n_attacked=n_att, n_matched=n_match,
    )


if __name__ == "__main__":
    # Self-test: build a clean signal, make a DELAYED + renamed-index copy, and
    # confirm (1) align_lag recovers the delay with the right sign, and
    # (2) match_points scores ~100% once the lag is applied.
    rng = np.random.default_rng(0)
    sr = 16000
    n = sr * 8
    x = (rng.standard_normal(n) * 0.01).astype(np.float64)
    # plant 20 sharp energy bursts (fake onsets)
    true_pts = np.sort(rng.integers(sr // 2, n - sr // 2, size=20))
    for p in true_pts:
        x[p:p + 200] += rng.standard_normal(200) * 0.5

    D = 640  # delay attacked by 40 ms
    att = np.concatenate([np.zeros(D), x])[:n]
    att += (rng.standard_normal(n) * 0.01)  # a little extra noise

    lag = align_lag(x, att, sr)
    print(f"true delay = {D} samples ({D/sr*1000:.1f} ms) | recovered lag = {lag} ({lag/sr*1000:.1f} ms)")

    att_pts = true_pts + D
    naive = match_points(true_pts, att_pts, sr, w_ms=20, lag=0)
    aligned = match_points(true_pts, att_pts, sr, w_ms=20, lag=lag)
    print(f"hit_rate without alignment: {naive['hit_rate']:.2f}")
    print(f"hit_rate with alignment   : {aligned['hit_rate']:.2f}  (expected ~1.00)")
    assert aligned["hit_rate"] > 0.9, "alignment self-test failed"
    print("self-test OK")
