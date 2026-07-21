"""
fsss/salient.py -- FSSS energy-ratio salient-point extractor.

Faithful implementation of the salient-point detector from:
  H. Malik, A. Khokhar, R. Ansari, "Robust Audio Watermarking using Frequency
  Selective Spread Spectrum Theory," ICASSP 2004 (Section 3), and the IET
  Information Security 2008 journal version.

The paper defines a short-time energy ratio at each sample and marks "fast energy
transition" points, which are grouped and thinned so that ~3-4 salient points per
second survive. Those points act as content-derived synchronization anchors: the
detector re-extracts them after an attack, so embedding tied to them survives
cropping / desynchronization.

This is the Tier-0 baseline for the thesis's salient-point experiments (Exp A
repeatability, Exp C anchor-definition A/B, Exp B rate sweep).

PAPER-FAITHFUL vs FILLED-IN
---------------------------
Stated by the paper and implemented exactly:
  * Eqs 2-4: Er(n) = E_after(n) / E_before(n), with r samples each side.
  * "E_after > Th2" floor to reject silence-to-silence ratio spikes.
  * Group points closer than Th3, keep the strongest per group.
  * "thresholds set adaptively to ensure 3-4 salient points per second."

Left UNSPECIFIED by the paper -> our documented design choices (marked [CHOICE]):
  * r (energy window length): paper gives no value -> `r_ms`, default 10 ms.
  * The exact adaptive rule for Th1: we realize "ensure N/sec" as "keep the top-N
    strongest transitions" (Th1 becomes the Er of the N-th point). Swappable.
  * Th2 form: percentile floor on frame energy (`silence_pct`).
  * Th3 value: fixed minimum spacing (`merge_ms`).

Core depends on numpy only (portable on Great Lakes without an audio stack).
"""

import numpy as np

__all__ = ["find_salient_points"]


def find_salient_points(
    x,
    sr,
    target_rate=3.5,
    r_ms=10.0,
    merge_ms=120.0,
    silence_pct=10.0,
    return_strength=False,
):
    """Extract FSSS energy-ratio salient points from a mono signal.

    Parameters
    ----------
    x : array_like, shape (N,)
        Mono audio samples (float). If a 2-D array is given it is downmixed.
    sr : int
        Sample rate of `x` in Hz. `r` and `merge` are derived from this, so the
        caller must resample to the working rate (e.g. 16 kHz) BEFORE calling.
    target_rate : float
        Desired salient points per second. Paper: 3-4. [STATED]
    r_ms : float
        Energy-window half-length in ms (r = r_ms * sr / 1000 samples). [CHOICE]
    merge_ms : float
        Minimum spacing between salient points in ms (Th3). [CHOICE]
    silence_pct : float
        Percentile of frame energy used as the Th2 silence floor (0-100). Points
        whose E_after is below this percentile are discarded. [CHOICE]
    return_strength : bool
        If True, also return the Er strength at each returned point.

    Returns
    -------
    idx : np.ndarray, shape (M,), int
        Sorted sample indices of the salient points.
    strength : np.ndarray, shape (M,), float
        (only if return_strength) Er value at each point.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:                       # downmix to mono
        x = x.mean(axis=tuple(range(1, x.ndim)))
    n = x.size
    r = max(1, int(round(r_ms * sr / 1000.0)))

    # Degenerate / too-short input: nothing to anchor to.
    if n < 2 * r + 1:
        empty = np.empty(0, dtype=np.int64)
        return (empty, np.empty(0)) if return_strength else empty

    # --- Step 1: energy ratio at every sample (Eqs 2-4), via O(N) prefix sums. --
    # csum[k] = sum_{i<k} x[i]^2 ; energy over [a, b) = csum[b] - csum[a] (O(1)).
    e = x * x
    csum = np.concatenate(([0.0], np.cumsum(e)))
    idx_all = np.arange(r, n - r)                       # valid centers (full windows)
    e_before = csum[idx_all] - csum[idx_all - r]        # sum over [n-r, n)
    e_after = csum[idx_all + r] - csum[idx_all]         # sum over [n, n+r)
    eps = 1e-12
    er = e_after / (e_before + eps)                     # Eq 2

    # --- Step 2: Th2 silence floor -- drop ratio spikes that are silence->silence.
    # [CHOICE] Th2 = percentile of the E_after distribution.
    th2 = np.percentile(e_after, silence_pct)
    keep = e_after > th2
    if not np.any(keep):
        empty = np.empty(0, dtype=np.int64)
        return (empty, np.empty(0)) if return_strength else empty
    cand_pos = idx_all[keep]                            # candidate sample indices
    cand_er = er[keep]                                  # their transition strengths

    # --- Step 4 (target count) drives Step 3 (grouping). ------------------------
    duration_sec = n / float(sr)
    n_target = max(1, int(round(target_rate * duration_sec)))

    # --- Step 3: group points within merge_ms, keep the strongest per group. ----
    # Process candidates strongest-first; accept a point only if it is at least
    # `merge` samples away from every already-accepted point. This realizes both
    # "merge within Th3 -> strongest per group" AND caps the result at n_target,
    # so Th1 is set adaptively to the Er of the weakest accepted point. [CHOICE]
    merge = max(1, int(round(merge_ms * sr / 1000.0)))
    order = np.argsort(-cand_er)                        # descending strength
    accepted_pos = []
    accepted_er = []
    for j in order:
        p = cand_pos[j]
        if all(abs(int(p) - int(q)) >= merge for q in accepted_pos):
            accepted_pos.append(int(p))
            accepted_er.append(float(cand_er[j]))
            if len(accepted_pos) >= n_target:
                break

    out = np.array(sorted(range(len(accepted_pos)), key=lambda k: accepted_pos[k]))
    pos = np.array(accepted_pos, dtype=np.int64)[out] if len(accepted_pos) else np.empty(0, dtype=np.int64)
    strn = np.array(accepted_er, dtype=np.float64)[out] if len(accepted_er) else np.empty(0)
    return (pos, strn) if return_strength else pos


if __name__ == "__main__":
    # Sanity check on the local dev clip: print salient points as timestamps so we
    # can eyeball whether they land on real speech onsets.
    import os, sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "audio", "client_original_16k.wav"
    )
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("soundfile not installed; run:  pip install soundfile")

    y, file_sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    work_sr = 16000
    if file_sr != work_sr:
        try:
            import librosa
            y = librosa.resample(y.astype("float32"), orig_sr=file_sr, target_sr=work_sr)
        except ImportError:
            sys.exit(f"file is {file_sr} Hz; install librosa to resample, or pass a 16 kHz wav")
        sr = work_sr
    else:
        sr = file_sr

    pts, strength = find_salient_points(y, sr, return_strength=True)
    dur = len(y) / sr
    print(f"file      : {os.path.abspath(path)}")
    print(f"duration  : {dur:.2f} s  ({len(y)} samples @ {sr} Hz)")
    print(f"points    : {len(pts)}  ({len(pts) / dur:.2f} /sec)")
    print("  idx        time(s)    Er")
    for p, s in zip(pts, strength):
        print(f"  {p:<9d}  {p / sr:7.3f}   {s:8.2f}")
