"""Psychoacoustic model used by Swanson et al. (MPEG-1 Layer I, model 1).

Pipeline (one 512-sample block at a time):
  1. power_spectrum_db   -> how loud is each frequency bin?        (this step)
  2. find maskers        -> which bins are tonal / noise maskers?
  3. masking threshold   -> how loud can hidden noise be per bin?
"""

import numpy as np

FRAME_LEN = 512          # samples per block (32 ms at 16 kHz)
FULL_SCALE_DB = 96.0     # a full-scale sine (amplitude 1.0) is defined as 96 dB SPL


def spl_offset_db(n: int = FRAME_LEN) -> float:
    """dB to add to 10*log10(|FFT/N|^2) so a full-scale sine reads 96 dB SPL."""
    window = np.hanning(n + 1)[:-1]
    # For a sine of amplitude A at a bin centre, the peak |X/N| = A * sum(w) / (2N).
    return FULL_SCALE_DB - 20 * np.log10(window.sum() / (2 * n))


def power_spectrum_db(frame: np.ndarray) -> np.ndarray:
    """Power spectral density of one block, calibrated to dB SPL.

    Args:
        frame: 1-D float array of length N, samples in [-1, 1].

    Returns:
        Array of length N//2 + 1 (bins 0 .. Nyquist) in dB SPL.
    """
    n = len(frame)

    # Hann window: tapers the block edges so the FFT doesn't see a hard cut
    # (a hard cut would smear energy across all bins = "spectral leakage").
    window = np.hanning(n + 1)[:-1]  # periodic Hann (better for FFT frames)

    # Real FFT -> only non-negative frequencies. Divide by N so the magnitude
    # does not depend on block length.
    spectrum = np.fft.rfft(frame * window) / n
    power = np.abs(spectrum) ** 2

    # Calibrate so a full-scale sine lands at exactly 96 dB SPL.
    # Floor avoids log10(0) = -inf on digital silence. It must stay far below
    # hearing AFTER calibration: 1e-20 -> about -92 dB SPL. (1e-12 would read
    # -12 dB SPL, and summed over a wide critical band it beats the threshold in
    # quiet near 3-4 kHz, so silence would create fake noise maskers.)
    return 10 * np.log10(power + 1e-20) + spl_offset_db(n)


TONAL_MARGIN_DB = 7.0    # a peak must stand this far above its neighbourhood


def _tonal_delta(freq_hz: float) -> int:
    """How many bins either side a tonal peak must dominate (MPEG table, in Hz).

    The ear's resolution gets coarser with frequency, so the neighbourhood grows.
    """
    if freq_hz < 5500:
        return 2
    if freq_hz < 11000:
        return 3
    return 6


def find_tonal_maskers(psd_db: np.ndarray, fs: int, n_fft: int = FRAME_LEN):
    """Find sharp spectral peaks ("tones") that will mask nearby frequencies.

    Args:
        psd_db: output of power_spectrum_db (length n_fft//2 + 1).
        fs: sample rate in Hz.
        n_fft: block length used for the FFT.

    Returns:
        bins:     int array, bin index of each tonal masker.
        power_db: float array, masker power (peak + its 2 neighbours combined).
    """
    freqs = np.arange(len(psd_db)) * fs / n_fft
    bins, powers = [], []

    for k in range(2, len(psd_db) - 1):
        p = psd_db[k]

        # Rule 1: it must be a local maximum.
        if not (p > psd_db[k - 1] and p >= psd_db[k + 1]):
            continue

        # Rule 2: it must tower >= 7 dB over bins further out. How far out we
        # look grows with frequency, because the ear's filters get wider.
        delta = _tonal_delta(freqs[k])
        lo, hi = max(k - delta, 0), min(k + delta, len(psd_db) - 1)
        # bins k-delta..k-2 and k+2..k+delta (the +-1 neighbours are part of the peak)
        neighbours = np.r_[psd_db[lo:k - 1], psd_db[k + 2:hi + 1]]
        if np.any(p - neighbours < TONAL_MARGIN_DB):
            continue

        # A windowed sine spreads over ~3 bins, so the tone's true power is the
        # sum of k-1, k, k+1. Add in linear power, then convert back to dB.
        bins.append(k)
        powers.append(10 * np.log10(np.sum(10 ** (psd_db[k - 1:k + 2] / 10))))

    return np.array(bins, dtype=int), np.array(powers, dtype=float)


def hz_to_bark(freq_hz):
    """Convert frequency in Hz to the Bark (critical-band) scale (Zwicker)."""
    f = np.asarray(freq_hz, dtype=float)
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def find_noise_maskers(psd_db: np.ndarray, tonal_bins: np.ndarray, fs: int,
                       n_fft: int = FRAME_LEN):
    """Collapse all non-tonal energy in each critical band into one noise masker.

    Args:
        psd_db: output of power_spectrum_db.
        tonal_bins: bins returned by find_tonal_maskers.
        fs: sample rate in Hz.
        n_fft: block length used for the FFT.

    Returns:
        bins:     int array, one representative bin per critical band.
        power_db: float array, total non-tonal power in that band.
    """
    n_bins = len(psd_db)
    freqs = np.arange(n_bins) * fs / n_fft
    band_of_bin = np.floor(hz_to_bark(freqs)).astype(int)   # 0, 0, 1, 1, 1, 2, ...

    # Bins already "owned" by a tone: the peak and its +-delta neighbourhood.
    # Bin 0 (DC offset) is not sound, so it is excluded too.
    claimed = np.zeros(n_bins, dtype=bool)
    claimed[0] = True
    for k in tonal_bins:
        delta = _tonal_delta(freqs[k])
        claimed[max(k - delta, 0):k + delta + 1] = True

    bins, powers = [], []
    for band in np.unique(band_of_bin):
        in_band = np.flatnonzero(band_of_bin == band)
        free = in_band[~claimed[in_band]]
        if len(free) == 0:
            continue

        # Sum the leftover energy in linear power, then back to dB.
        power = 10 * np.log10(np.sum(10 ** (psd_db[free] / 10)))

        # Place the masker at the geometric mean of the band's bins: the ear
        # hears frequency logarithmically, so this is the band's perceptual centre.
        lo, hi = max(in_band[0], 1), in_band[-1]
        centre = int(round(np.exp(np.mean(np.log(np.arange(lo, hi + 1))))))

        bins.append(centre)
        powers.append(power)

    return np.array(bins, dtype=int), np.array(powers, dtype=float)


def absolute_threshold_db(freq_hz):
    """Threshold in quiet (Terhardt): quietest audible level per frequency, dB SPL."""
    # Clamp at 20 Hz: the formula blows up at 0 Hz, and nobody hears below 20 Hz.
    f_khz = np.maximum(np.asarray(freq_hz, dtype=float), 20.0) / 1000
    return (3.64 * f_khz ** -0.8
            - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
            + 1e-3 * f_khz ** 4)


MERGE_DISTANCE_BARK = 0.5    # maskers closer than this compete; the stronger wins


def decimate_maskers(tonal_bins, tonal_db, noise_bins, noise_db, fs: int,
                     n_fft: int = FRAME_LEN):
    """Drop inaudible maskers, then merge maskers that sit too close together.

    Returns:
        bins:     int array of surviving masker bins, sorted low -> high.
        power_db: float array of their powers.
        is_tonal: bool array, True for tonal maskers, False for noise maskers.
    """
    # Pool both kinds into one list, remembering which kind each one is
    # (the next step treats tonal and noise maskers differently).
    bins = np.concatenate([tonal_bins, noise_bins]).astype(int)
    power = np.concatenate([tonal_db, noise_db]).astype(float)
    is_tonal = np.concatenate([np.ones(len(tonal_bins), dtype=bool),
                               np.zeros(len(noise_bins), dtype=bool)])

    # Rule 1: a masker you cannot hear cannot hide anything.
    audible = power >= absolute_threshold_db(bins * fs / n_fft)
    bins, power, is_tonal = bins[audible], power[audible], is_tonal[audible]

    # Rule 2: walk maskers from low to high frequency; if one lands within
    # 0.5 Bark of the last one we kept, only the stronger of the two survives.
    order = np.argsort(bins, kind="stable")
    bins, power, is_tonal = bins[order], power[order], is_tonal[order]
    barks = hz_to_bark(bins * fs / n_fft)

    keep = []
    for i in range(len(bins)):
        if keep and barks[i] - barks[keep[-1]] < MERGE_DISTANCE_BARK:
            if power[i] > power[keep[-1]]:
                keep[-1] = i          # newcomer is stronger: it replaces the old one
        else:
            keep.append(i)            # far enough away: a separate masker

    keep = np.array(keep, dtype=int)
    return bins[keep], power[keep], is_tonal[keep]


def spreading_function_db(dz, masker_db):
    """How much masking a masker casts at a Bark distance dz from itself (MPEG model 1).

    dz = z_maskee - z_masker. Negative = below the masker, positive = above.
    Returns dB relative to the masker (0 at dz = 0, more negative further away).
    Outside [-3, 8) Bark there is no masking at all (-inf dB).
    """
    dz = np.asarray(dz, dtype=float)
    conditions = [
        (dz >= -3) & (dz < -1),
        (dz >= -1) & (dz < 0),
        (dz >= 0) & (dz < 1),
        (dz >= 1) & (dz < 8),
    ]
    # The slopes depend on the masker's level: louder maskers spread further upward.
    choices = [
        17 * dz - 0.4 * masker_db + 11,
        (0.4 * masker_db + 6) * dz,
        -17 * dz,
        (0.15 * masker_db - 17) * dz - 0.15 * masker_db,
    ]
    return np.select(conditions, choices, default=-np.inf)


def global_masking_threshold(masker_bins, masker_db, is_tonal, fs: int,
                             n_fft: int = FRAME_LEN) -> np.ndarray:
    """Combine every masker's tent + the threshold in quiet into one curve.

    Returns:
        Array of length n_fft//2 + 1: the loudest inaudible noise per bin, dB SPL.
    """
    n_bins = n_fft // 2 + 1
    freqs = np.arange(n_bins) * fs / n_fft
    z_bins = hz_to_bark(freqs)
    z_maskers = z_bins[masker_bins]

    # Grid of distances: row = bin being masked, column = masker.
    dz = z_bins[:, None] - z_maskers[None, :]            # shape (n_bins, n_maskers)
    spread = spreading_function_db(dz, masker_db[None, :])

    # Tones hide less than noise: bigger offset below the masker, growing with z.
    slope = np.where(is_tonal, 0.275, 0.175)
    const = np.where(is_tonal, 6.025, 2.025)
    individual_db = masker_db - slope * z_maskers - const + spread

    # Masking adds up in linear power, not in dB. 10**(-inf/10) = 0, so bins
    # outside a masker's tent get nothing from it.
    total = (10 ** (absolute_threshold_db(freqs) / 10)
             + np.sum(10 ** (individual_db / 10), axis=1))
    return 10 * np.log10(total)


def masking_threshold(frame: np.ndarray, fs: int) -> np.ndarray:
    """Full model-1 pipeline for one block: samples -> allowed noise level per bin."""
    n_fft = len(frame)
    psd = power_spectrum_db(frame)
    tm_bins, tm_db = find_tonal_maskers(psd, fs, n_fft)
    nm_bins, nm_db = find_noise_maskers(psd, tm_bins, fs, n_fft)
    bins, power, is_tonal = decimate_maskers(tm_bins, tm_db, nm_bins, nm_db, fs, n_fft)
    return global_masking_threshold(bins, power, is_tonal, fs, n_fft)


# --- Temporal masking ---------------------------------------------------------

POST_MASK_MS = 50.0   # after a loud sound, the ear stays "deafened" for ~50-200 ms
PRE_MASK_MS = 2.0     # before a loud sound, masking reaches back only ~5 ms


def _peak_follower(x_abs: np.ndarray, alpha: float) -> np.ndarray:
    """Jump up instantly to any new peak, otherwise decay by alpha per sample."""
    env = np.empty_like(x_abs)
    level = 0.0
    for n, v in enumerate(x_abs):
        level = max(v, alpha * level)
        env[n] = level
    return env


def temporal_envelope(signal: np.ndarray, fs: int,
                      post_ms: float = POST_MASK_MS,
                      pre_ms: float = PRE_MASK_MS) -> np.ndarray:
    """Per-sample masking envelope: how loud the signal 'feels' at each instant.

    Models post-masking (slow exponential decay after each peak, run forward in
    time) and pre-masking (fast decay, run backward in time), and keeps the larger.

    Returns:
        Array, same length as signal, in linear amplitude (same units as samples).
    """
    x_abs = np.abs(np.asarray(signal, dtype=float))

    # Per-sample decay factor for time constant tau: alpha = exp(-1 / (tau * fs)).
    alpha_post = np.exp(-1000.0 / (post_ms * fs))
    alpha_pre = np.exp(-1000.0 / (pre_ms * fs))

    forward = _peak_follower(x_abs, alpha_post)                 # masking after peaks
    backward = _peak_follower(x_abs[::-1], alpha_pre)[::-1]     # masking before peaks
    return np.maximum(forward, backward)
