"""
spots/contexts.py -- what was happening in the speech at each probe.

Every MPEG block gets one CONTEXT label, from the original audio only:

    onset         level jumps > ONSET_DB inside the block (pre-echo risk)
    long_pause    quiet (< PAUSE_DB re clip peak) for > LONG_PAUSE_S around it
    short_gap     quiet dip < GAP_MAX_S long with loud speech on both sides (forward + backward masking)
    tail          within TAIL_S after loud speech, now quieter (forward masking)
    voiced        strong pitch periodicity, speech level
    fricative     weak periodicity, energy mostly above 3 kHz
    decay         speech level, falling
    other         none of the above

and each band of a block gets a BAND label relative to the block's own spectrum:
    peak    band energy within PEAK_DB of the block's strongest band (e.g. a formant)
    above   band starts above the frequency holding 90% of the block's energy
    valley  everything else (between peaks)
Thresholds are module constants so they can be changed and reported.
"""
import numpy as np

FS, N, HOP = 16000, 512, 256
LOUD_DB = -20.0          # "loud speech": within 20 dB of the clip's loudest block
SPEECH_DB = -30.0        # "speech level"
PAUSE_DB = -40.0         # "quiet"
LONG_PAUSE_S = 0.15
GAP_MAX_S = 0.08
TAIL_S = 0.20
ONSET_DB = 10.0
VOICED_R = 0.5           # normalised autocorrelation for voicing
HF_RATIO = 0.5           # share of energy above 3 kHz for fricatives
PEAK_DB = 6.0


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def block_features(x):
    """Per block: level (dB re clip peak), within-block rise, voicing, HF ratio."""
    x = np.asarray(x, dtype=float)
    starts = np.arange(0, len(x) - N + 1, HOP)
    lvl = np.array([_db(np.mean(x[s:s + N] ** 2)) for s in starts])
    lvl -= lvl.max()
    rise = np.array([_db(np.mean(x[s + N // 2:s + N] ** 2)) - _db(np.mean(x[s:s + N // 2] ** 2)) for s in starts])
    lags = np.arange(int(FS / 400), int(FS / 60) + 1)          # pitch 60-400 Hz
    win = np.hanning(N)
    f = np.fft.rfftfreq(N, 1 / FS)
    voicing, hf = np.zeros(len(starts)), np.zeros(len(starts))
    spectra = []
    for i, s in enumerate(starts):
        fr = x[s:s + N] - x[s:s + N].mean()
        ac = np.correlate(fr, fr, "full")[N - 1:]
        voicing[i] = ac[lags].max() / ac[0] if ac[0] > 0 else 0.0
        p = np.abs(np.fft.rfft(fr * win)) ** 2
        spectra.append(p)
        hf[i] = p[f >= 3000].sum() / max(p.sum(), 1e-30)
    return dict(starts=starts, level=lvl, rise=rise, voicing=voicing, hf=hf,
                spectra=np.array(spectra), freqs=f)


def block_contexts(feat):
    """One context label per block (see module docstring)."""
    lvl, n = feat["level"], len(feat["level"])
    hop_s = HOP / FS
    loud = lvl > LOUD_DB
    quiet = lvl < PAUSE_DB
    labels = []
    for i in range(n):
        prev_loud = [j for j in range(i - 1, -1, -1) if loud[j]]
        next_loud = [j for j in range(i + 1, n) if loud[j]]
        since = (i - prev_loud[0]) * hop_s if prev_loud else np.inf
        until = (next_loud[0] - i) * hop_s if next_loud else np.inf
        # length of the quiet-ish run (below speech level) this block sits in
        lo, hi = i, i
        while lo > 0 and lvl[lo - 1] < SPEECH_DB:
            lo -= 1
        while hi < n - 1 and lvl[hi + 1] < SPEECH_DB:
            hi += 1
        run_s = (hi - lo + 1) * hop_s if lvl[i] < SPEECH_DB else 0.0
        if feat["rise"][i] > ONSET_DB and lvl[i] > SPEECH_DB:
            lab = "onset"
        elif quiet[i] and run_s > LONG_PAUSE_S:
            lab = "long_pause"
        elif lvl[i] < SPEECH_DB and 0 < run_s <= GAP_MAX_S and since <= GAP_MAX_S and until <= GAP_MAX_S:
            lab = "short_gap"
        elif not loud[i] and since <= TAIL_S and feat["rise"][i] <= 0:
            lab = "tail"
        elif lvl[i] >= SPEECH_DB and feat["voicing"][i] >= VOICED_R:
            lab = "voiced"
        elif lvl[i] >= SPEECH_DB and feat["hf"][i] > HF_RATIO:
            lab = "fricative"
        elif lvl[i] >= SPEECH_DB and i > 1 and lvl[i] < lvl[i - 2] - 3:
            lab = "decay"
        else:
            lab = "other"
        labels.append(lab)
    return labels


def band_label(feat, i, band):
    """peak / valley / above for one band (lo, hi) in block i."""
    p, f = feat["spectra"][i], feat["freqs"]
    e_band = p[(f >= band[0]) & (f < band[1])].sum()
    c = np.cumsum(p) / max(p.sum(), 1e-30)
    f90 = f[np.searchsorted(c, 0.9)] if p.sum() > 0 else 0.0
    if band[0] >= f90:
        return "above"
    return "peak" if _db(e_band) >= _db(band_max(feat, i)) - PEAK_DB else "valley"


def band_max(feat, i, n_bands=12):
    """Energy of the block's strongest ERB band (used by band_label)."""
    import probes as P
    p, f = feat["spectra"][i], feat["freqs"]
    e = P.band_edges(n_bands)
    return max(p[(f >= e[k]) & (f < e[k + 1])].sum() for k in range(n_bands))
