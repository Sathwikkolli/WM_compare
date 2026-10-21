"""
spots/probes.py -- the probe and the two judges for the CASP-vs-MPEG headroom experiment.

A probe is ONE block (512 samples, 32 ms at 16 kHz) of the deployed MPEG watermark:
keyed PN noise shaped to that block's MPEG masking threshold, Hann-windowed, and
weighted by MPEG's temporal envelope, exactly as embed.embed_watermark builds each
block (optionally restricted to one frequency band). By construction MPEG rates every
probe "at its limit".

CASP's verdict on a probe is d' (JND units): the internal-representation distance
between the host segment with and without the probe, with PRE_S of context before so
CASP's adaptation memory is right, divided by the calibration distance.

Headroom H (dB) for a cell: probe it at LEVELS_DB relative to MPEG's level (3 random
PN seeds each), fit 20*log10(d') against level with a straight line, and solve for the
level where d' equals the clip's reference d' (median of all full-band probes at 0 dB).
H > 0: CASP says MPEG leaves room unused. H < 0: CASP says MPEG already allows too much.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MPEG = os.path.dirname(HERE)
if MPEG not in sys.path:
    sys.path.insert(0, MPEG)
import casp_model as CM                                    # noqa: E402
import embed as E                                          # noqa: E402
from psychoacoustic import masking_threshold, temporal_envelope   # noqa: E402

FS = 16000
N, HOP = E.FRAME_LEN, E.HOP
PRE_S, POST_S = 0.40, 0.25          # CASP context before / after the probe
LEVELS_DB = (-6.0, 0.0, 6.0)        # probe levels relative to MPEG's limit
H_LIMIT_DB = 18.0                   # headroom beyond +-18 dB is reported as clipped
BAND_LOW_HZ, BAND_HIGH_HZ, N_BANDS = 250.0, 8000.0, 12
_BIN_HZ = np.arange(N // 2 + 1) * FS / N
_SYNTH = np.hanning(N + 1)[:-1]


def band_edges(n_bands=N_BANDS):
    """ERB-equal band edges (Hz) over 250 Hz - 8 kHz, CASP's channel range."""
    e = np.linspace(CM.hz_to_erb_rate(BAND_LOW_HZ), CM.hz_to_erb_rate(BAND_HIGH_HZ), n_bands + 1)
    return CM.erb_rate_to_hz(e)


class Host:
    """A 16 kHz host clip with MPEG's per-block threshold and envelope weights cached."""

    def __init__(self, x16):
        self.x = np.asarray(x16, dtype=float)
        self.starts = list(E.block_starts(len(self.x)))
        env = temporal_envelope(self.x, FS)
        self.weight = [env[s:s + N] / max(env[s:s + N].max(), 1e-9) for s in self.starts]
        self._thr = {}

    def threshold(self, b):
        if b not in self._thr:
            s = self.starts[b]
            self._thr[b] = masking_threshold(self.x[s:s + N], FS)
        return self._thr[b]

    def probe(self, b, band=None, seed=0, level_db=0.0):
        """Deployed-MPEG watermark block b (sign +1), optionally one band, scaled by level_db."""
        thr = self.threshold(b).copy()
        if band is not None:
            thr[(_BIN_HZ < band[0]) | (_BIN_HZ >= band[1])] = -200.0
        w = E.shape_to_mask(E.pn_block(1000 + seed, b), thr) * _SYNTH * self.weight[b]
        return w * 10 ** (level_db / 20)

    def segment(self, b):
        """(first sample, last sample) of CASP's context window around block b."""
        s = self.starts[b]
        return max(0, s - int(PRE_S * FS)), min(len(self.x), s + N + int(POST_S * FS))


def casp_dprimes(host, b, probes):
    """CASP d' of each probe (list of arrays, length N) placed at block b. Host IR shared."""
    a, z = host.segment(b)
    seg = host.x[a:z]
    ref = CM.internal_representation(seg, FS)
    d_jnd = CM.calibration_distance(FS)
    s = host.starts[b] - a
    out = []
    for p in probes:
        y = seg.copy()
        y[s:s + N] += p
        out.append(CM.ir_distance(ref["ir"], CM.internal_representation(y, FS)["ir"], ref["fs_ir"]) / d_jnd)
    return out


def measure_cell(host, b, band=None, seeds=(0, 1, 2)):
    """d' at every level in LEVELS_DB, averaged over seeds. Returns dict level -> d'."""
    probes, keys = [], []
    for lv in LEVELS_DB:
        for sd in seeds:
            probes.append(host.probe(b, band, sd, lv))
            keys.append(lv)
    if not any(np.any(p) for p in probes):
        return None
    d = casp_dprimes(host, b, probes)
    return {lv: float(np.mean([v for k, v in zip(keys, d) if k == lv])) for lv in LEVELS_DB}


def headroom(d_by_level, d_ref):
    """Level (dB re MPEG's limit) where CASP's d' reaches d_ref, from a straight-line fit
    of 20*log10(d') against level. Returns (H_dB, slope, clipped)."""
    lv = np.array(sorted(d_by_level))
    y = 20 * np.log10(np.maximum([d_by_level[v] for v in lv], 1e-9))
    slope, icpt = np.polyfit(lv, y, 1)
    if slope <= 0.05:                                    # d' does not grow with level: no answer
        return float("nan"), float(slope), True
    h = (20 * np.log10(d_ref) - icpt) / slope
    clipped = abs(h) > H_LIMIT_DB
    return float(np.clip(h, -H_LIMIT_DB, H_LIMIT_DB)), float(slope), bool(clipped)
