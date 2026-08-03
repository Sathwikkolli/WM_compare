"""
fsss/salient_region.py -- salient-REGION embedder for AWARE (no frequency hopping).

Where StaircaseAWAREEmbedder hops between sub-bands on a key schedule, this
variant does the opposite: it keeps the FULL frequency range on every frame it
writes, and instead restricts embedding IN TIME to regions around librosa
spectral-flux salient points. Frames outside those regions are left untouched.

    staircase : WHICH sub-band, chosen by HMAC(key, segment)   -- band varies
    this file : WHICH time regions, chosen by spectral flux    -- band is full

Design notes
------------
* UNKEYED BY DESIGN. Salient points are computed from the audio, so an attacker
  holding the clip recovers the same regions we did. This variant makes no
  security claim -- it isolates whether content-gated PLACEMENT alone buys
  robustness/imperceptibility. Keying is the staircase's job.

* The detector never sees the mask (AWAREEmbedder._optimize feeds it the full
  reconstructed magnitude with everything outside the embedding band zeroed), so
  the regions do not have to survive attacks or be re-derivable at read time.

* Rows come from the base's own `_get_embedding_frequency_indices` whenever it is
  available, so the mask can never disagree with the band the detector reads
  (the v10 detector-band bug). `band_range` is only a fallback.

* Fewer written frames means fewer optimization variables. `last_mask_stats`
  records coverage and cell count so a weak result can be attributed to budget
  (see StaircaseAWAREEmbedder._record_stats) rather than guessed at.

Standalone smoke test (no AWARE / no GPU needed):
    python -m fsss.salient_region
"""

import numpy as np

from fsss.staircase import StaircaseAWAREEmbedder

try:
    from fsss.detectors import DETECTORS
except ImportError:
    from detectors import DETECTORS

__all__ = ["SalientRegionAWAREEmbedder", "region_frames", "build_region_mask"]

DEFAULT_REGION_MS = 250.0
DEFAULT_ANCHOR = "librosa_flux"
DEFAULT_ANCHOR_RATE = 3.5


# --------------------------------------------------------------------------- #
#  mask construction -- module level so it can be tested without an embedder
# --------------------------------------------------------------------------- #
def normalize_rows(idx, n_freq):
    """Coerce whatever `_get_embedding_frequency_indices` returned into int rows.

    It may hand back a boolean mask or an integer index array, numpy or torch;
    all four forms appear across AWARE versions, so normalize rather than assume.
    """
    if hasattr(idx, "detach"):                       # torch tensor
        idx = idx.detach().cpu().numpy()
    arr = np.asarray(idx)
    rows = np.where(arr.ravel())[0] if arr.dtype == bool else arr.astype(np.int64).ravel()
    return rows[(rows >= 0) & (rows < n_freq)]


def region_frames(audio, sampling_rate, n_frames, hop_length,
                  anchor=DEFAULT_ANCHOR, anchor_rate=DEFAULT_ANCHOR_RATE,
                  region_ms=DEFAULT_REGION_MS):
    """Frame ranges to write, one per salient point, centred on the point.

    Returns (segments, n_anchors). `segments` is a list of [start, end) frame
    pairs, merged where they overlap. An empty anchor set yields the full range,
    which makes this variant degrade to stock AWARE instead of embedding nothing.
    """
    if n_frames <= 0:
        return [], 0

    anchors = np.asarray(DETECTORS[anchor](audio, sampling_rate,
                                           target_rate=anchor_rate))
    if anchors.size == 0:
        return [(0, n_frames)], 0

    half = max(1, int(round(region_ms * sampling_rate / 1000.0 / hop_length / 2)))
    spans = []
    for a in anchors:
        centre = int(a) // hop_length
        s = max(0, centre - half)
        e = min(n_frames, centre + half)
        if e > s:
            spans.append((s, e))
    if not spans:
        return [(0, n_frames)], int(anchors.size)

    spans.sort()
    merged = [list(spans[0])]                        # union overlapping regions
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged], int(anchors.size)


def build_region_mask(audio, sampling_rate, n_freq, n_frames, hop_length, rows,
                      anchor=DEFAULT_ANCHOR, anchor_rate=DEFAULT_ANCHOR_RATE,
                      region_ms=DEFAULT_REGION_MS):
    """Boolean [n_freq x n_frames] mask: every row in `rows`, salient frames only.

    Returns (M, stats) where stats reports coverage and cell counts so a run can
    be compared against the full-stripe baseline it is competing with.
    """
    segs, n_anchors = region_frames(audio, sampling_rate, n_frames, hop_length,
                                    anchor=anchor, anchor_rate=anchor_rate,
                                    region_ms=region_ms)
    M = np.zeros((n_freq, n_frames), dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    for s, e in segs:
        M[rows[:, None], np.arange(s, e)[None, :]] = True

    covered = int(sum(e - s for s, e in segs))
    stats = {
        "n_anchors": n_anchors,
        "n_regions": len(segs),
        "frames_covered": covered,
        "n_frames": int(n_frames),
        "coverage": covered / float(n_frames) if n_frames else 0.0,
        "cells": int(M.sum()),
        "cells_full_stripe": int(len(rows) * n_frames),
        "fell_back": n_anchors == 0,
    }
    return M, stats


# --------------------------------------------------------------------------- #
#  embedder
# --------------------------------------------------------------------------- #
class SalientRegionAWAREEmbedder(StaircaseAWAREEmbedder):
    """Full-band embedding restricted to salient time regions.

    Inherits embed / _optimize / _record_stats from StaircaseAWAREEmbedder
    unchanged -- both are mask-agnostic, so only the mask builder is overridden.
    """

    @classmethod
    def from_embedder(cls, base, key=b"unused", band_range=(500, 5000),
                      anchor=DEFAULT_ANCHOR, anchor_rate=DEFAULT_ANCHOR_RATE,
                      region_ms=DEFAULT_REGION_MS, tolerance_db=None):
        """`key` is accepted for API parity with the staircase but is never read:
        this variant selects regions from content alone."""
        obj = super().from_embedder(base, key, n_bands=1, band_range=band_range,
                                    anchor=anchor, anchor_rate=anchor_rate,
                                    segment_mode="librosa", tolerance_db=tolerance_db)
        obj.region_ms = float(region_ms)
        obj.last_mask_stats = None
        return obj

    def _full_rows(self, sampling_rate, n_freq):
        """Rows the detector will actually read. Prefer the base's own indices so
        the mask cannot drift from the detector band; fall back to band_range."""
        getter = getattr(self, "_get_embedding_frequency_indices", None)
        if getter is not None:
            try:
                rows = normalize_rows(getter(sampling_rate, self.frame_length)[0], n_freq)
                if rows.size:
                    return rows
            except Exception:                        # signature drift across AWARE versions
                pass
        return np.concatenate(self._band_rows(sampling_rate, n_freq))

    def _build_staircase_mask(self, audio, sampling_rate, n_freq, n_frames):
        rows = self._full_rows(sampling_rate, n_freq)
        M, stats = build_region_mask(audio, sampling_rate, n_freq, n_frames,
                                     self.hop_length, rows,
                                     anchor=self.anchor,
                                     anchor_rate=self.anchor_rate,
                                     region_ms=self.region_ms)
        self.last_mask_stats = stats
        if self.verbose:
            from aware.utils.logger import logger
            if stats["fell_back"]:
                logger.warning("salient_region: no anchors found -- writing the "
                               "FULL stripe (equivalent to stock AWARE)")
            logger.info(
                f"salient_region: {stats['n_anchors']} anchors -> "
                f"{stats['n_regions']} regions, coverage "
                f"{stats['coverage']*100:.1f}% of frames, "
                f"{stats['cells']} writable cells "
                f"(full stripe would be {stats['cells_full_stripe']})")
        return M


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Mask logic only -- no AWARE, no GPU. Clicks at 0.5 s intervals so
    # spectral flux has something unambiguous to find.
    sr, hop, n_freq = 16000, 256, 513
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(4 * sr) * 0.01).astype(np.float32)
    for t in range(1, 8):
        s = int(t * 0.5 * sr)
        y[s:s + 400] += np.hanning(400).astype(np.float32) * 0.6
    n_frames = 1 + len(y) // hop
    rows = np.arange(32, 321)                        # 500-5000 Hz at n_fft=1024

    for region_ms in (100.0, 250.0, 500.0, 2000.0):
        M, st = build_region_mask(y, sr, n_freq, n_frames, hop, rows,
                                  region_ms=region_ms)
        assert M.shape == (n_freq, n_frames)
        assert M[:32].sum() == 0 and M[321:].sum() == 0, "wrote outside the band"
        assert M.sum() == st["cells"]
        print(f"region_ms={region_ms:7.1f}  anchors={st['n_anchors']:3d}  "
              f"regions={st['n_regions']:3d}  coverage={st['coverage']*100:5.1f}%  "
              f"cells={st['cells']:7d}  vs full {st['cells_full_stripe']}")
    print("OK")
