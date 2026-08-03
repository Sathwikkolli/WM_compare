"""
fsss/salient_mapped.py -- salient-region masks for embedders that RESAMPLE.

Both strip experiments hand AWARE a `host` whose timeline is not the original's:

    exp1 (band_critical)  keep every M-th sample     -> host time = orig * 1/M
    exp2 (band_steer)     resample_poly(up, down)    -> host time = orig * up/down

fsss/salient_region.py cannot be reused directly for either, because it calls the
anchor detector on whatever audio it is handed -- and by the time the inherited
mask builder runs, that is the HOST. fsss/chain_embed.py flagged this in advance:

    "the inherited mask builder will be handed `host`, not the original audio --
     anchors must be computed on the original and mapped through, or placement
     will drift."

So this module detects on the ORIGINAL and maps sample indices through a rational
time scale num/den. exp1 passes (1, M); exp2 passes (plan.up, plan.down).

WHY THIS IS SHARED RATHER THAN COPIED
-------------------------------------
exp1b vs exp2b is meant to isolate the SAMPLING strategy. If each arm built its
mask with its own copy of this arithmetic, any subtle divergence -- an off-by-one
in the half-width, a different merge rule -- would show up as a result. One
implementation, two time scales, nothing else different.

Region widths are specified in REAL milliseconds and converted internally, so
`region_ms=250` means the same physical span of speech in both arms even though
it lands on a different number of host frames (exp1 compresses time 10x, exp2
only 3.75x).

FRAME BUDGET -- the trap this module logs rather than hides
-----------------------------------------------------------
Resampling divides the host frame count by the same factor it divides time. A
10 s clip at hop 256 gives 610 frames stock, ~163 for exp2, but only ~61 for
exp1. At salient_region's default anchor_rate=3.5 that is 35 anchors over 61
frames, which saturates the mask to fully-on and makes gating a silent no-op.
`stats["coverage"]` is returned every call for exactly this reason; callers
should warn above ~0.85.

Standalone smoke test (no AWARE / no GPU needed):
    python -m fsss.salient_mapped
"""

import numpy as np

try:
    from fsss.detectors import DETECTORS
except ImportError:
    from detectors import DETECTORS

__all__ = ["build_mapped_region_mask", "DEFAULT_ANCHOR_RATE", "DEFAULT_REGION_MS",
           "COVERAGE_WARN"]

# 3.5 is salient_region's default and is correct for full-rate embedding. After
# a resample it saturates -- see "FRAME BUDGET" above.
DEFAULT_ANCHOR_RATE = 1.2
DEFAULT_REGION_MS = 250.0          # REAL milliseconds, converted to host frames
COVERAGE_WARN = 0.85


def _full(rows, n_freq, n_frames, n_anchors):
    """Fall back to the full stripe rather than embedding nothing.

    salient_region.py takes the same position: an empty anchor set degrades the
    variant to stock AWARE instead of producing a silent failure. `fell_back`
    makes it visible in the stats so a suspiciously good result can be checked.
    """
    M = np.zeros((n_freq, n_frames), dtype=bool)
    M[rows, :] = True
    return M, {"n_anchors": int(n_anchors), "n_regions": 1,
               "frames_covered": int(n_frames), "n_frames": int(n_frames),
               "coverage": 1.0, "cells": int(M.sum()),
               "cells_full_stripe": int(len(rows) * n_frames),
               "fell_back": True}


def build_mapped_region_mask(orig_audio, orig_sr, num, den, n_freq, n_frames,
                             hop_length, rows, anchor="librosa_flux",
                             anchor_rate=DEFAULT_ANCHOR_RATE,
                             region_ms=DEFAULT_REGION_MS):
    """Boolean [n_freq x n_frames] mask over the HOST, from anchors on the ORIGINAL.

    orig_audio : the untransformed input. NOT the host -- that is the whole point.
    num, den   : host time = original time * num/den.
                 exp1 -> (1, M);  exp2 -> (plan.up, plan.down)
    rows       : frequency rows the detector will read (ask the embedder, do not
                 assume, so the mask cannot drift from the detector band)

    Returns (M, stats). stats carries coverage and cell counts so a weak result
    can be attributed to budget rather than guessed at.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if n_frames <= 0 or rows.size == 0:
        return np.zeros((n_freq, max(n_frames, 0)), dtype=bool), {
            "n_anchors": 0, "n_regions": 0, "coverage": 0.0, "cells": 0,
            "cells_full_stripe": 0, "fell_back": True, "n_frames": int(n_frames)}

    anchors = np.asarray(DETECTORS[anchor](orig_audio, orig_sr,
                                           target_rate=anchor_rate))
    if anchors.size == 0:
        return _full(rows, n_freq, n_frames, 0)

    # A region of `region_ms` REAL milliseconds, expressed in host frames.
    # region_ms/1000 * orig_sr = samples in the original; * num/den moves it to
    # host samples; / hop_length to frames; / 2 because `half` is a half-width.
    half = max(1, int(round(region_ms * orig_sr / 1000.0 * num / den
                            / hop_length / 2)))

    spans = []
    for a in anchors:
        centre = (int(a) * num // den) // hop_length      # the time mapping
        s, e = max(0, centre - half), min(n_frames, centre + half)
        if e > s:
            spans.append((s, e))
    if not spans:                                          # every anchor fell off
        return _full(rows, n_freq, n_frames, int(anchors.size))

    spans.sort()
    merged = [list(spans[0])]                              # union overlapping regions
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    M = np.zeros((n_freq, n_frames), dtype=bool)
    for s, e in merged:
        M[rows[:, None], np.arange(s, e)[None, :]] = True

    covered = int(sum(e - s for s, e in merged))
    return M, {
        "n_anchors": int(anchors.size),
        "n_regions": len(merged),
        "frames_covered": covered,
        "n_frames": int(n_frames),
        "coverage": covered / float(n_frames),
        "cells": int(M.sum()),
        "cells_full_stripe": int(len(rows) * n_frames),
        "fell_back": False,
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Mask logic only -- no AWARE, no GPU. Clicks at 0.5 s intervals so spectral
    # flux has something unambiguous to find.
    sr, hop, n_freq = 16000, 256, 513
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(10 * sr) * 0.01).astype(np.float32)
    for t in range(1, 20):
        s = int(t * 0.5 * sr)
        y[s:s + 400] += np.hanning(400).astype(np.float32) * 0.6
    rows = np.arange(0, n_freq)

    print(f"{'arm':6s} {'num/den':>8s} {'frames':>7s} {'anchors':>8s} "
          f"{'regions':>8s} {'coverage':>9s} {'cells':>9s}")
    # exp1 compresses time 10x, exp2 3.75x, stock not at all
    for name, (num, den) in (("stock", (1, 1)), ("exp2", (4, 15)), ("exp1", (1, 10))):
        n_host = len(y) * num // den
        n_frames = 1 + n_host // hop
        M, st = build_mapped_region_mask(y, sr, num, den, n_freq, n_frames,
                                         hop, rows)
        assert M.shape == (n_freq, n_frames)
        assert M.sum() == st["cells"]
        flag = "  <-- SATURATED" if st["coverage"] > COVERAGE_WARN else ""
        print(f"{name:6s} {f'{num}/{den}':>8s} {n_frames:>7d} "
              f"{st['n_anchors']:>8d} {st['n_regions']:>8d} "
              f"{st['coverage']*100:>8.1f}% {st['cells']:>9d}{flag}")

    # The default anchor_rate exists because of this: salient_region's 3.5
    # saturates once time has been compressed.
    print("\nwhy DEFAULT_ANCHOR_RATE is 1.2 and not salient_region's 3.5:")
    for rate in (3.5, 1.2):
        n_frames = 1 + (len(y) // 10) // hop
        _, st = build_mapped_region_mask(y, sr, 1, 10, n_freq, n_frames, hop,
                                         rows, anchor_rate=rate)
        flag = "  <-- gating is a NO-OP" if st["coverage"] > COVERAGE_WARN else "  ok"
        print(f"  exp1 @ rate {rate:>4.1f}: {st['n_anchors']:3d} anchors, "
              f"coverage {st['coverage']*100:5.1f}%{flag}")
    print("\nOK")
