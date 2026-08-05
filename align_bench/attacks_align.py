"""
align_bench/attacks_align.py -- the 20-attack grid for the alignment bake-off,
with GROUND TRUTH logged at generation time.

Why this file exists instead of just using cascade/vox_attacks.py:

  1. VoxWatermark's VOX_GRID has NO cropping attack. Crop/splice/insert -- the
     whole reason we need an aligner -- must be added by hand.
  2. vox_attacks.apply() returns audio only. For alignment we need to know
     exactly WHERE every sample went, so each attack here also returns the
     segment map.

OFFSET CONVENTION (get this wrong and every number is garbage):

    offset = ref_index - dist_index      for corresponding samples

  crop_head removing K samples  ->  dist[0] == ref[K]        ->  offset = +K
  insert_silence of K at start  ->  dist[K] == ref[0]        ->  offset = -K
  anything that doesn't move    ->  offset = 0

FAMILIES (they are scored differently -- see score.py):

  null      true offset is exactly 0. Tests hallucination.
  shift     one constant, known offset.
  multiseg  several segments, each with its own offset.
  warp      offset varies with time (time_stretch, time_jitter).
  codec     round-trip through an encoder adds a small unknown delay. We do NOT
            know the exact truth, so these are scored as "must be within 100 ms
            of zero" -- catches hallucination without pretending to know the
            encoder's delay. Raw predictions are kept so the real delay is
            visible in the data.

A NOTE ON phase_shift: despite the name, vox_attacks.a_phase does NOT shift
anything. It builds [zeros(shift), y[shift:]], so content stays at its original
index -- it mutes the head. True offset is 0, so it lives in `null`.

Usage:
    from attacks_align import ALIGN_GRID, apply
    y2, gt = apply('crop_head', 0.25, y, sr, foreign=other_clip)
    # gt = {'segments': [...], 'offset': int, 'family': str, 'speed': float|None}
    # returns (None, None) if the attack's dependency is missing -> caller SKIPs
"""
import os
import sys
import numpy as np

# cascade/vox_attacks.py -- same import dance fsss/ uses
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

import vox_attacks  # noqa: E402


# --------------------------------------------------------------------------- #
#  Ground-truth helpers
# --------------------------------------------------------------------------- #
def _segments_from_keep(y, keep):
    """Build distorted audio by keeping `keep` = [(ref_start, ref_end), ...].

    Returns (z, segments) where each segment is
    (ref_start, ref_end, dist_start, dist_end).
    """
    parts, segs, pos = [], [], 0
    for (s, e) in keep:
        s, e = int(s), int(e)
        if e <= s:
            continue
        parts.append(y[s:e])
        segs.append((s, e, pos, pos + (e - s)))
        pos += e - s
    if not parts:
        return None, None
    return np.concatenate(parts).astype("float32"), segs


def _primary_offset(segments):
    """Offset of the LONGEST segment -- what a single-offset method should find."""
    if not segments:
        return 0
    ref_s, ref_e, dist_s, _ = max(segments, key=lambda s: s[1] - s[0])
    return int(ref_s - dist_s)


def _gt(segments, family, speed=None):
    return {
        "segments": segments,
        "offset": _primary_offset(segments),
        "family": family,
        "speed": speed,
        "n_segments": len(segments),
    }


def _passthrough_gt(n, family="null"):
    """Attack that does not move the time axis."""
    return _gt([(0, n, 0, n)], family)


# --------------------------------------------------------------------------- #
#  Desync attacks -- NOT in VoxWatermark, implemented here with exact truth
# --------------------------------------------------------------------------- #
def a_crop_head(y, sr, frac):
    """Remove `frac` of the clip from the START. Shifts everything: offset = +K."""
    n = len(y)
    k = int(round(frac * n))
    if n - k < sr:                     # need >= 1 s left to be meaningful
        return None, None
    z, segs = _segments_from_keep(y, [(k, n)])
    return z, _gt(segs, "shift")


def a_crop_tail(y, sr, frac):
    """Remove `frac` from the END. File is shorter but offset is still 0.

    Separates methods that confuse 'shorter' with 'shifted'.
    """
    n = len(y)
    k = int(round(frac * n))
    if n - k < sr:
        return None, None
    z, segs = _segments_from_keep(y, [(0, n - k)])
    return z, _gt(segs, "null")


def a_splice_cut(y, sr, n_cuts):
    """Remove `n_cuts` chunks from the MIDDLE -> n_cuts+1 surviving segments.

    Each cut is 8% of the clip, spread evenly, never touching the edges.
    """
    n = len(y)
    cut = int(round(0.08 * n))
    n_cuts = int(n_cuts)
    if cut < sr // 4 or n - n_cuts * cut < sr:
        return None, None

    # evenly spaced cut starts inside [10%, 90%] of the clip
    lo, hi = int(0.10 * n), int(0.90 * n)
    span = hi - lo
    starts = [lo + int((i + 1) * span / (n_cuts + 1)) - cut // 2 for i in range(n_cuts)]

    keep, pos = [], 0
    for s in starts:
        if s > pos:
            keep.append((pos, s))
        pos = s + cut
    if pos < n:
        keep.append((pos, n))

    z, segs = _segments_from_keep(y, keep)
    if z is None:
        return None, None
    return z, _gt(segs, "multiseg")


def a_insert_silence(y, sr, dur_s):
    """Insert `dur_s` of silence at the MIDPOINT. Shift happens mid-file."""
    n = len(y)
    k = int(round(dur_s * sr))
    at = n // 2
    z = np.concatenate([y[:at], np.zeros(k, dtype="float32"), y[at:]]).astype("float32")
    segs = [(0, at, 0, at), (at, n, at + k, n + k)]
    return z, _gt(segs, "multiseg")


def a_insert_foreign(y, sr, dur_s, foreign=None):
    """Insert `dur_s` of audio that has NO origin in the reference.

    Tests whether a method will admit 'this part isn't yours' rather than
    forcing a match.
    """
    if foreign is None or len(foreign) == 0:
        return None, None
    n = len(y)
    k = int(round(dur_s * sr))
    f = foreign
    if len(f) < k:
        f = np.tile(f, k // len(f) + 1)
    f = f[:k].astype("float32")
    at = n // 2
    z = np.concatenate([y[:at], f, y[at:]]).astype("float32")
    segs = [(0, at, 0, at), (at, n, at + k, n + k)]
    return z, _gt(segs, "multiseg")


# --------------------------------------------------------------------------- #
#  VoxWatermark attacks -- delegate audio to vox_attacks, attach truth here
# --------------------------------------------------------------------------- #
# family for every vox attack we use
_VOX_FAMILY = {
    "dynamic_compression": "null",
    "echo": "null",
    "quantization": "null",
    "lowpass": "null",
    "gaussian_noise": "null",
    "background_noise": "null",
    "highpass": "null",
    "inverse_polarity": "null",
    "phase_shift": "null",          # see module docstring -- does NOT shift
    "mp3": "codec",
    "opus": "codec",
    "aac": "codec",
    "encodec": "codec",
    "time_stretch": "warp",
    "time_jitter": "warp",
}


def _vox(name, param, y, sr):
    z = vox_attacks.apply(name, param, y.astype("float32"), sr)
    if z is None:
        return None, None
    fam = _VOX_FAMILY[name]

    if name == "time_stretch":
        # librosa stretches by `rate` then vox pads/trims back to len(y):
        # dist[i] corresponds to ref[i*rate], so offset grows linearly.
        # We score the warp family at the clip MIDPOINT.
        rate = float(param)
        n = len(y)
        mid = n // 2
        off = int(round(mid * (rate - 1.0)))
        return z, {"segments": [(0, n, 0, n)], "offset": off,
                   "family": fam, "speed": rate, "n_segments": 1}

    if name == "time_jitter":
        # zero-mean random resampling of the time axis -> expected offset 0,
        # but the path is non-monotone. Pure stress test.
        return z, _gt([(0, len(y), 0, len(y))], fam, speed=1.0)

    return z, _passthrough_gt(len(y), fam)


# --------------------------------------------------------------------------- #
#  THE GRID -- 20 attacks, 77 configurations
# --------------------------------------------------------------------------- #
_V = vox_attacks.VOX_GRID

ALIGN_GRID = {
    # --- Metapyxl mastering-chain proxy (6, FIXED -- matches
    #     fsss/exp_v12_metapxyl_compare.py METAPXYL_PROXY) --------------------
    "dynamic_compression": _V["dynamic_compression"],      # 2
    "echo":                _V["echo"],                     # 5
    "mp3":                 _V["mp3"],                      # 5
    "quantization":        _V["quantization"],             # 5
    "lowpass":             _V["lowpass"],                  # 5
    "gaussian_noise":      _V["gaussian_noise"],           # 5

    # --- desync, added by hand (absent from VoxWatermark) -------------------
    "crop_head":       [(f"{int(f*100)}pct", f) for f in [0.10, 0.25, 0.50, 0.75]],
    "crop_tail":       [(f"{int(f*100)}pct", f) for f in [0.10, 0.25, 0.50, 0.75]],
    "splice_cut":      [(f"{c}cuts", c) for c in [1, 2, 3]],
    "insert_silence":  [(f"{d}s", d) for d in [0.5, 1.0, 2.0]],
    "insert_foreign":  [(f"{d}s", d) for d in [1.0, 2.0, 3.0]],

    # --- desync / warp, from VoxWatermark ----------------------------------
    "time_stretch":    _V["time_stretch"],                 # 5
    "time_jitter":     _V["time_jitter"],                  # 2
    "phase_shift":     _V["phase_shift"],                  # 2  (really a null)

    # --- codec + zero-shift stress -----------------------------------------
    "opus":            _V["opus"],                         # 6
    "aac":             _V["aac"],                          # 2
    "encodec":         _V["encodec"],                      # 5   (GPU)
    "background_noise": _V["background_noise"],            # 5   (needs cascade/noises/)
    "highpass":        _V["highpass"],                     # 5
    "inverse_polarity": _V["inverse_polarity"],            # 1   (the trap)
}

_LOCAL = {
    "crop_head": a_crop_head,
    "crop_tail": a_crop_tail,
    "splice_cut": a_splice_cut,
    "insert_silence": a_insert_silence,
    "insert_foreign": a_insert_foreign,
}

FAMILY_OF = dict(_VOX_FAMILY)
FAMILY_OF.update({"crop_head": "shift", "crop_tail": "null",
                  "splice_cut": "multiseg", "insert_silence": "multiseg",
                  "insert_foreign": "multiseg"})


def apply(name, param, y, sr, foreign=None):
    """Apply one attack at one strength.

    Returns (distorted_audio, ground_truth_dict), or (None, None) if the attack
    is unavailable (missing dependency, or clip too short) -> caller records SKIP.
    """
    if name in _LOCAL:
        if name == "insert_foreign":
            return a_insert_foreign(y, sr, param, foreign=foreign)
        return _LOCAL[name](y, sr, param)
    if name in _VOX_FAMILY:
        return _vox(name, param, y, sr)
    raise KeyError(f"unknown attack '{name}'")


def n_configs():
    return sum(len(v) for v in ALIGN_GRID.values())


if __name__ == "__main__":
    print(f"{len(ALIGN_GRID)} attacks, {n_configs()} configurations")
    by_fam = {}
    for name, grid in ALIGN_GRID.items():
        by_fam.setdefault(FAMILY_OF[name], []).append((name, len(grid)))
    for fam in ("null", "shift", "multiseg", "warp", "codec"):
        items = by_fam.get(fam, [])
        tot = sum(c for _, c in items)
        print(f"  {fam:9s} {tot:3d} configs  " + ", ".join(n for n, _ in items))
