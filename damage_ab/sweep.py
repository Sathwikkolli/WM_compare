"""
damage_ab/sweep.py -- the strength sweep for the two conditions AWARE fails.

WHAT THIS GRID IS FOR. The A/B (results/2026-08-10_aware-detection-ab) found
`highpass_0.2` at 0/20 and `quantize_8lvl` at 1/20. It ran ONE strength per
attack, so it cannot say whether AWARE is fragile or whether the audio was
already destroyed. This grid sweeps both attacks from harmless to fatal and
scores unwatermarked audio through the identical path, so the two explanations
separate.

CUTOFFS ARE SPECIFIED IN Hz, NOT IN RATIO. `vox_attacks.a_highpass` takes a
cutoff expressed as a fraction of the SAMPLE RATE (julius' convention; the scipy
fallback's `min(2*ratio, 0.99)` is the same number re-expressed against Nyquist).
That indirection is what produced the "1600 Hz" error in THRESHOLD_DECISION.md
for an attack that actually cuts at 3200 Hz. Here the Hz is the source of truth
and the ratio is derived, so the mistake cannot recur:

    ratio = f_hz / sr        ->   0.2 * 16000 = 3200 Hz

THE TWO ANCHOR CELLS. `hp_3200hz` and `quant_8lvl` reproduce the A/B's
`highpass_0.2` and `quantize_8lvl` exactly at 16 kHz. `_assert_anchors()` checks
that against ab_aware/attacks_ab.py rather than trusting this comment, so if
either grid is edited the mismatch surfaces at import.

WHY THESE VALUES.

    high-pass    200, 500 Hz     below AWARE's 1000-4000 Hz band -- should be
                                 harmless. 500 Hz is the bench's passing
                                 `highpass_500` (conf 0.998 in bench_aware.csv).
                 1000-3200 Hz    progressively eats the band from the bottom.
                                 The breaking point lives in here.

    quantize     256 lvl         ~= the bench's passing `quantize_8bit`
                                 (conf 0.995). NOTE it is not identical: the
                                 bench quantiser is zero-centred, `a_quantization`
                                 anchors its grid on the file's own min/max, so
                                 zero is generally NOT a level. That difference
                                 is the whole story at 8 lvl and is why 256 is
                                 included rather than assumed equivalent.
                 64 .. 4 lvl     down through the A/B's 8 lvl to worse.

`mp3_32k` is the quality ANCHOR, not a subject. It is a condition AWARE survives
20/20 in the A/B, so it fixes the PESQ scale: without it, "PESQ 1.3" has no
reference point on this clip set.

Usage:
    from sweep import GRID, ORDER, apply_cond, group_of
    z = apply_cond('hp_3200hz', y, 16000)
    python sweep.py            # prints the grid, checks anchors, probes deps
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# cascade/ ships with the code; same resolution order attacks_ab.py uses, and for
# the same reason -- WM_COMPARE_BASE may point at a results tree with no source.
for _cand in (os.path.join(ROOT, "cascade"),
              os.path.join(os.environ.get("WM_COMPARE_BASE", ROOT), "cascade")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

WORK_SR = 16000

# AWARE's embedding window. Source: fsss/band_steer.py -- "AWARE writes in a
# fixed window (1000-4000 Hz at 16 kHz)". Every band-retention number in this
# experiment is measured over exactly this range.
AWARE_BAND = (1000.0, 4000.0)

HP_HZ = [200, 500, 1000, 1500, 2000, 2500, 3200]
Q_LEVELS = [256, 64, 32, 16, 8, 4]

# The A/B cells these two sweeps must reproduce, checked by _assert_anchors().
ANCHOR_HP_HZ = 3200
ANCHOR_Q_LVL = 8


def _hp_ratio(f_hz, sr=WORK_SR):
    """Cutoff in Hz -> the fraction-of-sample-rate that a_highpass expects."""
    return float(f_hz) / float(sr)


# condition -> (vox attack name, param, group, human label)
GRID = {}
GRID["mp3_32k"] = ("mp3", 32, "anchor", "mp3 32 kbps")
for f in HP_HZ:
    GRID[f"hp_{f}hz"] = ("highpass", _hp_ratio(f), "highpass", f"high-pass {f} Hz")
for l in Q_LEVELS:
    GRID[f"quant_{l}lvl"] = ("quantization", l, "quantize", f"quantize {l} levels")

# `clean` is the no-attack control and is always first: it carries the
# watermark-only PESQ, which is the number every attacked row is measured against.
ORDER = ["clean"] + list(GRID.keys())

# Numeric x for the sweep plots: Hz for high-pass, levels for quantisation.
# Deliberately NOT a shared axis -- the two sweeps are plotted separately.
STRENGTH_X = {f"hp_{f}hz": float(f) for f in HP_HZ}
STRENGTH_X.update({f"quant_{l}lvl": float(l) for l in Q_LEVELS})


def group_of(cond):
    return "control" if cond == "clean" else GRID[cond][2]


def label_of(cond):
    return "no attack" if cond == "clean" else GRID[cond][3]


def _assert_anchors():
    """Fail at import if this grid has drifted from the A/B it must reproduce.

    A comment claiming "hp_3200hz == highpass_0.2" is worth nothing once someone
    edits either file. This reads the A/B's own grid and compares the numbers.
    """
    sys.path.insert(0, os.path.join(ROOT, "ab_aware"))
    try:
        from attacks_ab import AB_GRID
    except Exception as e:                      # ab_aware absent -> skip, don't crash
        print(f"note: could not import ab_aware/attacks_ab.py ({e}); "
              f"anchor check skipped", file=sys.stderr)
        return

    ab_hp = AB_GRID["highpass_0.2"][1]
    ab_q = AB_GRID["quantize_8lvl"][1]
    mine_hp = GRID[f"hp_{ANCHOR_HP_HZ}hz"][1]
    mine_q = GRID[f"quant_{ANCHOR_Q_LVL}lvl"][1]

    if abs(mine_hp - ab_hp) > 1e-9:
        raise AssertionError(
            f"anchor drift: hp_{ANCHOR_HP_HZ}hz -> ratio {mine_hp} but the A/B's "
            f"highpass_0.2 is ratio {ab_hp} ({ab_hp * WORK_SR:.0f} Hz at {WORK_SR}). "
            f"These two cells must be the same attack or the sweep is not anchored."
        )
    if mine_q != ab_q:
        raise AssertionError(
            f"anchor drift: quant_{ANCHOR_Q_LVL}lvl -> {mine_q} levels but the "
            f"A/B's quantize_8lvl is {ab_q} levels."
        )


_assert_anchors()


def apply_cond(cond, y, sr):
    """Apply one condition. Returns float32, or None if the attack is unavailable.

    None means SKIP (missing ffmpeg, etc.) and the caller must record it as such
    rather than substituting a number.
    """
    if cond == "clean":
        return np.asarray(y, dtype="float32")
    if cond not in GRID:
        raise KeyError(f"unknown condition {cond!r}; known: {sorted(GRID)}")
    name, param, _g, _l = GRID[cond]
    try:
        import vox_attacks
        z = vox_attacks.apply(name, param, y, sr)
    except Exception as e:
        print(f"  attack {cond} failed: {e}", file=sys.stderr)
        return None
    return None if z is None else np.asarray(z, dtype="float32")


def _trapz(yv, xv):
    """np.trapezoid on numpy>=2, np.trapz below it. The cluster env and a local
    checkout are not guaranteed to agree on numpy major version."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(yv, xv))


def band_energy(x, sr, lo, hi):
    """Energy in [lo, hi) Hz via Welch PSD."""
    from scipy.signal import welch
    f, p = welch(np.asarray(x, dtype="float64"), sr, nperseg=2048)
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0
    return _trapz(p[m], f[m])


def band_keep(ref, deg, sr, band=AWARE_BAND):
    """Fraction of AWARE-band energy surviving the attack.

    This is the MECHANISM number. PESQ says the audio got worse; this says
    whether the specific frequencies AWARE writes into still carry their energy.
    A high-pass that empties the band and a quantiser that floods it with
    switching noise both destroy detection, but they are different failures and
    PESQ alone cannot tell them apart. Values >1 mean energy was ADDED.
    """
    e0 = band_energy(ref, sr, *band)
    e1 = band_energy(deg, sr, *band)
    return None if e0 <= 0 else float(e1 / e0)


def main():
    print(f"{len(GRID)} attack conditions + clean control = {len(ORDER)}\n")
    print(f"{'condition':<16s} {'group':<10s} {'label':<22s} {'vox call'}")
    print("-" * 74)
    print(f"{'clean':<16s} {'control':<10s} {'no attack':<22s} --")
    for c in GRID:
        name, param, g, lab = GRID[c]
        p = f"{param:.5f}" if isinstance(param, float) else str(param)
        print(f"{c:<16s} {g:<10s} {lab:<22s} {name}({p})")

    print(f"\nanchors OK: hp_{ANCHOR_HP_HZ}hz == A/B highpass_0.2, "
          f"quant_{ANCHOR_Q_LVL}lvl == A/B quantize_8lvl")
    print(f"AWARE band for retention: {AWARE_BAND[0]:.0f}-{AWARE_BAND[1]:.0f} Hz")

    # Probe availability here, not 100 rows into the run.
    print(f"\nprobing on 2 s of speech-shaped noise at {WORK_SR} Hz...")
    rs = np.random.RandomState(0)
    y = (0.05 * rs.randn(2 * WORK_SR)).astype("float32")
    bad = []
    for c in ORDER[1:]:
        z = apply_cond(c, y, WORK_SR)
        if z is None:
            bad.append(c)
            print(f"  {c:<16s} SKIP (unavailable)")
        else:
            k = band_keep(y, z, WORK_SR)
            print(f"  {c:<16s} ok   {len(z):>6d} samples   "
                  f"AWARE-band energy kept: {'n/a' if k is None else f'{100*k:8.2f}%'}")
    if bad:
        print(f"\nUNAVAILABLE: {bad}")
        print("  mp3_32k needs ffmpeg on PATH -- it is the quality anchor, so a run")
        print("  without it has no PESQ reference point. Fix before submitting.")
    else:
        print("\nall conditions available.")


if __name__ == "__main__":
    main()
