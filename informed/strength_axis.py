"""
informed/strength_axis.py -- a continuous, monotone strength axis per attack.

Phase A used fixed grids of 2-8 points per attack. That is fine for placing
attacks relative to each other, but useless for measuring a crossing: reading a
failure point off 5 points gives an answer +/- one grid step, and Phase B is
trying to measure a ~5 dB shift.

So each attack gets a continuous axis with a declared weakest and strongest end,
and Phase B bisects along it. 10 iterations narrows the bracket to ~0.1% of the
range, versus 50+ evaluations for a grid of that resolution.

THE PARAMETERISATION

Every attack is driven by t in [0, 1]:

    t = 0   weakest setting  -- both detectors should SUCCEED here
    t = 1   strongest        -- both detectors should FAIL here

`value(attack, t)` maps t to the attack's native units, linearly or
logarithmically as appropriate (bitrates and quantisation levels are perceptually
log-spaced; SNR in dB is already logarithmic).

Crossings are reported in NATIVE units, because "informed survives 5 dB more
music" is meaningful and "informed survives 0.11 more t" is not.

TWO-SIDED ATTACKS ARE SPLIT

`volume` and `time_stretch` get stronger in both directions from 1.0, so a single
monotone axis cannot describe them. `volume` is split into `volume_down` and
`volume_up`, each monotone. `time_stretch` is excluded from Phase B entirely --
the bake-off found no aligner handles it, so subtraction is impossible whatever
the result would have been.

WHAT HAS NO AXIS

  encodec, mastering_chain   3-5 fixed settings; cannot be bisected. Reported at
                             step resolution in Phase A, excluded from Phase B.
  inverse_polarity,          no strength parameter at all. Harness controls:
  phase_shift, time_jitter   they should not move detection, and if they do,
                             something is wrong with us rather than with AWARE.

    python strength_axis.py        # print every axis and its endpoints
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

# lo = weakest, hi = strongest. Direction is implied, not stated: `hi` may be
# numerically smaller (SNR, bitrate) or larger (cutoff, reverb). Interpolation
# handles both, so no separate sign flag is needed.
#
# `scale`: "lin" or "log". Bitrates, quantisation levels and sample rates are
# perceptually log-spaced -- linear interpolation would spend most of the sweep
# in the region where nothing happens.
#
# `cast`: applied to the native value ("int" where the attack needs one).
AXIS = {
    # ---- additive: SNR in dB, already logarithmic --------------------------
    # WIDENED after the first Phase B run: informed detection survived the
    # ENTIRE original range on every additive attack (49/50 clips censored
    # for gaussian and the three noises). The axes had been drawn around
    # where BLIND detection breaks, and informed operates far past that.
    # These ends are deliberately absurd -- noise 30-35 dB LOUDER than the
    # speech -- because the crossing has to be inside the bracket for a
    # bisection result to mean anything.
    "music_bed":        dict(lo=30.0, hi=-45.0, scale="lin", unit="dB SNR"),
    "gaussian_noise":   dict(lo=50.0, hi=-30.0, scale="lin", unit="dB SNR"),
    "noise_babble":     dict(lo=40.0, hi=-35.0, scale="lin", unit="dB SNR",
                             noise="babble.wav"),
    "noise_factory":    dict(lo=40.0, hi=-35.0, scale="lin", unit="dB SNR",
                             noise="factory1.wav"),
    "noise_machinegun": dict(lo=40.0, hi=-35.0, scale="lin", unit="dB SNR",
                             noise="machinegun.wav"),

    # ---- codec: bitrate, log-spaced ----------------------------------------
    "mp3":               dict(lo=128.0, hi=8.0,  scale="log", unit="kbps", cast="int"),
    "aac":               dict(lo=128.0, hi=8.0,  scale="log", unit="kbps", cast="int"),
    "opus":              dict(lo=128.0, hi=6.0,  scale="log", unit="kbps", cast="int"),
    "platform_reencode": dict(lo=128.0, hi=16.0, scale="log", unit="kbps", cast="int"),
    "resample_roundtrip": dict(lo=16000.0, hi=2000.0, scale="log", unit="Hz",
                               cast="int"),

    # ---- filtering ---------------------------------------------------------
    "highpass": dict(lo=0.02, hi=0.60, scale="lin", unit="cutoff ratio"),
    "lowpass":  dict(lo=0.02, hi=0.60, scale="lin", unit="cutoff ratio"),
    "smooth":   dict(lo=2.0,  hi=40.0, scale="lin", unit="window", cast="int"),

    # ---- acoustic ----------------------------------------------------------
    "reverb":       dict(lo=0.05, hi=2.50, scale="lin", unit="RT60 s"),
    "echo":         dict(lo=0.05, hi=0.95, scale="lin", unit="decay"),
    "stereo_widen": dict(lo=1.0,  hi=60.0, scale="lin", unit="ms", cast="int"),

    # ---- dynamics ----------------------------------------------------------
    "quantization": dict(lo=256.0, hi=2.0, scale="log", unit="levels", cast="int"),
    "denoise":      dict(lo=1.0,   hi=60.0, scale="lin", unit="nr dB", cast="int"),
    "dynamic_compression": dict(lo=1.5, hi=20.0, scale="lin", unit="ratio",
                                thresh=-25.0),
    "dynamic_expansion":   dict(lo=1.5, hi=20.0, scale="lin", unit="ratio",
                                thresh=-25.0),

    # ---- two-sided, split --------------------------------------------------
    "volume_down": dict(lo=1.0, hi=0.01, scale="log", unit="gain", base="volume"),
    "volume_up":   dict(lo=1.0, hi=20.0, scale="log", unit="gain", base="volume"),
}

# Present in the Phase A screen, deliberately absent here. Recorded so the
# omissions are visible rather than looking like oversights.
NO_AXIS = {
    "encodec": "only 5 fixed bandwidths -- cannot bisect",
    "mastering_chain": "only 3 presets -- cannot bisect",
    "inverse_polarity": "no strength parameter (harness control)",
    "phase_shift": "no strength parameter (harness control)",
    "time_jitter": "zero-mean, no monotone strength (harness control)",
    "time_stretch": "EXCLUDED: no aligner handles it, so subtraction is "
                    "impossible regardless of the result",
    "volume": "two-sided -- split into volume_down / volume_up",
}


def has_axis(attack):
    return attack in AXIS


def base_attack(attack):
    """The name `attacks_screen.apply` knows, for split attacks."""
    return AXIS.get(attack, {}).get("base", attack)


def value(attack, t):
    """Native parameter value at t in [0,1]. t=0 weakest, t=1 strongest."""
    ax = AXIS[attack]
    t = float(np.clip(t, 0.0, 1.0))
    lo, hi = float(ax["lo"]), float(ax["hi"])
    if ax.get("scale") == "log":
        if lo <= 0 or hi <= 0:
            raise ValueError(f"{attack}: log scale needs positive endpoints")
        v = float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo))))
    else:
        v = lo + t * (hi - lo)
    if ax.get("cast") == "int":
        v = int(round(v))
    return v


def t_of(attack, v):
    """Inverse of value(): where a native value sits on [0,1]. For reporting."""
    ax = AXIS[attack]
    lo, hi = float(ax["lo"]), float(ax["hi"])
    if ax.get("scale") == "log":
        return float((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return float((v - lo) / (hi - lo))


def to_param(attack, t):
    """The parameter `attacks_screen.apply` expects, at strength t.

    Some attacks take a tuple rather than a scalar, which is why this exists
    instead of passing `value()` straight through.
    """
    ax = AXIS[attack]
    v = value(attack, t)
    if "noise" in ax:                      # (filename, snr_db)
        return (ax["noise"], v)
    if attack in ("dynamic_compression", "dynamic_expansion"):
        return (ax["thresh"], float(v))    # (threshold_db, ratio)
    return v


def label(attack, t):
    ax = AXIS[attack]
    v = value(attack, t)
    unit = ax.get("unit", "")
    return f"{v:g}{(' ' + unit) if unit else ''}"


def grid(attack, n):
    """n evenly spaced t values, or None if the attack has no axis.

    Used by null_calibrate to build a threshold curve the bisection can
    interpolate; bisection itself does not use a grid.
    """
    if attack not in AXIS:
        return None
    return list(np.linspace(0.0, 1.0, int(n)))


def apply_at(attack, t, y, sr):
    """Run `attack` at strength t. Returns audio, or None if unavailable."""
    import attacks_screen as A
    return A.apply(base_attack(attack), to_param(attack, t), y, sr)


def main():
    print(f"{len(AXIS)} attacks with a continuous strength axis\n")
    print(f"  {'attack':22s} {'weakest (t=0)':>18s} {'strongest (t=1)':>18s}  scale")
    for a in sorted(AXIS):
        print(f"  {a:22s} {label(a, 0.0):>18s} {label(a, 1.0):>18s}  "
              f"{AXIS[a].get('scale', 'lin')}")
    print(f"\n{len(NO_AXIS)} excluded:")
    for a, why in sorted(NO_AXIS.items()):
        print(f"  {a:22s} {why}")
    print("\nMidpoints (sanity check that the mapping is monotone):")
    for a in sorted(AXIS):
        vals = [label(a, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print(f"  {a:22s} " + "  ".join(f"{v:>12s}" for v in vals))


if __name__ == "__main__":
    main()
