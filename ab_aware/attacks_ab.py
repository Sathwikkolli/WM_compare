"""
ab_aware/attacks_ab.py -- the 10-distortion grid for the AWARE detection A/B.

Thin wrapper over cascade/vox_attacks.py. Everything except `crop_50` is a
VOX_GRID entry at a fixed strength, so these numbers stay comparable with the
earlier robustness benchmarks in this repo.

WHY THESE TEN. One strength per attack, chosen to span failure *families*
rather than to sweep any single one -- a 40-clip run cannot support a strength
sweep and a detection A/B at once. Families:

    additive      gaussian_noise 20dB
    codec         mp3 32k, opus 16k, aac 40k
    band-limit    lowpass 0.3, highpass 0.2
    channel       echo 0.3
    requantize    quantization 8lvl
    desync        time_stretch 1.1, crop_50

The desync pair is the one that matters most for AWARE. align_bench established
that no aligner recovers time-stretched audio, so time_stretch 1.1 is expected
to be the worst cell in the table; crop_50 is recoverable in principle (gcc_phat
was sample-exact on crop) and is here to separate "signal destroyed" from
"merely misaligned". Do not read a low score on those two as the same kind of
failure as a low score on mp3.

WHY crop_50 IS LOCAL: run_bench.py's grid has crop_30s, which assumes minutes of
audio. Emilia clips run ~10 s, so an absolute 30 s crop is undefined here. A
proportional keep-half crop tests the same failure mode at the right scale.

IMPORTANT -- crop_50 changes length. PESQ and SNR need equal-length signals and
a sample-aligned reference, so score.py must skip perceptual metrics for it
(and for time_stretch). `needs_ref_align` flags exactly those two.

Usage:
    from attacks_ab import AB_GRID, apply
    y2 = apply('mp3', y, sr)          # -> numpy float32, or None if unavailable
    python attacks_ab.py              # prints the grid and what is installed
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

# cascade/ ships with the CODE, so prefer ROOT and treat WM_COMPARE_BASE as an
# override. align_bench keys off BASE alone, which breaks the moment BASE points
# somewhere that holds results but not source -- exactly what happens when
# analyze.py is pointed at a downloaded results tree.
for _cand in (os.path.join(ROOT, "cascade"), os.path.join(BASE, "cascade")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break


def _vox():
    """Import vox_attacks on first use, not at module import.

    AB_GRID/ORDER/family are pure data, and analyze.py wants them without
    dragging in ffmpeg and librosa -- re-analysing a downloaded results tree on
    a laptop should not require the attack toolchain.
    """
    import vox_attacks
    return vox_attacks


# name -> (vox_attacks name or None for local, param, family, needs_ref_align)
AB_GRID = {
    "gaussian_noise": ("gaussian_noise", 20,   "additive",   False),
    "mp3_32k":        ("mp3",            32,   "codec",      False),
    "opus_16k":       ("opus",           16,   "codec",      False),
    "aac_40k":        ("aac",            40,   "codec",      False),
    "lowpass_0.3":    ("lowpass",        0.3,  "band-limit", False),
    "highpass_0.2":   ("highpass",       0.2,  "band-limit", False),
    "echo_0.3":       ("echo",           0.3,  "channel",    False),
    "quantize_8lvl":  ("quantization",   8,    "requantize", False),
    "time_stretch_1.1": ("time_stretch", 1.1,  "desync",     True),
    "crop_50":        (None,             0.5,  "desync",     True),
}

# The order every table and plot uses. `clean` is the no-attack control and is
# always first -- it is the row that answers "does AWARE work at all here".
ORDER = ["clean"] + list(AB_GRID.keys())


def needs_ref_align(name):
    """True if this attack breaks sample alignment -> skip PESQ/SNR."""
    if name == "clean":
        return False
    return AB_GRID[name][3]


def family(name):
    return "control" if name == "clean" else AB_GRID[name][2]


def _crop_50(y, keep=0.5):
    """Keep the first half. Head-anchored on purpose: a centre crop would add an
    unknown offset on top of the length change and confound the two."""
    n = int(round(len(y) * keep))
    return y[:n].astype("float32") if n > 0 else None


def apply(name, y, sr):
    """Apply one distortion at its fixed strength.

    Returns numpy float32, or None if the attack is unavailable (missing ffmpeg,
    missing noise file) -- the caller records SKIP rather than a fake number.
    """
    if name == "clean":
        return np.asarray(y, dtype="float32")
    if name not in AB_GRID:
        raise KeyError(f"unknown attack {name!r}; known: {sorted(AB_GRID)}")

    vox_name, param, _fam, _na = AB_GRID[name]
    if vox_name is None:
        return _crop_50(y, param)
    try:
        z = _vox().apply(vox_name, param, y, sr)
    except Exception as e:
        print(f"  attack {name} failed: {e}", file=sys.stderr)
        return None
    return None if z is None else np.asarray(z, dtype="float32")


def main():
    print(f"{len(AB_GRID)} distortions + clean control = {len(ORDER)} conditions\n")
    print(f"{'condition':<20s} {'family':<12s} {'vox call':<24s} {'ref-align?'}")
    print("-" * 70)
    print(f"{'clean':<20s} {'control':<12s} {'--':<24s} {'ok'}")
    for name, (vox_name, param, fam, na) in AB_GRID.items():
        call = "local _crop_50" if vox_name is None else f"{vox_name}({param})"
        print(f"{name:<20s} {fam:<12s} {call:<24s} {'BREAKS' if na else 'ok'}")

    # Fail loudly here rather than 400 rows into the run.
    print("\nprobing availability on a 2 s tone...")
    sr = 16000
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)).astype("float32")
    bad = []
    for name in AB_GRID:
        z = apply(name, y, sr)
        status = "SKIP (unavailable)" if z is None else f"ok  {len(z)} samples"
        if z is None:
            bad.append(name)
        print(f"  {name:<20s} {status}")
    if bad:
        print(f"\nUNAVAILABLE: {bad} -- these will be recorded as SKIP, not scored.")
        print("  mp3/aac/opus     need ffmpeg on PATH")
        print("  time_stretch     needs librosa (present in the cluster wmcompare env;")
        print("                   absent on a bare local checkout -- not a real failure there)")
    else:
        print("\nall 10 available.")

    # aac comes back longer than it went in (ffmpeg strips the encoder delay for
    # mp3/opus but leaves AAC's trailing padding). Measured 2026-08-10 on
    # speech-like input: lag 0 for all three codecs, aac +128 samples at the
    # tail. Trailing-only padding is handled by the min(len) truncation in
    # run_ab.py's snr_db/pesq_wb, so aac does NOT need ref-align treatment.
    # Re-check this if the ffmpeg version changes.


if __name__ == "__main__":
    main()
