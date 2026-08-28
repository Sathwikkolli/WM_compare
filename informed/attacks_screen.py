"""
informed/attacks_screen.py -- the 27-attack screen for Phase A.

The question this grid exists to answer is NOT "which attacks break the
watermark". `2026-08-10_aware-detection-ab` already found two, and
`2026-08-17_attack-damage-control` then showed both only work by destroying the
audio (PESQ 1.04 and 1.32 against a clean 4.64). An attacker who wrecks the
audio has already lost.

The question is:

    which attacks break detection while the audio stays USABLE?

So every attack here is swept across strengths, and each strength is scored for
quality as well as detection. The output is a detection-vs-quality curve per
attack; the ones that fall below the detection threshold while still above the
usability floor are the real vulnerabilities.

REUSE, NOT REWRITE

`cascade/vox_attacks.py` already implements 17 families with the exact interface
we need (numpy in, numpy out, `apply(name, param, y, sr)`). Those are delegated
to, not copied. This file adds the ones missing from that grid -- the realistic,
quality-preserving ones, which is precisely where the vulnerability is expected.

Two fixes applied to the inherited grid:

  background_noise  vox's `a_background` always uses `wavs[0]`, so only
                    babble.wav is ever heard and factory1/machinegun are dead
                    files. Split here into three separate attacks.
  music_bed         absent from vox entirely, despite being the one case with
                    existing evidence of failure in usable audio
                    (real_attacks_summary.csv: conf 0.402 at +4 dB SNR while bit
                    accuracy is still 0.925).

Usage:
    python attacks_screen.py             # print the grid + what is installed
    python attacks_screen.py --check     # actually run each attack once
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import vox_attacks as VOX                      # noqa: E402

NOISE_DIR = os.path.join(ROOT, "cascade", "noises")
AUDIO_DIR = os.path.join(BASE, "real_audio")


# --------------------------------------------------------------------------- #
#  categories -- what each attack is FOR, which is how the screen is read
# --------------------------------------------------------------------------- #
CATEGORY = {
    # Additive interference. Subtraction should help most here: the watermark is
    # masked, not deleted, and the damage is linear.
    "music_bed": "additive", "gaussian_noise": "additive",
    "noise_babble": "additive", "noise_factory": "additive",
    "noise_machinegun": "additive",

    # Codec / transmission. Non-linear: these REBUILD the waveform, so informed
    # subtraction is expected to struggle.
    "mp3": "codec", "aac": "codec", "opus": "codec", "encodec": "codec",
    "resample_roundtrip": "codec", "platform_reencode": "codec",

    # Filtering. Removes a band outright -- what is gone cannot be recovered.
    "highpass": "filter", "lowpass": "filter", "smooth": "filter",

    # Dynamics / level.
    "quantization": "dynamics", "dynamic_compression": "dynamics",
    "dynamic_expansion": "dynamics", "volume": "dynamics",
    "mastering_chain": "dynamics",

    # Acoustic path. Realistic and quality-preserving -- the interesting group.
    "reverb": "acoustic", "echo": "acoustic", "stereo_widen": "acoustic",

    # Enhancement. Quality may go UP while the watermark is stripped as "noise".
    "denoise": "enhancement",

    # Temporal. NOTE: the bake-off found no aligner handles time stretch, so
    # informed detection is unavailable for these regardless of the result.
    "time_stretch": "temporal", "time_jitter": "temporal",

    # Controls. Should not move detection at all; if they do, something is wrong
    # with the harness rather than with the watermark.
    "inverse_polarity": "control", "phase_shift": "control",
}

# Attacks that defeat alignment itself, so Phase B cannot use them even if
# Phase A flags them as vulnerabilities. Recorded, not silently dropped.
ALIGNMENT_BREAKING = {"time_stretch"}


# --------------------------------------------------------------------------- #
#  the grid
# --------------------------------------------------------------------------- #
def _grid():
    g = {}

    # ---- inherited from vox_attacks, minus background_noise (split below) ---
    for name, params in VOX.VOX_GRID.items():
        if name == "background_noise":
            continue
        g[name] = list(params)

    # ---- background noise, one attack per noise type -----------------------
    # vox only ever reads wavs[0]; factory1 and machinegun were unreachable.
    for label, fname in (("babble", "babble.wav"),
                         ("factory", "factory1.wav"),
                         ("machinegun", "machinegun.wav")):
        g[f"noise_{label}"] = [(f"{s}dB", (fname, s))
                               for s in [30, 20, 15, 10, 5, 0]]

    # ---- new: realistic, quality-preserving --------------------------------
    # The lead candidate. Grid is coarser than informed/music_sweep.py on
    # purpose -- that run measures the crossing precisely, this one only has to
    # place music among the other attacks.
    g["music_bed"] = [(f"{s}dB", s) for s in [20, 15, 10, 6, 4, 2, 0, -3]]

    # Room acoustics, by RT60. Convolution with a synthetic decaying RIR rather
    # than ffmpeg aecho: aecho is discrete taps, not reverb, and RT60 is a
    # parameter a reviewer can interpret.
    g["reverb"] = [(f"rt{t}", t) for t in [0.15, 0.3, 0.5, 0.8, 1.2]]

    # Speech enhancement. The one attack where quality may IMPROVE while the
    # watermark is removed as if it were noise.
    g["denoise"] = [(f"nr{n}", n) for n in [6, 12, 20, 30, 40]]

    g["volume"] = [(f"x{v}", v) for v in [0.1, 0.25, 0.5, 2.0, 4.0]]

    g["resample_roundtrip"] = [(f"{r}Hz", r) for r in [4000, 8000, 11025, 16000]]

    # What an upload chain actually does: re-encode plus loudness normalisation.
    g["platform_reencode"] = [(f"aac{b}k", b) for b in [24, 48, 96, 128]]

    # Approximates the client's mastering chain (fsss/exp_v12_metapxyl_compare).
    g["mastering_chain"] = [("light", 0), ("medium", 1), ("heavy", 2)]

    # Haas widening, then the detector's own mono downmix -- isolates phase
    # cancellation. From real_attacks_experiment.py TEST 2.
    g["stereo_widen"] = [(f"{d}ms", d) for d in [6, 12, 20, 30]]

    return g


SCREEN_GRID = _grid()


def n_attacks():
    return len(SCREEN_GRID)


def n_configs():
    return sum(len(v) for v in SCREEN_GRID.values())


# --------------------------------------------------------------------------- #
#  helpers for the new attacks
# --------------------------------------------------------------------------- #
def _ffmpeg_filter(y, sr, af):
    """Round-trip through one ffmpeg filter chain. None if ffmpeg is missing."""
    import soundfile as sf
    d = tempfile.mkdtemp(prefix="screen_")
    src, out = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    try:
        sf.write(src, y.astype("float32"), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-af", af, "-ar", str(sr), "-ac", "1", out], check=True)
        z, _ = sf.read(out)
        return np.asarray(z, dtype="float32")
    except Exception:
        return None
    finally:
        for p in (src, out):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


def _fix_len(y, n):
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def _rir(rt60, sr, seed=0):
    """Synthetic room impulse response: noise with exponential decay to -60 dB."""
    n = max(int(rt60 * sr), 8)
    t = np.arange(n) / sr
    env = np.exp(-6.907 * t / rt60)           # ln(1000) = 6.907 -> -60 dB at rt60
    h = np.random.RandomState(seed).randn(n).astype("float32") * env
    h[0] += 1.0                               # keep the direct path
    return (h / (np.linalg.norm(h) + 1e-12)).astype("float32")


def _a_reverb(y, sr, rt60):
    h = _rir(rt60, sr)
    z = np.convolve(y, h, mode="full")[:len(y)]
    # match RMS: reverb should change the acoustics, not the level
    r0 = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    r1 = float(np.sqrt(np.mean(z ** 2))) + 1e-12
    return (z * (r0 / r1)).astype("float32")


def _a_background(y, sr, param):
    """param = (noise filename, snr_db). One attack per noise type."""
    fname, snr_db = param
    import soundfile as sf
    path = os.path.join(NOISE_DIR, fname)
    if not os.path.exists(path):
        return None
    noise, nsr = sf.read(path)
    if getattr(noise, "ndim", 1) > 1:
        noise = noise.mean(1)
    if nsr != sr:
        import math

        from scipy.signal import resample_poly
        g = math.gcd(int(nsr), int(sr))
        noise = resample_poly(noise, sr // g, int(nsr) // g)
    noise = np.asarray(noise, dtype="float32")
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)
    noise = noise[:len(y)]
    sp = float(np.mean(y ** 2))
    npow = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(sp / ((10 ** (snr_db / 10.0)) * npow))
    return (y + noise * scale).astype("float32")


_MUSIC = {}


def _music(sr):
    """The canonical music bed, cached. Same track as real_attacks_experiment."""
    if sr in _MUSIC:
        return _MUSIC[sr]
    import soundfile as sf
    path = os.path.join(AUDIO_DIR, "music_march.mp3")
    if not os.path.exists(path):
        url = "https://archive.org/download/MarchForHonor/March_For_Honor.mp3"
        os.makedirs(AUDIO_DIR, exist_ok=True)
        try:
            subprocess.run(["curl", "-L", "-s", "-o", path, url], check=True)
        except Exception:
            return None
    try:
        m, msr = sf.read(path)
    except Exception:
        return None
    if getattr(m, "ndim", 1) > 1:
        m = m.mean(1)
    if msr != sr:
        import math

        from scipy.signal import resample_poly
        g = math.gcd(int(msr), int(sr))
        m = resample_poly(m, sr // g, int(msr) // g)
    _MUSIC[sr] = np.asarray(m, dtype="float32")
    return _MUSIC[sr]


def _a_music(y, sr, snr_db, ctx=None):
    m = (ctx or {}).get("music")
    if m is None:
        m = _music(sr)
    if m is None:
        return None
    if len(m) < len(y):
        m = np.tile(m, len(y) // len(m) + 1)
    m = m[:len(y)].astype("float32")
    sp = float(np.mean(y ** 2))
    pm = float(np.mean(m ** 2)) + 1e-20
    a = float(np.sqrt(sp / (pm * (10.0 ** (snr_db / 10.0)))))
    return (y + a * m).astype("float32")


def _a_resample_rt(y, sr, target):
    import math

    from scipy.signal import resample_poly
    g1 = math.gcd(int(sr), int(target))
    down = resample_poly(y, target // g1, int(sr) // g1)
    g2 = math.gcd(int(target), int(sr))
    up = resample_poly(down, sr // g2, int(target) // g2)
    return _fix_len(np.asarray(up, dtype="float32"), len(y))


def _a_platform(y, sr, kbps):
    """AAC re-encode + EBU R128 loudness normalisation -- what uploads do."""
    import soundfile as sf
    d = tempfile.mkdtemp(prefix="plat_")
    src = os.path.join(d, "in.wav")
    enc = os.path.join(d, "c.m4a")
    out = os.path.join(d, "out.wav")
    try:
        sf.write(src, y.astype("float32"), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-c:a", "aac", "-b:a", f"{kbps}k", enc], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc,
                        "-ar", str(sr), "-ac", "1", out], check=True)
        z, _ = sf.read(out)
        return _fix_len(np.asarray(z, dtype="float32"), len(y))
    except Exception:
        return None
    finally:
        for p in (src, enc, out):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


_MASTER = {
    0: "acompressor=threshold=-18dB:ratio=2:attack=20:release=250,highpass=f=60,alimiter=limit=0.95",
    1: "acompressor=threshold=-24dB:ratio=4:attack=10:release=180,equalizer=f=3000:t=q:w=1:g=3,highpass=f=80,alimiter=limit=0.92",
    2: "acompressor=threshold=-30dB:ratio=8:attack=5:release=120,equalizer=f=200:t=q:w=1:g=-3,equalizer=f=5000:t=q:w=1:g=5,highpass=f=100,alimiter=limit=0.89",
}


def _a_master(y, sr, level):
    return _ffmpeg_filter(y, sr, _MASTER[int(level)])


def _a_stereo_widen(y, sr, delay_ms):
    """Haas widening, then mono downmix -- the detector downmixes anyway."""
    z = _ffmpeg_filter(y, sr,
                       f"pan=stereo|c0=c0|c1=c0,adelay=0|{int(delay_ms)},"
                       f"pan=mono|c0=0.5*c0+0.5*c1")
    return None if z is None else _fix_len(z, len(y))


def _a_denoise(y, sr, nr_db):
    return _ffmpeg_filter(y, sr, f"afftdn=nr={int(nr_db)}")


def _a_volume(y, sr, v):
    return (y * float(v)).astype("float32")


_NEW = {
    "music_bed": _a_music,
    "reverb": lambda y, sr, p, ctx=None: _a_reverb(y, sr, p),
    "denoise": lambda y, sr, p, ctx=None: _a_denoise(y, sr, p),
    "volume": lambda y, sr, p, ctx=None: _a_volume(y, sr, p),
    "resample_roundtrip": lambda y, sr, p, ctx=None: _a_resample_rt(y, sr, p),
    "platform_reencode": lambda y, sr, p, ctx=None: _a_platform(y, sr, p),
    "mastering_chain": lambda y, sr, p, ctx=None: _a_master(y, sr, p),
    "stereo_widen": lambda y, sr, p, ctx=None: _a_stereo_widen(y, sr, p),
    "noise_babble": lambda y, sr, p, ctx=None: _a_background(y, sr, p),
    "noise_factory": lambda y, sr, p, ctx=None: _a_background(y, sr, p),
    "noise_machinegun": lambda y, sr, p, ctx=None: _a_background(y, sr, p),
}


def apply(name, param, y, sr, ctx=None):
    """Apply one attack at one strength. Returns numpy y2, or None if unavailable.

    None means "could not run" (missing ffmpeg, missing model, missing noise
    file) and MUST be recorded as a skip -- never as a failure of the watermark.
    """
    if name in _NEW:
        fn = _NEW[name]
        try:
            return fn(y, sr, param, ctx) if name == "music_bed" else fn(y, sr, param, ctx)
        except Exception:
            return None
    try:
        return VOX.apply(name, param, y, sr)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  report / check
# --------------------------------------------------------------------------- #
def main(argv):
    cats = {}
    for name in SCREEN_GRID:
        cats.setdefault(CATEGORY.get(name, "uncategorised"), []).append(name)

    print(f"{n_attacks()} attacks, {n_configs()} configs\n")
    for cat in ("additive", "codec", "filter", "dynamics", "acoustic",
                "enhancement", "temporal", "control", "uncategorised"):
        names = sorted(cats.get(cat, []))
        if not names:
            continue
        n_cfg = sum(len(SCREEN_GRID[n]) for n in names)
        print(f"  {cat:12s} {len(names):>2d} attacks, {n_cfg:>3d} configs")
        for n in names:
            flag = "  <- breaks alignment" if n in ALIGNMENT_BREAKING else ""
            src = "vox" if n in VOX.VOX_GRID else "new"
            print(f"      {n:22s} {len(SCREEN_GRID[n]):>2d}  [{src}]{flag}")
        print()

    if "--check" not in argv:
        print("pass --check to actually run each attack once on synthetic audio")
        return 0

    sr = 22050
    n = int(4.0 * sr)
    t = np.arange(n) / sr
    rng = np.random.RandomState(0)
    y = (rng.randn(n) * 0.05).astype("float32")
    for f in (180.0, 520.0, 1400.0, 3000.0):
        y += (0.2 * np.sin(2 * np.pi * f * t)).astype("float32")
    y *= (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 0.5 * t))).astype("float32")

    print(f"{'attack':22s} {'param':>12s}  {'result':>10s}  note")
    ok = skipped = 0
    for name in sorted(SCREEN_GRID):
        label, param = SCREEN_GRID[name][0]
        z = apply(name, param, y, sr, ctx=None)
        if z is None:
            skipped += 1
            print(f"{name:22s} {label:>12s}  {'SKIP':>10s}  unavailable "
                  f"(missing dependency or asset)")
        else:
            ok += 1
            d = 20 * np.log10((np.std(np.asarray(z)[:len(y)] - y) + 1e-12) /
                              (np.std(y) + 1e-12))
            print(f"{name:22s} {label:>12s}  {'ok':>10s}  "
                  f"len={len(z)} delta={d:+.1f}dB")
    print(f"\n{ok} runnable, {skipped} unavailable")
    if skipped:
        print("Unavailable attacks are recorded as SKIP in the sweep, never as "
              "watermark failures. Install ffmpeg / transformers / pydub to "
              "close the gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
