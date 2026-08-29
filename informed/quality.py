"""
informed/quality.py -- audio quality scoring, reference-based and no-reference.

WHY THIS FILE EXISTS

`cascade/cascade_lib.py:quality_metrics()` already gives PESQ, SNR, SI-SNR and
STOI. All four are REFERENCE-BASED: they measure how far the audio has moved
from the original. That is the wrong question for this experiment.

An attacker does not need fidelity. They need the result to SOUND ACCEPTABLE.
The two come apart badly:

  * Speech mixed with a music bed scores poorly on PESQ -- it differs a lot from
    the clean speech -- while sounding completely normal. It is a podcast.
  * Anything that re-synthesises audio (neural codecs, vocoders, voice
    conversion) sounds fine and scores terribly.

Screen on PESQ alone and the music bed gets filed under "destroyed audio"
alongside `highpass_0.2`, which would throw away the best vulnerability we have.
See results/2026-08-28_informed-detection/README.md.

So this module adds a NO-REFERENCE score and reports both.

WHY DNSMOS SPECIFICALLY

`cascade/emilia_bench.py:98` already filters source clips with
`df['dnsmos'] >= 3.0`. Using the same metric makes our usability floor THE SAME
3.0 the project already uses to choose clips -- a defensible number rather than
an arbitrary one. A different scorer would need its own justification.

BACKENDS, in preference order

  speechmos     `pip install speechmos`. Bundles DNSMOS P.835; no manual model
                download. Returns sig/bak/ovrl on the standard 1-5 MOS scale.
  torchaudio    SQUIM_SUBJECTIVE. Already a dependency, but a DIFFERENT SCALE --
                not comparable to the Emilia manifest. Fallback only, and the
                backend used is recorded in every row so the two never get mixed
                silently.

If neither is available, `score()` returns None for the no-reference fields and
says why. It never guesses.

    python quality.py            # report backends + self-test on synthetic audio
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

SR_16K = 16000

# --------------------------------------------------------------------------- #
#  THE USABILITY FLOOR IS RELATIVE, NOT ABSOLUTE
# --------------------------------------------------------------------------- #
# An absolute floor was tried first and does not work here. Measured on clean,
# unattacked Emilia clips with this very scorer:
#
#     3.046  3.364  3.185  3.265  2.859  2.996  3.347  3.281
#     median 3.22, min 2.86
#
# An absolute floor of 3.0 already fails a QUARTER of undamaged audio, so every
# attack would be classified "destroys the audio" regardless of what it does.
#
# The original justification for 3.0 -- that it matched cascade/emilia_bench.py's
# DNSMOS_MIN -- also turned out to be empty: the Emilia manifest's `dnsmos`
# column runs 3.200 to 3.721 (min exactly 3.200, so it was pre-filtered when the
# manifest was built), which means that filter removes nothing at all. And that
# column is on a different scale from this scorer, so it cannot set our floor.
#
# So the floor is a DROP from each clip's own clean score. This is the better
# definition anyway: an attacker degrades a file from wherever it already was,
# not from some absolute ideal.
DROP_FLOOR = 0.5          # MOS. Conventional "clearly worse" step on a 1-5 scale.
DROP_FLOOR_STRICT = 0.3

# Kept only so old absolute-floor numbers remain interpretable. Do not use these
# to decide anything -- see above.
DNSMOS_FLOOR = 3.0
DNSMOS_FLOOR_STRICT = 3.5

_BACKEND = None          # resolved once, cached
_BACKEND_NOTE = ""


# --------------------------------------------------------------------------- #
#  backend resolution
# --------------------------------------------------------------------------- #
def _quiet_onnxruntime():
    """Silence onnxruntime's thread-affinity errors.

    On a Slurm node the job is confined to a cpuset, onnxruntime tries to pin
    each of its worker threads to a CPU outside that set, and every attempt logs
    a line like:

        [E:onnxruntime:Default, env.cc:226 ThreadMain] pthread_setaffinity_np
        failed for thread: ..., error code: 22

    It is harmless -- inference completes and the scores are correct -- but it is
    ~50 lines PER SCORED FILE. At 118 configs x 2 arms x 50 clips that is roughly
    half a million lines, which makes the array logs unreadable and hides real
    failures.

    speechmos builds its own InferenceSession, so there is no supported way to
    pass it session options and cap the thread count, which would be the proper
    fix. Raising the logger severity is the available one.
    """
    try:
        import onnxruntime
        # 4 = FATAL. 3 would still let these through: they are logged at E.
        onnxruntime.set_default_logger_severity(4)
    except Exception:
        pass                                  # not fatal; you just get the noise


def _resolve_backend():
    """Pick a no-reference scorer. Cached. Never raises."""
    global _BACKEND, _BACKEND_NOTE
    if _BACKEND is not None:
        return _BACKEND

    try:
        from speechmos import dnsmos  # noqa: F401
        _quiet_onnxruntime()
        _BACKEND = "speechmos"
        _BACKEND_NOTE = "DNSMOS P.835 via speechmos"
        return _BACKEND
    except ModuleNotFoundError as e:
        # Name the missing module. `pip install speechmos` succeeding while the
        # import still fails means one of ITS dependencies is absent (commonly
        # onnxruntime), and "speechmos unavailable" would send you looking in
        # the wrong place entirely.
        missing = getattr(e, "name", None) or str(e)
        hint = ("pip install onnxruntime" if missing and "onnx" in missing
                else f"pip install {missing}")
        first = (f"speechmos import failed: no module '{missing}' -- "
                 f"speechmos itself may be installed but a dependency is not. "
                 f"Try: {hint}")
    except Exception as e:
        first = f"speechmos import failed ({type(e).__name__}: {e})"

    try:
        import torchaudio
        _ = torchaudio.pipelines.SQUIM_SUBJECTIVE  # noqa: F841
        _BACKEND = "squim"
        _BACKEND_NOTE = (f"{first}; falling back to torchaudio SQUIM. "
                         f"DIFFERENT SCALE -- not comparable to the manifest's "
                         f"dnsmos column or to emilia_bench's 3.0 filter.")
        return _BACKEND
    except Exception as e:
        _BACKEND = "none"
        _BACKEND_NOTE = (f"{first}; torchaudio SQUIM unavailable "
                         f"({type(e).__name__}). No-reference scoring disabled -- "
                         f"install with: pip install speechmos")
        return _BACKEND


def backend():
    """(name, explanation). Recorded in params.json and in every output row."""
    b = _resolve_backend()
    return b, _BACKEND_NOTE


# --------------------------------------------------------------------------- #
#  no-reference scoring
# --------------------------------------------------------------------------- #
def _as_16k_mono(y, sr):
    """Mono float32 at 16 kHz.

    Uses scipy rather than cascade_lib.resample so the no-reference path needs
    only numpy + scipy + a backend. cascade_lib pulls in librosa and the
    watermark models, which this does not need.
    """
    y = np.asarray(y, dtype="float32").squeeze()
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR_16K:
        import math

        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR_16K)
        y = resample_poly(y, SR_16K // g, int(sr) // g)
    return np.asarray(y, dtype="float32")


def _clean_speech(index=0, seconds=None):
    """Real speech from audio/clean_set, at 16 kHz. None if unavailable.

    MOS predictors are trained on speech. Feeding them tones is
    out-of-distribution and the output is meaningless -- the same objection we
    raise against feeding residuals to a neural detector. So anything that
    exercises a MOS backend uses real speech.

    Bandwidth matters as much as content. `audio/clean_set` is 8 kHz narrowband:
    upsampled to 16 kHz it has nothing above 4 kHz, and a MOS predictor scores
    that as poor -- measured, clean narrowband scored 2.57 while the same speech
    with added broadband noise scored 3.98, because the noise filled the empty
    top octave. So wideband sources are tried first and narrowband is a last
    resort.
    """
    import glob

    candidates = (
        sorted(glob.glob(os.path.join(ROOT, "audio", "client_original_16k.wav")))
        + sorted(glob.glob(os.path.join(ROOT, "audio", "clean_set",
                                        "clean_0[1-6]_speech*.wav")))
    )
    if not candidates:
        return None
    try:
        import soundfile as sf
        path = candidates[index % len(candidates)]
        y, sr = sf.read(path, dtype="float32")
        y = _as_16k_mono(y, sr)
        if seconds:
            # skip the first second: files often open on silence
            y = y[SR_16K:SR_16K + int(seconds * SR_16K)]
        return y if len(y) else None
    except Exception:
        return None


def no_reference(y, sr):
    """{'ovrl','sig','bak','backend'} on a 1-5 MOS scale, or Nones if disabled.

    `ovrl` is the primary field -- overall quality, the thing an attacker has to
    keep high. `sig` (speech) and `bak` (background) are kept because they
    separate two very different degradations: a music bed should hurt `bak` far
    more than `sig`, and that distinction is diagnostic.
    """
    b = _resolve_backend()
    out = {"ovrl": None, "sig": None, "bak": None, "backend": b}
    if b == "none":
        return out
    try:
        y16 = _as_16k_mono(y, sr)
        if len(y16) < SR_16K:              # under 1 s -- scorers are unreliable
            out["backend"] = b + "/too_short"
            return out
        if b == "speechmos":
            from speechmos import dnsmos
            r = dnsmos.run(y16, SR_16K)
            out["ovrl"] = float(r["ovrl_mos"])
            out["sig"] = float(r["sig_mos"])
            out["bak"] = float(r["bak_mos"])
        else:
            import torch
            import torchaudio
            model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
            # SQUIM_SUBJECTIVE needs a non-matching clean reference; use a
            # deterministic synthetic one so the score is reproducible.
            nmr = _nmr_reference()
            with torch.no_grad():
                mos = model(torch.from_numpy(y16)[None, :],
                            torch.from_numpy(nmr)[None, :])
            out["ovrl"] = float(mos.item())
    except Exception as e:
        out["backend"] = f"{b}/error:{type(e).__name__}"
    return out


_NMR = None


def _nmr_reference(seconds=5.0):
    """Clean speech for SQUIM's non-matching reference. Fallback backend only.

    SQUIM_SUBJECTIVE compares against a reference of unrelated CLEAN SPEECH.
    Handing it synthetic tones is out-of-distribution on both inputs and the MOS
    it returns is meaningless -- measured: the score rose as noise was added.
    So the reference must be real speech.

    Fixed file, fixed length, so scores are reproducible across runs.
    """
    global _NMR
    if _NMR is None:
        y = _clean_speech(index=0, seconds=seconds)
        if y is None:
            raise RuntimeError(
                "SQUIM needs a clean-speech reference and audio/clean_set was "
                "not found. Install speechmos instead: pip install speechmos")
        _NMR = y
    return _NMR


# --------------------------------------------------------------------------- #
#  combined
# --------------------------------------------------------------------------- #
def score(clean_path, deg_path, deg_audio=None, sr=None):
    """Every quality number for one degraded file, reference and no-reference.

    `clean_path` is the reference for PESQ/STOI/SNR. Pass `deg_audio`+`sr` to
    avoid re-reading a file already in memory.

    Returns a flat dict ready to be a CSV row. Any field may be None; nothing
    is estimated or filled in.
    """
    import cascade_lib as cl

    out = {"pesq": None, "snr_db": None, "si_snr_db": None, "stoi": None,
           "dnsmos_ovrl": None, "dnsmos_sig": None, "dnsmos_bak": None,
           "nr_backend": None}

    try:
        ref_based = cl.quality_metrics(clean_path, deg_path)
        out.update({k: ref_based.get(k) for k in
                    ("pesq", "snr_db", "si_snr_db", "stoi")})
    except Exception:
        pass                                # leave as None; never invent a value

    try:
        if deg_audio is None:
            deg_audio = cl.read_wav(deg_path, cl.SR_MASTER)
            sr = cl.SR_MASTER
        nr = no_reference(deg_audio, sr or cl.SR_MASTER)
        out["dnsmos_ovrl"] = nr["ovrl"]
        out["dnsmos_sig"] = nr["sig"]
        out["dnsmos_bak"] = nr["bak"]
        out["nr_backend"] = nr["backend"]
    except Exception as e:
        out["nr_backend"] = f"error:{type(e).__name__}"

    return out


def drop(clean_score, degraded_score):
    """How far quality fell from this clip's own clean score. nan if unknown.

    Positive = worse. Negative is legitimate and interesting: `denoise` can
    raise the score while stripping the watermark.
    """
    if clean_score is None or degraded_score is None:
        return float("nan")
    try:
        return float(clean_score) - float(degraded_score)
    except (TypeError, ValueError):
        return float("nan")


def usable(clean_score, degraded_score, strict=False):
    """Would an attacker still want this file?

    Relative: usable while quality has not fallen more than DROP_FLOOR below
    where THIS clip started. See the constants block for why absolute floors
    fail on this corpus.

    Returns None when either score is missing -- 'unknown' and 'unusable' are
    different, and collapsing them would silently drop conditions from the screen.
    """
    d = drop(clean_score, degraded_score)
    if not np.isfinite(d):
        return None
    return bool(d <= (DROP_FLOOR_STRICT if strict else DROP_FLOOR))


# --------------------------------------------------------------------------- #
#  self-test
# --------------------------------------------------------------------------- #
def _selftest():
    """Check the backend loads and that scores move in the right direction.

    Synthetic audio, so it runs anywhere. It does NOT validate DNSMOS itself --
    only that we can call it and that adding noise lowers the score.
    """
    b, note = backend()
    print(f"no-reference backend: {b}")
    print(f"  {note}\n")

    if b == "none":
        print("No no-reference scorer available. The music sweep CANNOT run:")
        print("  pip install --dry-run -r requirements_extra.txt")
        print("  pip install -r requirements_extra.txt")
        return False

    sr = SR_16K
    # Real speech, not tones. A tone-based check is not a check: measured with
    # synthetic tones, SQUIM scored clean 3.92 and heavy noise 3.95.
    clean = _clean_speech(index=3, seconds=8.0)
    if clean is None:
        print("audio/clean_set not found -- cannot validate a MOS backend on "
              "synthetic audio, so this self-test is INCONCLUSIVE, not passing.")
        return False

    rng = np.random.RandomState(0)
    p_sig = float(np.mean(clean ** 2))

    print(f"  {'condition':16s} {'ovrl':>7s} {'sig':>7s} {'bak':>7s}")
    scores = []
    for label, snr_db in (("clean", None), ("noise +20dB", 20.0),
                          ("noise +5dB", 5.0), ("noise -5dB", -5.0)):
        if snr_db is None:
            y = clean
        else:
            noise = rng.randn(len(clean)).astype("float32")
            noise *= np.sqrt(p_sig / (np.mean(noise ** 2) * 10 ** (snr_db / 10.0)))
            y = (clean + noise).astype("float32")
        r = no_reference(y, sr)
        scores.append(r["ovrl"])
        fmt = lambda v: f"{v:7.3f}" if v is not None else f"{'-':>7s}"  # noqa: E731
        print(f"  {label:16s} {fmt(r['ovrl'])} {fmt(r['sig'])} {fmt(r['bak'])}")

    got = [s for s in scores if s is not None]
    if len(got) < 2:
        print("\nFAIL -- backend returned no usable scores")
        return False
    # Monotone decreasing, not just endpoints: a backend that dips and recovers
    # cannot be used to locate a crossing point.
    ok = all(a >= b - 1e-6 for a, b in zip(got, got[1:]))
    print(f"\n{'OK -- score falls monotonically as noise rises' if ok else 'FAIL -- score is not monotone in noise; this backend cannot locate a crossing'}")
    print(f"usability floor: {DNSMOS_FLOOR} (matches emilia_bench DNSMOS_MIN), "
          f"strict {DNSMOS_FLOOR_STRICT}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
