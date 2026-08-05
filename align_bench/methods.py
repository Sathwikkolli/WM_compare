"""
align_bench/methods.py -- the 6 alignment methods behind one interface.

Every method returns the SAME dict so score.py doesn't care which ran:

    {
      'offset':     float,   # samples, convention: ref_index - dist_index
      'segments':   [(ref_s, ref_e, dist_s, dist_e), ...],   # [] if single-offset
      'confidence': float,   # [0,1], or nan if the method has no score
      'ok':         bool,    # False = crashed / unavailable
      'note':       str,
    }

SIGN CONVENTION IS AUTO-CALIBRATED. Third-party tools disagree about which way
an offset points, and a silent sign flip would poison every number. Rather than
trusting docs, calibrate() feeds each method a synthetic clip with a KNOWN shift
and derives the multiplier. Run it once at startup; run_bench.py does this and
records the result in params.json.

Methods
  gcc_phat        scipy only. Our control/floor. Sample-accurate, single offset.
                  Uses argmax(|cc|) so it survives inverse_polarity, and reports
                  the detected polarity.
  aof             bbc/audio-offset-finder -- MFCC cross-correlation.
  audalign_fp     audalign FingerprintRecognizer -- the only multi-segment one.
  audalign_corr   audalign CorrelationRecognizer.
  audalign_spec   audalign CorrelationSpectrogramRecognizer.
  dtw_subseq      librosa subsequence DTW -- the warping cases.
"""
import os
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  1. GCC-PHAT -- the control
# --------------------------------------------------------------------------- #
def gcc_phat(ref, dist, sr, max_shift_s=None):
    """Generalised Cross-Correlation with Phase Transform.

    Whitens both spectra so only TIMING survives -- gain, EQ and coloured noise
    stop smearing the peak. Sub-sample precision via parabolic interpolation.
    Confidence = peak-to-sidelobe ratio, squashed to [0,1].
    """
    n = len(ref) + len(dist)
    nfft = 1 << int(np.ceil(np.log2(n)))          # zero-pad: linear, not circular

    R = np.fft.rfft(ref, nfft)
    D = np.fft.rfft(dist, nfft)
    cross = R * np.conj(D)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(cross / denom, nfft)        # PHAT weighting
    cc = np.concatenate([cc[-(len(dist) - 1):], cc[:len(ref)]])
    lags = np.arange(-(len(dist) - 1), len(ref))

    if max_shift_s is not None:
        m = np.abs(lags) <= int(max_shift_s * sr)
        cc, lags = cc[m], lags[m]

    mag = np.abs(cc)                              # abs -> survives polarity flip
    k = int(np.argmax(mag))
    peak = mag[k]

    # parabolic interpolation for sub-sample precision
    frac = 0.0
    if 0 < k < len(mag) - 1:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-20:
            frac = 0.5 * (a - c) / d

    # peak-to-sidelobe: ignore +/-1 ms around the peak
    guard = max(1, int(0.001 * sr))
    side = np.concatenate([mag[:max(0, k - guard)], mag[k + guard + 1:]])
    psr = float(peak / (np.mean(side) + 1e-20)) if len(side) else 1.0

    return {
        "offset": float(lags[k] + frac),
        "segments": [],
        "confidence": float(np.clip(np.log10(max(psr, 1.0)) / 3.0, 0.0, 1.0)),
        "ok": True,
        "note": f"psr={psr:.1f} polarity={'neg' if cc[k] < 0 else 'pos'}",
    }


# --------------------------------------------------------------------------- #
#  2. bbc/audio-offset-finder
# --------------------------------------------------------------------------- #
def aof(ref, dist, sr, **_):
    try:
        from audio_offset_finder.audio_offset_finder import find_offset_between_buffers
    except Exception as e:
        return _fail(f"audio-offset-finder unavailable: {e}")
    try:
        r = find_offset_between_buffers(ref.astype("float64"),
                                        dist.astype("float64"), sr)
        off_s = float(r.get("time_offset", np.nan))
        score = float(r.get("standard_score", np.nan))
        return {
            "offset": off_s * sr,
            "segments": [],
            # standard_score is unbounded; ~10+ is a solid match in their docs
            "confidence": float(np.clip(score / 20.0, 0.0, 1.0)),
            "ok": np.isfinite(off_s),
            "note": f"standard_score={score:.2f}",
        }
    except Exception as e:
        return _fail(f"aof error: {e}")


# --------------------------------------------------------------------------- #
#  3-5. audalign  (works on FILES, so we stage temp wavs)
# --------------------------------------------------------------------------- #
def _audalign(ref, dist, sr, recognizer_name):
    try:
        import audalign as ad
        import soundfile as sf
    except Exception as e:
        return _fail(f"audalign unavailable: {e}")

    ctor = {
        "fingerprint": "FingerprintRecognizer",
        "correlation": "CorrelationRecognizer",
        "spectrogram": "CorrelationSpectrogramRecognizer",
    }[recognizer_name]
    if not hasattr(ad, ctor):
        return _fail(f"audalign has no {ctor}")

    d = tempfile.mkdtemp(prefix="alignbench_")
    fr = os.path.join(d, "ref.wav")
    fd = os.path.join(d, "dist.wav")
    try:
        sf.write(fr, ref.astype("float32"), sr)
        sf.write(fd, dist.astype("float32"), sr)
        rec = getattr(ad, ctor)()
        res = ad.align_files(fr, fd, recognizer=rec)
        off_s, conf, segs = _dig_audalign(res, os.path.basename(fd))
        return {
            "offset": -off_s * sr if np.isfinite(off_s) else np.nan,
            "segments": segs,
            "confidence": conf,
            "ok": bool(np.isfinite(off_s)),
            "note": f"{ctor}",
        }
    except Exception as e:
        return _fail(f"{ctor} error: {e}")
    finally:
        for p in (fr, fd):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


def _dig_audalign(res, dist_name):
    """audalign's return shape has moved between versions -- dig defensively.

    We want: seconds the distorted file must move to line up, a confidence, and
    (fingerprinting only) per-segment matches. Sign is fixed by calibrate().
    """
    off, conf, segs = np.nan, float("nan"), []
    if not isinstance(res, dict):
        return off, conf, segs

    shifts = res.get("names_and_paths") or res.get("match_info") or res
    for key in (dist_name, os.path.splitext(dist_name)[0]):
        if isinstance(shifts, dict) and key in shifts:
            v = shifts[key]
            if isinstance(v, (int, float)):
                off = float(v)
            elif isinstance(v, dict):
                for k in ("offset_seconds", "offset", "time_offset"):
                    if k in v:
                        off = float(v[k])
                        break
                for k in ("confidence", "match_confidence"):
                    if k in v:
                        conf = float(v[k])
                        break
            break

    if not np.isfinite(off):
        for k in ("offset_seconds", "offset", "time_offset"):
            if k in res and isinstance(res[k], (int, float)):
                off = float(res[k])
                break
    if np.isfinite(conf):
        conf = float(np.clip(conf / 100.0, 0.0, 1.0)) if conf > 1.0 else conf
    return off, conf, segs


def audalign_fp(ref, dist, sr, **_):
    return _audalign(ref, dist, sr, "fingerprint")


def audalign_corr(ref, dist, sr, **_):
    return _audalign(ref, dist, sr, "correlation")


def audalign_spec(ref, dist, sr, **_):
    return _audalign(ref, dist, sr, "spectrogram")


# --------------------------------------------------------------------------- #
#  6. librosa subsequence DTW
# --------------------------------------------------------------------------- #
def dtw_subseq(ref, dist, sr, hop=512, **_):
    """Subsequence DTW: lets `dist` match any INTERIOR stretch of `ref`.

    Classic DTW forces both endpoints to match, which is exactly wrong when the
    distorted clip is a crop. subseq=True is the whole point.
    """
    try:
        import librosa
    except Exception as e:
        return _fail(f"librosa unavailable: {e}")
    try:
        X = librosa.feature.mfcc(y=ref.astype("float32"), sr=sr,
                                 n_mfcc=20, hop_length=hop)
        Y = librosa.feature.mfcc(y=dist.astype("float32"), sr=sr,
                                 n_mfcc=20, hop_length=hop)
        X = librosa.util.normalize(X, axis=0)
        Y = librosa.util.normalize(Y, axis=0)

        D, wp = librosa.sequence.dtw(X=X, Y=Y, subseq=True, backtrack=True)
        wp = np.asarray(wp)[::-1]                 # librosa returns it reversed
        lag_frames = wp[:, 0] - wp[:, 1]
        offset = float(np.median(lag_frames) * hop)

        # spread of the per-frame lag: tight = confident, wide = warped or lost
        spread = float(np.std(lag_frames) * hop / sr)
        conf = float(np.clip(1.0 - spread, 0.0, 1.0))

        return {"offset": offset, "segments": [], "confidence": conf,
                "ok": True, "note": f"lag_spread={spread * 1000:.0f}ms"}
    except Exception as e:
        return _fail(f"dtw error: {e}")


# --------------------------------------------------------------------------- #
#  registry / plumbing
# --------------------------------------------------------------------------- #
def _fail(note):
    return {"offset": np.nan, "segments": [], "confidence": float("nan"),
            "ok": False, "note": note}


METHODS = {
    "gcc_phat":      gcc_phat,
    "aof":           aof,
    "audalign_fp":   audalign_fp,
    "audalign_corr": audalign_corr,
    "audalign_spec": audalign_spec,
    "dtw_subseq":    dtw_subseq,
}

MULTISEG_CAPABLE = {"audalign_fp"}


def run(name, ref, dist, sr, sign=1.0):
    """Run one method, timed, with the calibrated sign applied."""
    t0 = time.time()
    try:
        out = METHODS[name](ref, dist, sr)
    except Exception as e:
        out = _fail(f"uncaught: {e}")
    out["runtime_s"] = time.time() - t0
    if out["ok"] and np.isfinite(out["offset"]):
        out["offset"] *= sign
    return out


def calibrate(sr=16000, dur_s=8.0, shift_s=1.0, seed=0):
    """Derive each method's sign convention from a KNOWN shift.

    Build noise-like reference audio, crop `shift_s` off the head (true offset
    = +shift_s*sr by our convention), and see what each method reports. If it
    comes back negative, that method's sign multiplier is -1.

    Returns {method_name: +1.0 | -1.0 | nan}. nan = method never produced a
    usable answer; treat as broken, not as +1.
    """
    rng = np.random.RandomState(seed)
    n = int(dur_s * sr)
    # filtered noise + tones: broadband but not flat, so every feature front-end
    # (MFCC, spectral peaks, raw correlation) has something to lock onto
    ref = rng.randn(n).astype("float32") * 0.1
    t = np.arange(n) / sr
    for f in (220.0, 440.0, 1300.0, 2700.0):
        ref += 0.15 * np.sin(2 * np.pi * f * t).astype("float32")
    ref *= np.linspace(0.4, 1.0, n).astype("float32")      # break time symmetry

    k = int(shift_s * sr)
    dist = ref[k:].copy()
    truth = float(k)

    signs = {}
    for name in METHODS:
        out = run(name, ref, dist, sr, sign=1.0)
        if not out["ok"] or not np.isfinite(out["offset"]):
            signs[name] = float("nan")
            continue
        got = out["offset"]
        if abs(got - truth) <= abs(-got - truth):
            signs[name] = 1.0
        else:
            signs[name] = -1.0
    return signs


if __name__ == "__main__":
    print("calibrating sign conventions (synthetic 1.0 s head crop)...")
    for k, v in calibrate().items():
        state = "BROKEN / not installed" if not np.isfinite(v) else f"sign={v:+.0f}"
        print(f"  {k:16s} {state}")
