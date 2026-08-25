"""
aligners_service.py -- audio alignment for service use. Self-contained.

Locates a short piece of audio (`query`) inside a longer recording
(`reference`) and tells you whether the answer can be trusted.

Source: WM_compare @ 5aea6cd, results/2026-08-18_frame-align-null.
Copy this file in as-is. It has no imports outside numpy (plus one optional
package, see below). Keep the commit reference above so it stays traceable.

--------------------------------------------------------------------------
THE ONE THING TO KNOW BEFORE USING THIS
--------------------------------------------------------------------------
Neither underlying method can say "no match". Given two completely unrelated
recordings they will still return an offset, confidently. We measured this:
across 7,200 mismatched pairs, an answer came back 100% of the time.

So the offset alone is meaningless. ALWAYS branch on `trusted`:

    r = align(reference, query)
    if r["trusted"]:
        use(r["offset_s"])
    else:
        show(r["message"])          # do not use the offset

A threshold calibrated only on audio that genuinely matches let through
30-41% of unrelated audio at 250 ms. The thresholds in this file were
calibrated against deliberately mismatched audio, which is why they are
different numbers.

--------------------------------------------------------------------------
INPUT REQUIREMENTS
--------------------------------------------------------------------------
* Both signals mono, float32/float64, 1-D numpy arrays.
* Both at the SAME sample rate. 16000 Hz is what everything here was
  calibrated at; pass `sr` if you use another rate.
* `reference` must be at least as long as `query`.

Decoding/resampling is deliberately not included -- use whatever the host
application already has, so there is one decode path rather than two.

--------------------------------------------------------------------------
LENGTH RULES (enforced, not advisory)
--------------------------------------------------------------------------
    < 250 ms      REFUSED. Returns ok=False with a warning. At this length
                  alignment is 11-16% accurate and a naively-set threshold
                  admits ~80% of unrelated audio. Returning nothing is safer
                  than returning a number.
    250 - 500 ms  Works, ~80%. Returns a warning alongside the answer.
    >= 500 ms     Normal operation, ~96%.

--------------------------------------------------------------------------
METHODS
--------------------------------------------------------------------------
    aof        default. `pip install audio-offset-finder`. More reliable at
               every length from 250 ms up. Answers land on a ~8 ms grid.
    gcc_phat   numpy only, no install. Sample-exact when it succeeds, but
               less reliable at short lengths. Use it if you need precision
               finer than 8 ms, or if the optional package is unavailable.

With method="auto" (the default) `aof` is used when installed and
`gcc_phat` otherwise, with a warning noting the substitution.

--------------------------------------------------------------------------
    python aligners_service.py        # self-check, verifies the install
--------------------------------------------------------------------------
"""
from __future__ import annotations

import time
import warnings

import numpy as np

__all__ = ["align", "self_check", "MIN_QUERY_MS", "WARN_QUERY_MS"]

DEFAULT_SR = 16000

# Refuse below this. See LENGTH RULES above.
MIN_QUERY_MS = 250.0
# Answer, but warn, below this.
WARN_QUERY_MS = 500.0

# Thresholds calibrated so that at most 1% of MISMATCHED audio is accepted.
# Measured at 16 kHz on 30 speech clips (results/2026-08-18_frame-align-null).

# aof's standard_score is stable across query length (measured 4.61-5.31),
# so one value works everywhere. 5.0 sits at the conservative end.
AOF_THRESHOLD = 5.0

# gcc_phat's peak-to-sidelobe ratio is NOT stable across query length -- it
# depends on how much audio is being searched, so it needs a table. These are
# measured operating points, not a fitted curve: each applies from its own
# length up to the next entry.
GCC_THRESHOLDS = [
    (250.0, 39.1),
    (500.0, 33.4),
    (1000.0, 36.2),
    (2000.0, 28.3),
]


def _gcc_threshold(query_ms: float) -> float:
    """Threshold for the largest measured length at or below `query_ms`."""
    thr = GCC_THRESHOLDS[0][1]
    for length_ms, value in GCC_THRESHOLDS:
        if query_ms >= length_ms:
            thr = value
    return thr


def _result(ok=False, trusted=False, offset_samples=float("nan"),
            confidence=float("nan"), score=float("nan"), method="",
            warning=None, message="", sr=DEFAULT_SR, runtime_s=0.0):
    off_s = offset_samples / float(sr) if np.isfinite(offset_samples) else float("nan")
    return {
        "ok": bool(ok),                    # did the method produce an answer
        "trusted": bool(trusted),          # BRANCH ON THIS, not on `ok`
        "offset_s": off_s,
        "offset_samples": float(offset_samples),
        "confidence": float(confidence),   # 0..1, comparable across methods
        "score": float(score),             # raw statistic the threshold uses
        "method": method,
        "warning": warning,                # non-None = answer is usable but weak
        "message": message,
        "runtime_s": round(float(runtime_s), 4),
    }


# --------------------------------------------------------------------------- #
#  gcc_phat -- numpy only
# --------------------------------------------------------------------------- #
def _gcc_phat(reference, query, sr):
    """Cross-correlation with the magnitude divided out, so only timing counts.

    Whitening makes the correlation peak a sharp spike rather than a broad
    hill, which is why this is sample-exact when it works. Confidence is
    peak-to-sidelobe ratio, squashed to [0,1] for reporting only -- the
    threshold is applied to the raw ratio.
    """
    n = len(reference) + len(query)
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))

    R = np.fft.rfft(reference, nfft)
    Q = np.fft.rfft(query, nfft)
    cross = R * np.conj(Q)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(cross / denom, nfft)
    cc = np.concatenate([cc[-(len(query) - 1):], cc[:len(reference)]])
    lags = np.arange(-(len(query) - 1), len(reference))

    mag = np.abs(cc)                       # abs() -> survives a polarity flip
    k = int(np.argmax(mag))
    peak = mag[k]

    frac = 0.0                             # sub-sample refinement
    if 0 < k < len(mag) - 1:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        d = a - 2 * b + c
        if abs(d) > 1e-20:
            frac = 0.5 * (a - c) / d

    guard = max(1, int(0.001 * sr))        # ignore +/-1 ms around the peak
    side = np.concatenate([mag[:max(0, k - guard)], mag[k + guard + 1:]])
    psr = float(peak / (np.mean(side) + 1e-20)) if len(side) else 1.0

    return {
        "offset": float(lags[k] + frac),
        "score": psr,
        "confidence": float(np.clip(np.log10(max(psr, 1.0)) / 3.0, 0.0, 1.0)),
        "ok": True,
    }


# --------------------------------------------------------------------------- #
#  aof -- optional package
# --------------------------------------------------------------------------- #
def _aof_available():
    try:
        from audio_offset_finder.audio_offset_finder import (  # noqa: F401
            find_offset_between_buffers)
        return True
    except Exception:
        return False


def _aof(reference, query, sr):
    from audio_offset_finder.audio_offset_finder import find_offset_between_buffers
    r = find_offset_between_buffers(reference.astype("float64"),
                                    query.astype("float64"), sr)
    off_s = float(r.get("time_offset", float("nan")))
    score = float(r.get("standard_score", float("nan")))
    return {
        "offset": off_s * sr,
        "score": score,
        "confidence": float(np.clip(score / 20.0, 0.0, 1.0)),
        "ok": bool(np.isfinite(off_s)),
    }


# --------------------------------------------------------------------------- #
#  sign calibration
# --------------------------------------------------------------------------- #
# Alignment libraries disagree about which direction an offset points. Rather
# than trusting documentation, derive it once from a synthetic clip with a
# known 1 s head crop. Cached per process; costs ~0.2 s on first use.
_SIGNS: dict[str, float] = {}


def _calibrate(sr=DEFAULT_SR):
    """{method: +1.0 or -1.0}. Derived from a known shift, never assumed."""
    if _SIGNS:
        return _SIGNS

    rng = np.random.RandomState(0)
    n = int(8.0 * sr)
    ref = (rng.randn(n) * 0.1).astype("float32")
    t = np.arange(n) / sr
    for f in (220.0, 440.0, 1300.0, 2700.0):
        ref += (0.15 * np.sin(2 * np.pi * f * t)).astype("float32")
    ref *= np.linspace(0.4, 1.0, n).astype("float32")   # break time symmetry

    k = int(1.0 * sr)
    query = ref[k:].copy()
    truth = float(k)

    for name, fn in (("gcc_phat", _gcc_phat), ("aof", _aof)):
        try:
            out = fn(ref, query, sr)
            got = out["offset"]
            if not out["ok"] or not np.isfinite(got):
                continue
            _SIGNS[name] = 1.0 if abs(got - truth) <= abs(-got - truth) else -1.0
        except Exception:
            continue                        # unavailable; align() reports it
    return _SIGNS


# --------------------------------------------------------------------------- #
#  the entry point
# --------------------------------------------------------------------------- #
def align(reference, query, sr: int = DEFAULT_SR, method: str = "auto") -> dict:
    """Locate `query` inside `reference`.

    Returns a dict; **branch on `trusted`**, never on `offset_s` alone.
    An untrusted result means "no usable answer", not "offset is zero".

        reference : 1-D mono array, the recording to search
        query     : 1-D mono array, the piece to find
        sr        : sample rate of BOTH, default 16000
        method    : "auto" | "aof" | "gcc_phat"
    """
    t0 = time.time()

    # ---- input validation ------------------------------------------------
    try:
        reference = np.asarray(reference, dtype="float32").squeeze()
        query = np.asarray(query, dtype="float32").squeeze()
    except Exception as e:
        return _result(message=f"could not read input as audio: {e}", sr=sr)

    if reference.ndim != 1 or query.ndim != 1:
        return _result(message="both inputs must be mono (1-D). Downmix first.",
                       sr=sr)
    if not len(reference) or not len(query):
        return _result(message="empty audio", sr=sr)
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(query))):
        return _result(message="audio contains NaN or Inf -- check the decoder",
                       sr=sr)
    if sr <= 0:
        return _result(message=f"invalid sample rate {sr}", sr=DEFAULT_SR)
    if len(query) > len(reference):
        return _result(
            message="query is longer than reference -- arguments are probably "
                    "swapped. reference = the long recording to search.", sr=sr)

    query_ms = 1000.0 * len(query) / float(sr)

    # ---- the length gate -------------------------------------------------
    if query_ms < MIN_QUERY_MS:
        return _result(
            sr=sr, runtime_s=time.time() - t0,
            message=(f"audio too short to align: {query_ms:.0f} ms, minimum is "
                     f"{MIN_QUERY_MS:.0f} ms. Below this the result would be "
                     f"unreliable (11-16% accurate) and could not be "
                     f"distinguished from unrelated audio. No offset returned."),
            warning="TOO_SHORT")

    warning = None
    if query_ms < WARN_QUERY_MS:
        warning = "SHORT"

    # ---- method selection ------------------------------------------------
    if method == "auto":
        chosen = "aof" if _aof_available() else "gcc_phat"
        if chosen == "gcc_phat":
            warning = warning or "AOF_UNAVAILABLE"
    elif method in ("aof", "gcc_phat"):
        chosen = method
    else:
        return _result(message=f"unknown method '{method}'", sr=sr)

    if chosen == "aof" and not _aof_available():
        return _result(
            message="aof requested but not installed: pip install "
                    "audio-offset-finder (or use method='gcc_phat')", sr=sr)

    # ---- run -------------------------------------------------------------
    signs = _calibrate(sr)
    fn = _aof if chosen == "aof" else _gcc_phat
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(reference, query, sr)
    except Exception as e:
        return _result(method=chosen, sr=sr, runtime_s=time.time() - t0,
                       message=f"{chosen} failed: {e}")

    if not out["ok"] or not np.isfinite(out["offset"]):
        return _result(method=chosen, sr=sr, runtime_s=time.time() - t0,
                       message=f"{chosen} returned no usable offset")

    offset = out["offset"] * signs.get(chosen, 1.0)
    score = out["score"]
    threshold = AOF_THRESHOLD if chosen == "aof" else _gcc_threshold(query_ms)
    trusted = bool(np.isfinite(score) and score >= threshold)

    if trusted and warning == "SHORT":
        msg = (f"aligned, but only {query_ms:.0f} ms of audio was available -- "
               f"about 80% of such cases are correct. Prefer "
               f">= {WARN_QUERY_MS:.0f} ms where possible.")
    elif trusted:
        msg = "aligned"
    else:
        msg = (f"no trustworthy alignment: score {score:.2f} is below the "
               f"{threshold:.2f} required at this length. The two files most "
               f"likely do not match. Do not use the offset.")

    return _result(ok=True, trusted=trusted, offset_samples=offset,
                   confidence=out["confidence"], score=score, method=chosen,
                   warning=warning, message=msg, sr=sr,
                   runtime_s=time.time() - t0)


# --------------------------------------------------------------------------- #
#  self-check
# --------------------------------------------------------------------------- #
def self_check(sr: int = DEFAULT_SR) -> bool:
    """Verify the install: a known offset is found, and unrelated audio is not.

    Synthetic audio, so this runs anywhere with no test files.
    """
    rng = np.random.RandomState(7)

    def tone(seconds, base, seed):
        r = np.random.RandomState(seed)
        n = int(seconds * sr)
        t = np.arange(n) / sr
        y = (r.randn(n) * 0.06).astype("float32")
        for m in (1, 2, 3, 5):
            y += (0.2 * np.sin(2 * np.pi * base * m * t)).astype("float32")
        env = (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.4 * t))).astype("float32")
        return (y * env).astype("float32")

    reference = tone(12.0, 190.0, 1)
    unrelated = tone(12.0, 610.0, 2)
    del rng

    print(f"aof installed: {_aof_available()}")
    print(f"sign calibration: {_calibrate(sr)}\n")

    ok = True
    for label, method in (("auto", "auto"), ("gcc_phat", "gcc_phat")):
        if method == "aof" and not _aof_available():
            continue
        print(f"--- method={label} ---")

        # 1. a known offset must be recovered
        start = int(4.0 * sr)
        piece = reference[start:start + int(1.0 * sr)]
        r = align(reference, piece, sr, method=method)
        err_ms = abs(r["offset_samples"] - start) / sr * 1000.0
        good = r["trusted"] and err_ms < 50.0
        ok &= good
        print(f"  known offset   trusted={r['trusted']!s:5s} "
              f"err={err_ms:8.2f}ms  {'OK' if good else 'FAIL'}")

        # 2. unrelated audio must NOT be trusted
        r = align(unrelated, piece, sr, method=method)
        good = not r["trusted"]
        ok &= good
        print(f"  unrelated      trusted={r['trusted']!s:5s} "
              f"score={r['score']:8.2f}  {'OK' if good else 'FAIL — would '
              'have accepted unrelated audio'}")

        # 3. too-short input must be refused
        r = align(reference, reference[:int(0.1 * sr)], sr, method=method)
        good = (not r["ok"]) and r["warning"] == "TOO_SHORT"
        ok &= good
        print(f"  100ms input    refused={not r['ok']!s:5s} "
              f"{'OK' if good else 'FAIL'}")
        print()

    print("SELF-CHECK PASSED" if ok else "SELF-CHECK FAILED — do not deploy")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
