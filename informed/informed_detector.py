"""
informed/informed_detector.py -- detection that gets to see the original.

Blind detection must find the watermark buried inside the host speech, and the
host is the dominant interference. Given the unwatermarked original we can
subtract it, leaving watermark + attack noise, and then match against the
watermark we know we embedded.

    align -> shift -> gain-match -> subtract -> correlate

WHY CORRELATION, AND NOT A RECONSTRUCTION

Under Neyman-Pearson the optimal detector is the likelihood ratio test, and for
an additive watermark in Gaussian noise the LRT reduces exactly to a correlation
detector (matched filter). Normalised correlation is the practical form: bounded
in [-1, 1] and immune to the level changes many attacks introduce.

A tempting alternative is to rebuild a cleaner file and feed it back to AWARE's
own detector, so both arms share a scale. It is circular: for an additive attack
`attacked = wm + n`, the residual is `w + n`, and adding that back to `org`
returns `wm + n` -- the file we started with. THE GAIN LIVES IN THE STATISTIC,
NOT IN RECONSTRUCTION. Blind AWARE searches for `w` inside `org + w + n`;
this searches for it inside `w + n`.

The cost of that is unavoidable: the two arms produce different quantities on
different scales, so their thresholds MUST be matched on a null before their
crossings can be compared. See PHASE_B_PLAN.md.

WINDOWED IS PRIMARY, AND THAT WAS FIXED BEFORE THE RUN

AWARE hops its detector in 42 ms steps, so a global correlation over 10 s
dilutes a locally-surviving watermark. Windowed is more sensitive and will very
likely score higher -- which is exactly why the choice is registered in advance.
`score()` returns BOTH, so the effect of the choice stays visible.

    python informed_detector.py        # self-test on synthetic audio
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, os.path.join(ROOT, "align_bench"))

SR_16K = 16000

# One AWARE(20bps) detector hop. Registered as primary in PHASE_B_PLAN.md.
WINDOW_MS = 42.0
WINDOW_OVERLAP = 0.5

# Windows quieter than this (relative to the clip's mean residual energy) are
# skipped: a near-silent window gives 0/0 and an arbitrary correlation.
MIN_WINDOW_ENERGY = 0.05

# Alignment search range. Attacks here do not deliberately desynchronise; codec
# delay is the real case and is milliseconds. A wide search would only invite
# spurious peaks.
MAX_SHIFT_S = 1.0

FIR_TAPS = 64          # 4 ms at 16 kHz -- enough for an EQ curve, not for reverb


# --------------------------------------------------------------------------- #
#  alignment
# --------------------------------------------------------------------------- #
def find_offset(org, attacked, sr, max_shift_s=MAX_SHIFT_S):
    """Sample offset of `attacked` relative to `org`, with the fractional part.

    Uses gcc_phat from align_bench -- the only aligner measured as sample-exact
    (2026-08-18_frame-align-null S5). `aof` quantises to a ~7.7 ms grid, which
    for subtraction means the host smears instead of cancelling.
    """
    import methods as M
    out = M.gcc_phat(np.asarray(org, dtype="float32"),
                     np.asarray(attacked, dtype="float32"),
                     sr, max_shift_s=max_shift_s)
    psr = float("nan")
    note = out.get("note", "")
    if "psr=" in note:
        try:
            psr = float(note.split("psr=")[1].split()[0])
        except (ValueError, IndexError):
            pass
    return float(out["offset"]), psr


def shift_signal(x, delta):
    """Shift `x` by `delta` samples. Negative advances, positive delays.

    SIGN, because getting it backwards is silent and fatal. gcc_phat's
    convention is `offset = ref_index - dist_index`. If `attacked` is delayed
    relative to `org`, content at org index k lands at attacked index k+d, so
    offset = -d. To put `attacked` back on org's timebase we must ADVANCE it by
    d, i.e. shift by -d = offset. So the caller passes `offset` unchanged --
    negating it doubles the misalignment instead of removing it.

    The integer part is done by slicing with zero fill, not by FFT rotation.
    An FFT shift is CIRCULAR: it wraps the tail of the signal round to the
    front, which for a real recording splices unrelated audio into the region
    being correlated. Only the sub-sample remainder goes through the FFT, where
    |frac| < 1 makes the wrap negligible.

    The fractional part matters: rounding to the nearest sample leaves up to
    half a sample of misalignment, and at 16 kHz that is enough host leakage to
    swamp a watermark sitting 35 dB down.
    """
    x = np.asarray(x, dtype="float32")
    if abs(delta) < 1e-9:
        return x
    n = len(x)
    k_int = int(np.round(delta))
    frac = float(delta - k_int)

    if k_int > 0:                                   # delay
        y = np.concatenate([np.zeros(k_int, dtype="float32"), x[:n - k_int]])
    elif k_int < 0:                                 # advance
        y = np.concatenate([x[-k_int:], np.zeros(-k_int, dtype="float32")])
    else:
        y = x.copy()

    if abs(frac) > 1e-6:
        Y = np.fft.rfft(y)
        kk = np.arange(Y.shape[0])
        Y = Y * np.exp(-2j * np.pi * kk * frac / n)
        y = np.fft.irfft(Y, n)
    return np.asarray(y, dtype="float32")


# Kept so older callers do not break silently on a rename.
frac_shift = shift_signal


# --------------------------------------------------------------------------- #
#  host removal
# --------------------------------------------------------------------------- #
def gain_match(org, attacked):
    """Least-squares scalar: a = <attacked, org> / <org, org>.

    Correct when the attack added something. WRONG for anything that filters --
    a single number cannot undo an EQ curve, and the leftover host then swamps
    the watermark. `fir_match` is the fallback, reported as secondary.
    """
    d = float(np.dot(org, org))
    if d <= 1e-20:
        return 1.0
    return float(np.dot(attacked, org) / d)


def fir_match(org, attacked, taps=FIR_TAPS):
    """Least-squares FIR h minimising ||attacked - h*org||, via Wiener-Hopf.

    Solves the normal equations with the autocorrelation Toeplitz matrix rather
    than building an N x taps design matrix, which for a 10 s clip would be
    160000 x 64.

    Still a "generic inverse parameter" in the sense of the bake-off's validity
    rule -- it estimates a filter, it does not inject signal.
    """
    from scipy.linalg import solve_toeplitz

    org = np.asarray(org, dtype="float64")
    attacked = np.asarray(attacked, dtype="float64")
    n = min(len(org), len(attacked))
    org, attacked = org[:n], attacked[:n]

    full = np.correlate(org, org, mode="full")
    mid = len(full) // 2
    r_xx = full[mid:mid + taps]
    if r_xx[0] <= 1e-20:
        return None
    r_xx = r_xx.copy()
    r_xx[0] *= 1.0 + 1e-6                       # ridge, keeps it invertible

    r_xy = np.array([float(np.dot(attacked[k:], org[:n - k]))
                     for k in range(taps)])
    try:
        h = solve_toeplitz((r_xx, r_xx), r_xy)
    except Exception:
        return None
    return np.asarray(h, dtype="float32")


# --------------------------------------------------------------------------- #
#  correlation
# --------------------------------------------------------------------------- #
def _ncorr(a, b):
    """Normalised correlation in [-1, 1]. nan if either side is silent."""
    na = float(np.sqrt(np.dot(a, a)))
    nb = float(np.sqrt(np.dot(b, b)))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def windowed_ncorr(w_clean, w_obs, sr, window_ms=WINDOW_MS,
                   overlap=WINDOW_OVERLAP):
    """Mean normalised correlation over 42 ms windows. THE PRIMARY STATISTIC.

    Also returns the per-window values, because their spread separates
    "uniformly weakened" from "destroyed in patches", which the mean alone hides.
    """
    n = min(len(w_clean), len(w_obs))
    w_clean, w_obs = w_clean[:n], w_obs[:n]
    win = max(8, int(window_ms / 1000.0 * sr))
    hop = max(1, int(win * (1.0 - overlap)))
    if n < win:
        return _ncorr(w_clean, w_obs), np.array([])

    energy = np.array([float(np.dot(w_clean[i:i + win], w_clean[i:i + win]))
                       for i in range(0, n - win + 1, hop)])
    if not len(energy):
        return float("nan"), np.array([])
    thresh = MIN_WINDOW_ENERGY * float(np.mean(energy)) if np.mean(energy) > 0 else 0.0

    vals = []
    for j, i in enumerate(range(0, n - win + 1, hop)):
        if energy[j] < thresh:
            continue                            # near-silent: 0/0, meaningless
        v = _ncorr(w_clean[i:i + win], w_obs[i:i + win])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals, dtype=float)
    return (float(vals.mean()) if len(vals) else float("nan")), vals


# --------------------------------------------------------------------------- #
#  the entry point
# --------------------------------------------------------------------------- #
def score(org, wm, attacked, sr=SR_16K, method="scalar", min_psr=None):
    """Informed detection score for one attacked file.

        org       clean original (we hold it -- that is the premise)
        wm        the watermarked file we produced, unattacked
        attacked  what the detector actually receives
        method    "scalar" (registered primary) or "fir" (secondary)

    Returns a dict. `corr_windowed` is the primary statistic; `corr_global` is
    reported so the windowing choice stays auditable.

    `ok=False` means the measurement could not be made (alignment failed, silent
    input). It must be excluded and COUNTED, never scored as a detection failure
    -- otherwise the informed arm is charged for our plumbing.
    """
    org = np.asarray(org, dtype="float32")
    wm = np.asarray(wm, dtype="float32")
    attacked = np.asarray(attacked, dtype="float32")

    out = {"corr_windowed": float("nan"), "corr_global": float("nan"),
           "corr_window_sd": float("nan"), "n_windows": 0,
           "offset": float("nan"), "psr": float("nan"), "gain": float("nan"),
           "method": method, "ok": False, "note": ""}

    n = min(len(org), len(wm))
    if n < int(0.5 * sr):
        out["note"] = "input shorter than 0.5 s"
        return out
    org, wm = org[:n], wm[:n]

    # The reference pattern: exactly what we embedded. We made this file, so it
    # is known rather than estimated.
    w_clean = wm - org
    if float(np.dot(w_clean, w_clean)) <= 1e-20:
        out["note"] = "watermark residual is zero -- embed failed?"
        return out

    # ---- align ------------------------------------------------------------
    try:
        offset, psr = find_offset(org, attacked, sr)
    except Exception as e:
        out["note"] = f"alignment failed: {type(e).__name__}"
        return out
    out["offset"], out["psr"] = offset, psr
    if min_psr is not None and np.isfinite(psr) and psr < min_psr:
        out["note"] = f"alignment not trusted (psr={psr:.1f} < {min_psr})"
        return out

    # Shift `attacked` onto `org`'s timebase, fractional part included.
    # `offset`, NOT `-offset`: see shift_signal's docstring for the convention.
    shifted = shift_signal(attacked, offset) if abs(offset) > 1e-9 else attacked
    m = min(len(shifted), n)
    if m < int(0.5 * sr):
        out["note"] = "overlap after alignment too short"
        return out
    o, s, wc = org[:m], shifted[:m], w_clean[:m]

    # ---- remove the host --------------------------------------------------
    if method == "fir":
        h = fir_match(o, s)
        if h is None:
            out["note"] = "FIR estimation failed"
            return out
        pred = np.convolve(o, h, mode="full")[:m].astype("float32")
        out["gain"] = float(np.sum(h))
    else:
        a = gain_match(o, s)
        pred = (a * o).astype("float32")
        out["gain"] = a

    w_obs = (s - pred).astype("float32")

    # ---- correlate --------------------------------------------------------
    cw, vals = windowed_ncorr(wc, w_obs, sr)
    out["corr_windowed"] = cw
    out["corr_global"] = _ncorr(wc, w_obs)
    out["n_windows"] = int(len(vals))
    out["corr_window_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["ok"] = bool(np.isfinite(cw))
    if not out["ok"]:
        out["note"] = "correlation undefined (silent residual?)"
    return out


# --------------------------------------------------------------------------- #
#  self-test
# --------------------------------------------------------------------------- #
def _selftest():
    """Does the chain recover a known watermark, and reject one that is absent?

    Synthetic, so it runs anywhere. It checks the four things that would each
    silently break the experiment:

      1. watermarked audio scores high, unwatermarked scores ~0
      2. a time shift is undone (this is what the aligner is for)
      3. a gain change is undone
      4. an EQ-type change is undone by FIR but NOT by scalar -- the reason the
         FIR variant exists at all
    """
    sr = SR_16K
    rng = np.random.RandomState(0)
    n = int(6.0 * sr)
    t = np.arange(n) / sr

    # speech-like host: harmonics with a syllabic envelope
    org = np.zeros(n, dtype="float32")
    for k, f in enumerate((140.0, 280.0, 560.0, 1120.0, 2240.0)):
        org += (0.3 / (k + 1) * np.sin(2 * np.pi * f * t)).astype("float32")
    org += (rng.randn(n) * 0.01).astype("float32")
    org *= (0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 0.7 * t))).astype("float32")

    # watermark: 35 dB below the host, the realistic regime
    w = (rng.randn(n) * 0.01).astype("float32")
    w *= float(np.sqrt(np.mean(org ** 2))) / (float(np.sqrt(np.mean(w ** 2))) + 1e-12)
    w *= 10 ** (-35.0 / 20.0)
    wm = (org + w).astype("float32")
    print(f"watermark is {10*np.log10(np.mean(w**2)/np.mean(org**2)):.1f} dB "
          f"below the host\n")

    def noisy(x, snr_db):
        p = float(np.mean(x ** 2))
        e = rng.randn(len(x)).astype("float32")
        e *= np.sqrt(p / (np.mean(e ** 2) * 10 ** (snr_db / 10.0)))
        return (x + e).astype("float32")

    # Expectations come from theory, not from a number picked by hand.
    #
    # The residual is w + n, so correlating it against w gives
    #     rho = |w| / sqrt(|w|^2 + |n|^2)
    # which is fully determined by the watermark and noise powers. Checking
    # against that validates the IMPLEMENTATION rather than asserting some
    # arbitrary bar -- an earlier version of this test demanded rho > 0.30 and
    # flagged a correct detector as broken, because at -35 dB watermark and
    # +20 dB SNR the true answer is 0.175.
    p_w = float(np.mean(w ** 2))
    p_host = float(np.mean(org ** 2))

    def expected_rho(snr_db):
        p_n = p_host / (10.0 ** (snr_db / 10.0))
        return float(np.sqrt(p_w / (p_w + p_n)))

    ok = True
    print(f"  {'case':34s} {'windowed':>9s} {'expect':>8s} {'offset':>8s} "
          f"{'gain':>7s}")

    def run(label, attacked, method="scalar", expect=None, tol=0.45):
        """expect: a number (theoretical rho), 'null', or None to just report."""
        nonlocal ok
        r = score(org, wm, attacked, sr, method=method)
        good, exp_s = True, ""
        if expect == "null":
            good = r["ok"] and abs(r["corr_windowed"]) < 0.05
            exp_s = "~0"
        elif isinstance(expect, float):
            exp_s = f"{expect:.4f}"
            good = (r["ok"] and np.isfinite(r["corr_windowed"])
                    and abs(r["corr_windowed"] - expect) <= tol * expect)
        ok &= good
        mark = "" if good else "   <-- FAIL"
        print(f"  {label:34s} {r['corr_windowed']:9.4f} {exp_s:>8s} "
              f"{r['offset']:8.1f} {r['gain']:7.3f}{mark}")
        return r

    # 1. present vs absent, against the predicted correlation
    run("watermarked, +20 dB noise", noisy(wm, 20), expect=expected_rho(20))
    run("UNwatermarked, +20 dB noise", noisy(org, 20), expect="null")
    run("watermarked, +5 dB noise", noisy(wm, 5), expect=expected_rho(5))
    run("UNwatermarked, +5 dB noise", noisy(org, 5), expect="null")

    # 2. time shift -- the aligner's job. Must land on the SAME value as the
    # unshifted case; if it does not, the offset is being applied wrongly, which
    # is silent and fatal.
    shifted = np.concatenate([np.zeros(137, dtype="float32"), wm])[:n]
    run("watermarked, shifted 137 samples", noisy(shifted, 20),
        expect=expected_rho(20))

    # 3. gain change -- must also land on the same value, proving the scalar
    # match removed it rather than the correlation merely tolerating it.
    run("watermarked, x0.4 gain", noisy((wm * 0.4).astype("float32"), 20),
        expect=expected_rho(20))

    # 4. EQ-type change: scalar should struggle, FIR should not
    from scipy.signal import lfilter
    b = np.array([1.0, -0.7, 0.2], dtype="float64")     # a mild colouring filter
    eq = lfilter(b, [1.0], wm).astype("float32")
    r_s = run("watermarked, EQ  [scalar]", noisy(eq, 20))
    r_f = run("watermarked, EQ  [fir]", noisy(eq, 20), method="fir")
    if np.isfinite(r_s["corr_windowed"]) and np.isfinite(r_f["corr_windowed"]):
        better = r_f["corr_windowed"] > r_s["corr_windowed"]
        print(f"\n  FIR beats scalar under EQ: {better} "
              f"({r_f['corr_windowed']:.4f} vs {r_s['corr_windowed']:.4f})")
        if not better:
            print("  NOTE: not a hard failure, but the FIR variant exists for "
                  "exactly this case -- worth understanding before relying on it.")

    print("\n" + ("SELF-TEST PASSED" if ok else
                  "SELF-TEST FAILED -- do not run Phase B until this passes"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
