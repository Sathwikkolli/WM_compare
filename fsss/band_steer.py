"""
fsss/band_steer.py -- point a frozen AWARE at any frequency band.

AWARE writes in a fixed window (1000-4000 Hz at 16 kHz); that window is baked
into the trained detector and cannot be moved by asking. This module moves the
AUDIO instead: it maps an arbitrary target band [f1, f2] onto AWARE's window
with a frequency shift followed by a resample, runs whatever the caller wants
on the result, then inverts the map exactly.

    analyze   : x            -> (lo, host)      host is what AWARE sees
    <caller embeds in host, or just detects>
    synthesize: (lo, host)   -> y

Why shift AND resample
----------------------
A resample alone is a pure scaling, so it preserves frequency RATIOS: it can
never carry [500, 4000] (ratio 8) onto [1000, 4000] (ratio 4). A shift alone
cannot change the width. Composing them gives an affine map with exactly enough
freedom to pin both endpoints:

    perceived(f) = (f + shift) / r,     r = (f2 - f1) / (a2 - a1)
                                        shift = a1 * r - f1

For [500, 4000] -> [1000, 4000] that is r = 7/6 and shift = +666.67 Hz. The
shift lands on sr/24, so the modulator has period 24 samples -- exact, no drift.

Why nothing gets erased
-----------------------
`host` has more samples than degrees of freedom (74667 vs 64000 for a 4 s clip
at r = 7/6), so most of its spectrum is interpolation, not signal, and anything
written there dies on the way back. That is survivable only because the map
puts the REAL content at 1000-4000 -- precisely where AWARE writes. Aligning to
baseband instead (shift to 0, no stretch) is what forces the target band to be
at least a2 Hz wide; aligning to the window removes that constraint.

`guard_hz` widens the analysis filter past the nominal band so AWARE's window
sits strictly inside real content rather than on the filter's roll-off. It must
exceed the FIR transition width (~4*sr/numtaps).

Standalone smoke test (no AWARE / no GPU needed):
    python -m fsss.band_steer
"""

from fractions import Fraction

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert, resample_poly

__all__ = ["BandPlan", "split", "to_model", "from_model", "analyze", "synthesize"]

SR = 16000
AWARE_BAND = (1000.0, 4000.0)
NUMTAPS = 513
GUARD_HZ = 150.0


# --------------------------------------------------------------------------- #
#  the map
# --------------------------------------------------------------------------- #
class BandPlan:
    """Affine frequency map carrying [f1, f2] onto AWARE's window.

    Attributes worth reading: `up`/`down` (resample ratio as integers), `shift`
    (Hz, positive means the band moves up), `f_lo`/`f_hi` (the widened band the
    analysis filter actually passes).
    """

    def __init__(self, f1, f2, aware_band=AWARE_BAND, sr=SR,
                 guard_hz=GUARD_HZ, max_den=64):
        a1, a2 = float(aware_band[0]), float(aware_band[1])
        f1, f2 = float(f1), float(f2)
        if not 0.0 < f1 < f2 < sr / 2.0:
            raise ValueError(f"band {(f1, f2)} outside (0, {sr / 2})")

        frac = Fraction((f2 - f1) / (a2 - a1)).limit_denominator(max_den)
        self.up, self.down = frac.numerator, frac.denominator
        self.r = self.up / self.down
        self.shift = a1 * self.r - f1

        self.sr, self.f1, self.f2 = sr, f1, f2
        self.aware_band = (a1, a2)
        self.f_lo = max(1.0, f1 - guard_hz)
        self.f_hi = min(sr / 2.0 - 1.0, f2 + guard_hz)

        # the shifted band must stay strictly inside (0, sr/2) or the analytic
        # signal wraps and the inverse shift stops being an inverse
        lo_s, hi_s = self.f_lo + self.shift, self.f_hi + self.shift
        if not 0.0 < lo_s < hi_s < sr / 2.0:
            raise ValueError(f"shifted band {(lo_s, hi_s)} leaves (0, {sr / 2})")

    def perceived(self, f):
        """Where AWARE thinks a component at `f` Hz sits."""
        return (np.asarray(f, dtype=float) + self.shift) / self.r

    def host_length(self, n):
        return -(-n * self.up // self.down)          # ceil, matches resample_poly

    def payload_bits(self, n, bps=16.0):
        """Bits AWARE writes, since it charges per NOMINAL second."""
        return bps * self.host_length(n) / self.sr

    def __repr__(self):
        return (f"BandPlan({self.f1:g}-{self.f2:g} Hz -> "
                f"{self.aware_band[0]:g}-{self.aware_band[1]:g} Hz, "
                f"shift={self.shift:+.2f} Hz, r={self.up}/{self.down})")


# --------------------------------------------------------------------------- #
#  pieces
# --------------------------------------------------------------------------- #
def _fit(sig, n):
    """Trim or zero-pad to exactly n samples (resample_poly rounds up)."""
    if len(sig) >= n:
        return sig[:n]
    return np.concatenate([sig, np.zeros(n - len(sig), dtype=sig.dtype)])


def _shift(sig, hz, sr):
    """Single-sideband frequency shift. Amplitude-preserving and, for a signal
    whose band stays inside (0, sr/2) throughout, invertible by negating `hz`."""
    n = np.arange(len(sig))
    return np.real(hilbert(sig) * np.exp(2j * np.pi * hz * n / sr))


def split(x, plan, numtaps=NUMTAPS):
    """(lo, hi). `lo` is untouched by everything downstream; lo + hi == x to
    machine precision regardless of how good the filter is, since lo is defined
    by subtraction rather than by a complementary filter."""
    taps = firwin(numtaps, [plan.f_lo, plan.f_hi], pass_zero=False, fs=plan.sr)
    hi = filtfilt(taps, [1.0], x)
    return x - hi, hi


def to_model(hi, plan):
    """Band-limited signal -> what AWARE reads (longer, band at its window)."""
    return resample_poly(_shift(hi, plan.shift, plan.sr), plan.up, plan.down)


def from_model(host, plan, n_out):
    """Inverse of `to_model`, resolved to exactly n_out samples."""
    sig = _fit(resample_poly(host, plan.down, plan.up), n_out)
    return _shift(sig, -plan.shift, plan.sr)


def analyze(x, plan, numtaps=NUMTAPS):
    lo, hi = split(x, plan, numtaps)
    return lo, to_model(hi, plan)


def synthesize(lo, host, plan):
    return lo + from_model(host, plan, len(lo))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def snr_db(ref, test, trim=0):
        if trim:
            ref, test = ref[trim:-trim], test[trim:-trim]
        err = np.sum((ref - test) ** 2)
        return 10.0 * np.log10(np.sum(ref ** 2) / max(err, 1e-300))

    def peak_hz(sig, sr):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return np.fft.rfftfreq(len(sig), 1.0 / sr)[int(np.argmax(spec))]

    sr, dur = SR, 4.0
    n = int(sr * dur)
    t = np.arange(n) / sr
    plan = BandPlan(500, 4000)
    print(plan)
    print(f"  {n} samples -> host {plan.host_length(n)}  "
          f"({plan.host_length(n) / sr:.3f} nominal s, "
          f"{plan.payload_bits(n):.0f} bits vs 64 stock)\n")

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) * 0.05).astype(np.float64)
    for f in (700.0, 1800.0, 3200.0):
        x += 0.3 * np.sin(2 * np.pi * f * t)

    # T1 -- the split loses nothing
    lo, hi = split(x, plan)
    print(f"T1 split exactness        {snr_db(x, lo + hi):8.1f} dB   (want > 200)")

    # T2 -- the map inverts on the band it was built for
    hi2 = from_model(to_model(hi, plan), plan, n)
    print(f"T2 map round trip         {snr_db(hi, hi2, trim=2048):8.1f} dB   (want >  40)")

    # T3 -- full pipeline with nothing embedded
    y = synthesize(*analyze(x, plan), plan)
    print(f"T3 pipeline null          {snr_db(x, y, trim=2048):8.1f} dB   (want >  40)")

    # T4 -- tones land where the plan says, including both band edges
    print("\nT4 placement")
    for f in (500.0, 700.0, 1800.0, 3200.0, 4000.0):
        tone = np.sin(2 * np.pi * f * t)
        _, host = analyze(tone, plan)
        got, want = peak_hz(host[2048:-2048], sr), float(plan.perceived(f))
        flag = "ok" if abs(got - want) < 15.0 else "MISMATCH"
        print(f"   {f:6.0f} Hz -> {got:7.1f} Hz   (plan says {want:7.1f})  {flag}")

    print("\n   AWARE's window 1000-4000 Hz now reads the 500-4000 Hz band.")
