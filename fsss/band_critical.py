"""
fsss/band_critical.py -- critically-sampled subband extraction for AWARE.

The whiteboard plan: split 0-sr/2 into M uniform strips, pick one, RESAMPLE that
strip down to its own Nyquist rate, embed there, then put it back and recombine
with the strips we did not touch. This module is the resampling half;
fsss/chain_embed_critical.py wires it into AWARE's optimization loop.

Relationship to fsss.band_steer  (READ THIS FIRST -- it is the whole design)
---------------------------------------------------------------------------
band_steer already solves the same problem -- point a frozen AWARE at a band it
was not built for -- using an AFFINE map (frequency shift, then rational
resample) that pins an arbitrary target band onto AWARE's window. It works for
ANY band, which this module does not. But the host it produces is OVERCOMPLETE.
Its own docstring says so:

    "host has more samples than degrees of freedom (74667 vs 64000 for a 4 s
     clip at r = 7/6), so most of its spectrum is interpolation, not signal, and
     anything written there dies on the way back."

This module takes the opposite position. Strip k of a uniform M-band split is
exactly W = sr/(2M) Hz wide, so keeping every M-th sample is alias-free and
lands the strip on [0, W] CRITICALLY sampled -- as many samples as degrees of
freedom, not one more. Every optimizer variable then corresponds to a direction
that can actually survive the trip home.

    band_steer      arbitrary band, OVERCOMPLETE host   -> exp2
    band_critical   aligned strip,  CRITICAL host       -> exp1

fsss/exp_v16_critical_decimation.py runs the two against each other. The question
that pair answers: does removing the wasted directions help, or was the optimizer
already ignoring them?

Why keeping every M-th sample is not lossy
------------------------------------------
Sampling folds the frequency axis at every multiple of fs/2. A band sitting
INSIDE one fold survives intact; a band STRADDLING a fold is destroyed, because
its two halves land on top of each other and cannot be separated again.

Strip k spans [k*W, (k+1)*W]. Decimating by exactly M puts the folds at
0, W, 2W, 3W, ... -- so the strip fills one fold edge to edge and nothing
overlaps. That alignment is the entire reason the decimation factor must equal
the number of strips.

It also explains why "resample less aggressively to be safe" is wrong. At M = 8
on the same strip the folds land at multiples of 1000 Hz, the crease at 2000 Hz
cuts strip 2 in half, and the strip aliases onto itself. T5 in the smoke test
below demonstrates that failure on purpose, so the alignment argument is
measured rather than asserted.

ODD strips arrive frequency-REVERSED, because the fold they land in runs
backwards. Harmless as long as analysis and synthesis agree (they do), but it is
why CriticalPlan.perceived() has two branches, and why even strips are easier to
reason about. Strip 2 (1600-2400 Hz at sr=16000, M=10) is even.

The guard band goes INWARD -- opposite to band_steer
----------------------------------------------------
band_steer WIDENS its analysis filter past the nominal band (f_lo = f1 - guard)
because the affine map leaves room on either side. Here there is no room: any
energy outside [k*W, (k+1)*W] folds straight onto the strip and cannot be
removed afterwards. So the guard NARROWS the passband instead, and the filter's
roll-off is spent inside the slot where it cannot alias.

Cost: usable width is W - 2*guard rather than W. At sr=16000, M=10, guard=100
that is 600 Hz of the 800 Hz slot.

`guard_hz` must exceed the FIR transition width (~4*sr/numtaps). At numtaps=1023
that is ~63 Hz, so the default 100 Hz has margin. Raise numtaps before lowering
guard; the constructor enforces this rather than letting it fail silently as
aliasing.

Standalone smoke test (no AWARE / no GPU needed):
    python -m fsss.band_critical
"""

import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.signal import firwin

from fsss.band_steer_torch import _fit

__all__ = ["CriticalPlan", "taps_for", "t_to_model", "t_from_model",
           "t_analyze", "t_synthesize", "t_split", "t_bandpass"]


# --------------------------------------------------------------------------- #
#  filtering -- same math as band_steer_torch.t_bandpass, made loop-affordable
# --------------------------------------------------------------------------- #
#  band_steer_torch picks L = n + 2*numtaps and transforms the taps on every
#  call. For a handful of analyses that is fine. Here _through_chain calls the
#  bandpass TWICE PER ITERATION at full rate, 400 iterations per file, so both
#  choices become the dominant cost:
#
#    * L = 160000 + 2046 = 162046 has large prime factors, so torch falls back
#      to Bluestein's algorithm instead of a fast radix transform.
#    * rfft(taps, L) is recomputed every call although the taps never change.
#
#  Rounding L up to an FFT-friendly length is EXACT, not an approximation. The
#  autocorrelation of h has support [-(m-1), m-1], so x*|h|^2 has support
#  [-(m-1), n+m-2]; any L >= n+m-1 keeps the wrapped negative-lag tail clear of
#  [0, n-1], and y[..., :n] is then identical for every such L. Both the old and
#  new lengths satisfy that, so this changes speed and nothing else.
#
#  band_steer_torch is left untouched on purpose: exp2a imports from there, and
#  editing it would silently move results the earlier fsss experiments already
#  published.
_H2_CACHE = {}


def _h2(taps, L):
    """|H(f)|^2 for these taps at transform length L, cached."""
    key = (id(taps), L, str(taps.device), str(taps.dtype))
    if key not in _H2_CACHE:
        H = torch.fft.rfft(taps, L)
        _H2_CACHE[key] = H.real ** 2 + H.imag ** 2
    return _H2_CACHE[key]


def t_bandpass(x, taps):
    """Zero-phase bandpass -- the torch analogue of filtfilt.

    Multiplying by |H|^2 applies h followed by time-reversed h. The DFT of that
    is the autocorrelation of h, symmetric about lag 0, so the convolution
    carries NO net delay provided the transform is long enough that the
    two-sided response cannot wrap around.
    """
    n = x.shape[-1]
    L = int(next_fast_len(n + 2 * taps.shape[-1]))
    y = torch.fft.irfft(torch.fft.rfft(x, L) * _h2(taps, L), L)
    return y[..., :n]


def t_split(x, taps):
    """(lo, hi). `lo` is defined by subtraction, so lo + hi == x exactly however
    the filter behaves -- reconstruction can never be blamed on the taps."""
    hi = t_bandpass(x, taps)
    return x - hi, hi

SR = 16000
NUM_BANDS = 10
NUMTAPS = 1023
GUARD_HZ = 100.0


# --------------------------------------------------------------------------- #
#  the plan
# --------------------------------------------------------------------------- #
class CriticalPlan:
    """Uniform M-band split with one strip selected and critically decimated.

    Mirrors the role of band_steer.BandPlan: pure arithmetic, no signal, so it
    can be constructed and inspected without touching audio.

    Attributes worth reading: `slot` (the strip's nominal edges, which decide
    the fold alignment), `f_lo`/`f_hi` (what the filter actually passes, guard
    applied inward), `M` (both the number of strips AND the decimation factor --
    they are the same number by construction), `reversed` (odd strips land
    mirrored).
    """

    def __init__(self, band_index, num_bands=NUM_BANDS, sr=SR,
                 guard_hz=GUARD_HZ, numtaps=NUMTAPS):
        if not 0 <= band_index < num_bands:
            raise ValueError(f"band_index {band_index} outside [0, {num_bands})")

        W = sr / (2.0 * num_bands)
        if guard_hz >= W / 2.0:
            raise ValueError(
                f"guard_hz={guard_hz} leaves no passband in a {W:g} Hz slot")

        # The filter cannot be sharper than its own transition width. If the
        # guard is narrower than that, the roll-off spills outside the slot and
        # aliases on decimation -- a silent corruption, so refuse up front.
        min_guard = 4.0 * sr / numtaps
        if guard_hz < min_guard:
            raise ValueError(
                f"guard_hz={guard_hz} below the FIR transition width "
                f"{min_guard:.1f} Hz at numtaps={numtaps}; raise numtaps or guard")

        self.k = int(band_index)
        self.M = int(num_bands)          # strips AND decimation factor: same number
        self.sr = float(sr)
        self.W = float(W)
        self.numtaps = int(numtaps)
        self.guard_hz = float(guard_hz)

        self.slot = (self.k * W, (self.k + 1) * W)      # fold-aligned edges
        self.f_lo = self.k * W + guard_hz               # guard applied INWARD
        self.f_hi = (self.k + 1) * W - guard_hz
        self.reversed = bool(self.k % 2)                # odd strips land mirrored
        self.sr_dec = sr / num_bands                    # true rate of the host

    def perceived(self, f):
        """Where AWARE thinks a component at `f` Hz sits.

        Two steps folded into one formula. First the decimation aliases the
        strip down into [0, W]; even strips keep their orientation, odd strips
        arrive reversed. Then the host is handed to AWARE with no sample rate
        attached, so AWARE reads it as if it were sr -- which multiplies every
        apparent frequency by M.

        That second step is the "relabel", and note that it costs nothing: it is
        not an operation we perform, it is what happens because torch.stft never
        asks for a sample rate and the detector's mel bank was built once at sr.
        """
        f = np.asarray(f, dtype=float)
        alias = ((self.k + 1) * self.W - f) if self.reversed else (f - self.k * self.W)
        return alias * self.M

    def host_length(self, n):
        """Samples AWARE sees. ceil(n/M), matching x[..., ::M]."""
        return -(-int(n) // self.M)

    def slot_dof(self, n):
        """Degrees of freedom in the SLOT: 2 * W * duration.

        This is the number host_length(n) should equal exactly -- that equality
        IS the criticality claim, and is what band_steer's host does not satisfy
        (74667 samples for 64000 dof on its own example).
        """
        return 2.0 * self.W * (n / self.sr)

    def dof(self, n):
        """Degrees of freedom actually CARRYING CONTENT: 2 * passband * duration.

        Smaller than slot_dof by exactly the guard band, which is spent to stop
        aliasing rather than lost to overcompleteness. Do not conflate the two:
        the host is critically sampled for the slot, and separately some of the
        slot is deliberately left empty. At W=800, guard=100 that is 600/800, so
        25% of the host's directions are guard.
        """
        return 2.0 * (self.f_hi - self.f_lo) * (n / self.sr)

    def __repr__(self):
        return (f"CriticalPlan(strip {self.k}/{self.M}: "
                f"{self.slot[0]:g}-{self.slot[1]:g} Hz, "
                f"pass {self.f_lo:g}-{self.f_hi:g}, "
                f"decimate x{self.M} -> {self.sr_dec:g} Hz, "
                f"{'REVERSED' if self.reversed else 'upright'})")


# --------------------------------------------------------------------------- #
#  pieces
# --------------------------------------------------------------------------- #
def taps_for(plan, device=None, dtype=torch.float64):
    """FIR bandpass taps for the plan's guarded passband. Cache these -- firwin
    is numpy and does not belong inside an optimization loop."""
    t = firwin(plan.numtaps, [plan.f_lo, plan.f_hi], pass_zero=False, fs=plan.sr)
    return torch.as_tensor(t, device=device, dtype=dtype)


def t_to_model(hi, plan):
    """Strip at full rate -> what AWARE reads. Pure selection, nothing computed.

    NOTE the thing that is deliberately missing: there is no anti-alias filter
    here. Every resampling tutorial says to lowpass before decimating, and doing
    so would be catastrophic -- the strip lives ABOVE the decimated Nyquist, so
    an anti-alias filter would delete precisely the content we are trying to
    keep. The naive slice is correct here only because the strip is already
    confined to one fold; that integer-band alignment is doing the work an
    anti-alias filter would normally do.
    """
    return hi[..., ::plan.M]


def t_from_model(host, plan, n_out, taps):
    """Inverse of t_to_model, resolved to exactly n_out samples.

    Zero-stuffing replicates the spectrum every sr/M Hz. The bandpass then keeps
    the single copy that lands back in the original slot and discards the other
    M-1. The xM gain undoes the energy the inserted zeros spread out.

    Signature note: this takes `taps`, where band_steer_torch.t_from_model does
    not. The affine map inverts with a resample; this one needs the filter to
    pick the right spectral copy, so the taps have to come along.
    """
    n_up = host.shape[-1] * plan.M
    up = torch.zeros(*host.shape[:-1], n_up, dtype=host.dtype, device=host.device)
    up[..., ::plan.M] = host                  # differentiable: index_put, not a copy
    return _fit(t_bandpass(up, taps) * plan.M, n_out)


def t_analyze(x, plan, taps):
    """x -> (lo, host). `lo` is everything we do not touch."""
    lo, hi = t_split(x, taps)                 # lo = x - hi, exact by subtraction
    return lo, t_to_model(hi, plan)


def t_synthesize(lo, host, plan, taps):
    """(lo, host) -> y. With host unmodified this returns x to float precision."""
    return lo + t_from_model(host, plan, lo.shape[-1], taps)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def snr_db(a, b, trim=0):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        if trim and m > 2 * trim:
            a, b = a[trim:-trim], b[trim:-trim]
        return 10.0 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-300))

    def peak_hz(sig, sr):
        sig = np.asarray(sig, dtype=float)
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return np.fft.rfftfreq(len(sig), 1.0 / sr)[int(np.argmax(spec))]

    sr, n = SR, 160000                        # 10 s, the length exp_v16 uses
    t = np.arange(n) / sr
    plan = CriticalPlan(2)                    # strip 2 = 1600-2400 Hz, even/upright
    taps = taps_for(plan)
    print(plan)
    print(f"  {n} samples -> host {plan.host_length(n)}   "
          f"slot dof {plan.slot_dof(n):.0f}   "
          f"{'EQUAL => critically sampled' if plan.host_length(n) == int(plan.slot_dof(n)) else 'MISMATCH'}")
    print(f"  passband dof {plan.dof(n):.0f}  -- the remaining "
          f"{100.0 * (1.0 - plan.dof(n) / plan.slot_dof(n)):.0f}% is guard band, "
          f"deliberately empty, NOT overcompleteness\n")

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(n) * 0.05
    for f in (700.0, 1800.0, 2100.0, 3200.0):   # two tones inside the strip, two outside
        x_np += 0.3 * np.sin(2 * np.pi * f * t)
    x = torch.as_tensor(x_np, dtype=torch.float64)

    # T1 -- the split loses nothing. lo is defined by subtraction, so this is a
    # property of arithmetic, not of the filter, and should be near machine eps.
    lo, hi = t_split(x, taps)
    print(f"T1 split exactness         {snr_db(x_np, (lo + hi).numpy()):8.1f} dB   (want > 200)")

    # T2 -- decimate then interpolate. THE claim of this module: the 9 of every
    # 10 samples we drop were redundant, so putting them back is exact.
    hi2 = t_from_model(t_to_model(hi, plan), plan, n, taps)
    print(f"T2 decimation round trip   {snr_db(hi.numpy(), hi2.numpy(), 4096):8.1f} dB   (want >  40)")

    # T3 -- full pipeline with nothing embedded. This is the identity the
    # optimizer implicitly assumes at iteration 0.
    y = t_synthesize(*t_analyze(x, plan, taps), plan, taps)
    print(f"T3 pipeline null           {snr_db(x_np, y.numpy(), 4096):8.1f} dB   (want >  40)")

    # T4 -- tones land where the plan says. Confirms the fold arithmetic AND the
    # relabel: we measure the host with sr=SR even though it is really sr/M.
    print("\nT4 placement (host measured at sr, i.e. as AWARE reads it)")
    _, host_probe = t_analyze(x, plan, taps)
    for f in (1800.0, 2000.0, 2200.0):
        tone = torch.as_tensor(np.sin(2 * np.pi * f * t), dtype=torch.float64)
        _, h = t_analyze(tone, plan, taps)
        got, want = peak_hz(h.numpy()[512:-512], sr), float(plan.perceived(f))
        flag = "ok" if abs(got - want) < 60.0 else "MISMATCH"
        print(f"   {f:6.0f} Hz -> {got:7.1f} Hz   (plan says {want:7.1f})  {flag}")

    # T5 -- the alignment argument, measured. Decimating by 8 instead of 10 puts
    # a fold at 2000 Hz, straight through strip 2, so the strip aliases onto
    # itself. Expect this to be dramatically worse than T2; if it is NOT, the
    # test signal has no energy in the strip and T2 proved nothing either.
    M_bad = 8
    host_bad = hi[..., ::M_bad]
    up_bad = torch.zeros(host_bad.shape[-1] * M_bad, dtype=hi.dtype)
    up_bad[::M_bad] = host_bad
    hi_bad = _fit(t_bandpass(up_bad, taps) * M_bad, n)
    good = snr_db(hi.numpy(), hi2.numpy(), 4096)
    bad = snr_db(hi.numpy(), hi_bad.numpy(), 4096)
    print(f"\nT5 wrong factor (x{M_bad})     {bad:8.1f} dB   "
          f"(want << T2's {good:.1f} -- a fold cuts the strip in half)")

    # T6 -- the whole point: gradients must survive the round trip, or the
    # chain cannot go inside AWARE's optimizer.
    h = t_to_model(hi, plan).detach().requires_grad_(True)
    t_synthesize(lo, h, plan, taps).pow(2).sum().backward()
    g = h.grad
    ok = bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0
    print(f"\nT6 gradient through chain  max|grad|={float(g.abs().max()):.3e}  "
          f"finite={bool(torch.isfinite(g).all())}  {'ok' if ok else 'BROKEN'}")
