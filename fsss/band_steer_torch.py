"""
fsss/band_steer_torch.py -- differentiable mirror of fsss.band_steer.

Same affine map, same BandPlan (reused verbatim -- it is pure arithmetic), but
built from torch ops so the whole analyse/synthesise round trip can sit INSIDE
AWARE's optimization loop and receive gradients.

Why this exists
---------------
AWARE optimizes a perturbation against a frozen detector. If it optimizes
against `host` alone, it is solving the wrong problem: the signal that reaches
the detector at read time has been through from_model -> +lo -> split ->
to_model first, and that trip is not identity. Put the trip in the loop and the
optimizer compensates for it automatically.

Operational vs reference
------------------------
THIS module is the operational path: use it on BOTH sides (embed and detect) so
there is no model mismatch. `fsss.band_steer` is the numpy reference used to
sanity-check placement and invertibility; it will NOT agree bit-for-bit, because
scipy's resample_poly and torchaudio's resample use different filters, and
filtfilt handles edges differently from the FFT-domain equivalent here. The
smoke test below reports the actual gap so it can be judged rather than assumed.

Standalone smoke test (no AWARE / no GPU needed):
    python -m fsss.band_steer_torch
"""

import numpy as np
import torch
import torchaudio
from scipy.signal import firwin

from fsss.band_steer import GUARD_HZ, NUMTAPS, BandPlan

__all__ = ["BandPlan", "taps_for", "t_bandpass", "t_hilbert", "t_shift",
           "t_split", "t_to_model", "t_from_model", "t_analyze", "t_synthesize"]


def taps_for(plan, numtaps=NUMTAPS, device=None, dtype=torch.float64):
    """FIR bandpass taps for the plan's widened band. Cache these -- firwin is
    numpy and does not belong inside an optimization loop."""
    t = firwin(numtaps, [plan.f_lo, plan.f_hi], pass_zero=False, fs=plan.sr)
    return torch.as_tensor(t, device=device, dtype=dtype)


def t_bandpass(x, taps):
    """Zero-phase bandpass, the torch analogue of filtfilt.

    Multiplying by |H|^2 applies h followed by time-reversed h. The DFT of that
    is the autocorrelation of h, which is symmetric about lag 0, so the circular
    convolution carries NO net delay -- provided the transform is long enough
    that the two-sided response cannot wrap around.
    """
    n = x.shape[-1]
    L = n + 2 * taps.shape[-1]
    H = torch.fft.rfft(taps, L)
    y = torch.fft.irfft(torch.fft.rfft(x, L) * (H.real ** 2 + H.imag ** 2), L)
    return y[..., :n]


def t_hilbert(x):
    """Analytic signal. Same construction as scipy.signal.hilbert."""
    n = x.shape[-1]
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * h, dim=-1)


def t_shift(x, hz, sr):
    """Single-sideband frequency shift; invertible by negating `hz` as long as
    the band stays inside (0, sr/2) at every point of the trip."""
    k = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
    ph = 2.0 * np.pi * float(hz) * k / float(sr)
    z = t_hilbert(x)
    return z.real * torch.cos(ph) - z.imag * torch.sin(ph)


def _resample(x, up, down):
    return torchaudio.functional.resample(x, orig_freq=int(down), new_freq=int(up))


def _fit(sig, n):
    if sig.shape[-1] >= n:
        return sig[..., :n]
    pad = torch.zeros(*sig.shape[:-1], n - sig.shape[-1],
                      dtype=sig.dtype, device=sig.device)
    return torch.cat([sig, pad], dim=-1)


def t_split(x, taps):
    """(lo, hi). `lo` is defined by subtraction, so lo + hi == x exactly however
    the filter behaves -- the reconstruction can never be blamed on the taps."""
    hi = t_bandpass(x, taps)
    return x - hi, hi


def t_to_model(hi, plan):
    return _resample(t_shift(hi, plan.shift, plan.sr), plan.up, plan.down)


def t_from_model(host, plan, n_out):
    return t_shift(_fit(_resample(host, plan.down, plan.up), n_out),
                   -plan.shift, plan.sr)


def t_analyze(x, plan, taps):
    lo, hi = t_split(x, taps)
    return lo, t_to_model(hi, plan)


def t_synthesize(lo, host, plan):
    return lo + t_from_model(host, plan, lo.shape[-1])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from fsss import band_steer as ref

    def snr_db(a, b, trim=0):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if trim:
            a, b = a[trim:-trim], b[trim:-trim]
        return 10.0 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-300))

    def peak_hz(sig, sr):
        sig = np.asarray(sig, dtype=float)
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return np.fft.rfftfreq(len(sig), 1.0 / sr)[int(np.argmax(spec))]

    sr, n = 16000, 64000
    t = np.arange(n) / sr
    plan = BandPlan(500, 4000)
    taps = taps_for(plan)
    print(plan, "\n")

    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal(n) * 0.05)
    for f in (700.0, 1800.0, 3200.0):
        x_np += 0.3 * np.sin(2 * np.pi * f * t)
    x = torch.as_tensor(x_np, dtype=torch.float64)

    lo, hi = t_split(x, taps)
    print(f"T1 split exactness        {snr_db(x_np, (lo + hi).numpy()):8.1f} dB   (want > 200)")

    hi2 = t_from_model(t_to_model(hi, plan), plan, n)
    print(f"T2 map round trip         {snr_db(hi.numpy(), hi2.numpy(), 2048):8.1f} dB   (want >  40)")

    y = t_synthesize(*t_analyze(x, plan, taps), plan)
    print(f"T3 pipeline null          {snr_db(x_np, y.numpy(), 2048):8.1f} dB   (want >  40)")

    # T4 -- does the torch path land in the same place as the numpy reference?
    _, host_t = t_analyze(x, plan, taps)
    _, host_n = ref.analyze(x_np, plan)
    m = min(host_t.shape[-1], len(host_n))
    print(f"T4 torch vs numpy host    {snr_db(host_n[:m], host_t.numpy()[:m], 4096):8.1f} dB   (informational)")

    print("\nT5 placement (torch path)")
    for f in (500.0, 1800.0, 4000.0):
        tone = torch.as_tensor(np.sin(2 * np.pi * f * t), dtype=torch.float64)
        _, h = t_analyze(tone, plan, taps)
        got, want = peak_hz(h.numpy()[2048:-2048], sr), float(plan.perceived(f))
        flag = "ok" if abs(got - want) < 15.0 else "MISMATCH"
        print(f"   {f:6.0f} Hz -> {got:7.1f} Hz   (plan says {want:7.1f})  {flag}")

    # T6 -- the whole point: gradients must survive the round trip
    h = t_to_model(hi, plan).requires_grad_(True)
    loss = t_synthesize(lo, h, plan).pow(2).sum()
    loss.backward()
    g = h.grad
    ok = torch.isfinite(g).all().item() and g.abs().max().item() > 0
    print(f"\nT6 gradient through chain  max|grad|={g.abs().max().item():.3e}  "
          f"finite={torch.isfinite(g).all().item()}  {'ok' if ok else 'BROKEN'}")
