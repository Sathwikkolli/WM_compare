"""
casp/casp_front.py -- thin wrapper around the vendored revised CASP model
(Paulick, Relano-Iborra & Dau 2025; torch_amt port of AMT `paulick2024`).

What it adds on top of `torch_amt.models.paulick2024.Paulick2024`:
  * level handling: audio in digital units -> Pascals at a stated dB SPL
  * a reduced channel set covering the watermark band (the model is a
    sample-by-sample recursion, so cost is ~ samples; the 50-channel default is
    far more than 1-4 kHz needs)
  * the adaptation stage's hard-coded 50 quiet-threshold values are
    interpolated (log-log) onto the reduced channel grid
  * a single "internal representation" tensor, so differences are one norm

Everything here is read-only with respect to the vendored package.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor", "src")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from torch_amt.models.paulick2024 import Paulick2024  # noqa: E402

P_REF = 20e-6                      # Pa, 0 dB SPL
REF_IR_FS = 4000.0                 # IR rate at fs=16 kHz; distances are scaled to it


def spl_scale(x, level_db, ref_rms=None):
    """Scale `x` so that its RMS (or `ref_rms`, if given -- use the HOST's rms
    for the marked signal so both share one calibration) is `level_db` dB SPL.
    Returns (pressure_signal_in_Pa, gain)."""
    r = float(np.sqrt(np.mean(np.square(x)))) if ref_rms is None else float(ref_rms)
    gain = P_REF * 10.0 ** (level_db / 20.0) / max(r, 1e-12)
    return (x * gain).astype(np.float32), gain


class CASP:
    """Revised CASP front end on a reduced, band-limited channel grid."""

    def __init__(self, fs=16000, flow=700.0, fhigh=5500.0, n_channels=24,
                 threads=None):
        if threads:
            torch.set_num_threads(int(threads))
        self.fs = fs
        ref = Paulick2024(fs=44100, n_channels=50).eval()      # default grid
        fc50 = ref.fc.numpy()
        ml50 = ref.adaptation.minlvl_per_channel.detach().numpy()
        probe = Paulick2024(fs=fs, flow=flow, fhigh=fhigh, n_channels=n_channels)
        fc = probe.fc.numpy()
        ml = np.exp(np.interp(np.log(fc), np.log(fc50), np.log(ml50)))
        self.model = Paulick2024(
            fs=fs, flow=flow, fhigh=fhigh, n_channels=n_channels,
            adaptation_kwargs={"minlvl_per_channel": torch.tensor(ml, dtype=torch.float32)},
        ).eval()
        self.fc = self.model.fc.numpy().copy()
        self.fs_ir = self.model.fs_resampled       # sample rate of the IR

    @torch.no_grad()
    def ir(self, x_pa):
        """x_pa: (T,) or (B,T) pressure in Pa -> (B, K, T_ir) tensor.
        K = sum over channels of modulation filters (concatenated)."""
        x = torch.as_tensor(np.asarray(x_pa, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        outs = self.model(x)                       # list of (B, M_i, T_ir)
        return torch.cat(outs, dim=1)

    def dist(self, ir_a, ir_b, t0=None, t1=None):
        """L2 norm of the IR difference over an optional [t0,t1) IR-sample window,
        per batch item. For a template equal to the difference signal and
        constant internal noise sigma, d' = this / sigma."""
        d = ir_a - ir_b
        if t0 is not None:
            d = d[..., t0:t1]
        # the sum runs over IR samples, so the raw norm grows with the IR sample
        # rate (4 kHz at fs=16 kHz, 11.025 kHz at 44.1 kHz). Scale to a 4 kHz
        # reference so runs at different fs are comparable.
        return torch.sqrt((d ** 2).sum(dim=(-2, -1)) * (REF_IR_FS / self.fs_ir))
