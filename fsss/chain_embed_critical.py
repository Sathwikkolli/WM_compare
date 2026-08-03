"""
fsss/chain_embed_critical.py -- AWARE embedding into ONE critically-sampled strip,
with the decimate/interpolate chain IN THE LOOP.

This is the exp1 embedder. It is the deliberate counterpart of
fsss/chain_embed.py (BandSteerAWAREEmbedder, exp2): same structure, same
insertion point, same verify() discipline -- the ONLY thing that differs is how
the strip is carried to AWARE and back.

    chain_embed.py           affine map (shift + resample), OVERCOMPLETE host
    chain_embed_critical.py  decimate by M,                 CRITICAL host

Keeping the two files line-for-line comparable is intentional: any difference in
results should be attributable to the sampling strategy and nothing else.

The insertion point (unchanged from chain_embed.py)
---------------------------------------------------
StaircaseAWAREEmbedder._optimize calls, every iteration:

    wm = self._recompute_watermarked_magnitude(wm, phase)
    wm[not_freq_indices] = 0.0
    predicted = self.detection_net(wm.unsqueeze(0))

`_recompute_watermarked_magnitude` is the last thing to touch the magnitude
before the detector, and _record_stats calls it too. Overriding that one method
is therefore the whole integration: the 400-iteration loop, the perceptual
clamp, the optimizer and the scheduler are all inherited untouched.

Why the chain MUST be in the loop
---------------------------------
At read time the signal reaches the detector only after
to_model -> +lo -> synthesize -> re-analyze, and that trip is not the identity.
In particular the synthesis bandpass DELETES anything the optimizer pushed
outside the strip. Optimize against `host` in isolation and the loss goes
happily to zero while the watermark is quietly erased on the way out. Put the
trip in the loop and the optimizer compensates for it automatically.

Why embedding_bands is forced wide open
---------------------------------------
After decimation the strip fills the host's whole spectrum, and the relabel
stretches it across 0..sr/2 as AWARE reads it. AWARE's own window (1000-4000)
would therefore mask off most of the strip -- exactly the detector-band bug that
fsss/exp_v9_band_compare.py was written to fix, one level up.

So both embedder and detector are set to (0, sr/2): mask nothing. This is safe
because the perceptual bounds already do the masking for us. AWARE sets
    delta = coeff * 10**(-tolerance_db/20)
    bounds = (max(0, coeff - delta), coeff + delta)
so a bin with no content (coeff ~ 0) gets bounds ~ (0, 0) and is pinned. Bins
that fall in the guard band freeze themselves; no explicit mask needed.

The caller must set embedding_bands on the DETECTOR too -- see
exp_v16_critical_decimation.py. Every earlier fsss experiment that got this
wrong reported artificially weak numbers.

Salient gating: anchors come from the ORIGINAL, never from the host
-------------------------------------------------------------------
chain_embed.py flagged this in advance:

    "the inherited mask builder will be handed `host`, not the original audio --
     anchors must be computed on the original and mapped through, or placement
     will drift."

That is why this class does not simply reuse SalientRegionAWAREEmbedder. The
inherited embed() passes whatever it was given to _build_staircase_mask, and by
that point we are inside the host. We stash the original in embed(), run
librosa_flux on THAT, and divide the anchor sample indices by M to land them in
host time. A 250 ms region in real time is 250/M ms of host time, so regions
shrink by the same factor the timeline does.

Frame budget -- read before choosing anchor_rate
------------------------------------------------
Decimating by M divides the frame count by M as well. A 10 s clip at hop 256
gives 626 frames stock but only 63 here. At the salient_region default
anchor_rate=3.5 that is 35 anchors over 63 frames, which saturates the mask to
fully-on and makes gating a silent no-op.

DEFAULT_ANCHOR_RATE below is therefore 1.2, not 3.5:
    1.2/s x 10 s = 12 anchors x 2 host frames ~= 24 of 63 frames ~= 38% coverage.
_build_staircase_mask logs the achieved coverage every run and WARNS above 85%,
so a saturated mask shows up in the log instead of masquerading as a null result.

Run verify() BEFORE any real embedding
--------------------------------------
Two conventions cannot be checked without a live AWAREEmbedder (whether its STFT
is centred, and how it scales magnitudes), and one is specific to this file (how
much the decimation round trip actually costs on real audio rather than on the
synthetic smoke-test signal). verify() measures all three and prints dB. A
convention mismatch produces a silently wrong watermark, not an exception.

    from fsss.chain_embed_critical import CriticalBandAWAREEmbedder
    e = CriticalBandAWAREEmbedder.from_embedder(base, band_index=2)
    e.verify(audio, 16000)          # <- read the numbers before trusting anything
    y = e.embed(audio, 16000, watermark)
"""

import numpy as np
import torch

from fsss.band_critical import (CriticalPlan, t_analyze, t_from_model,
                                t_split, t_synthesize, t_to_model, taps_for)
from fsss.band_steer_torch import _fit
from fsss.salient_mapped import (COVERAGE_WARN, DEFAULT_ANCHOR_RATE,
                                 DEFAULT_REGION_MS, build_mapped_region_mask)
from fsss.salient_region import normalize_rows
from fsss.staircase import StaircaseAWAREEmbedder

__all__ = ["CriticalBandAWAREEmbedder"]               # above this, gating is effectively a no-op


def _snr_db(ref, test, trim=0):
    ref = np.asarray(ref, dtype=float).ravel()
    test = np.asarray(test, dtype=float).ravel()
    m = min(len(ref), len(test))
    ref, test = ref[:m], test[:m]
    if trim and m > 2 * trim:
        ref, test = ref[trim:-trim], test[trim:-trim]
    return 10.0 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - test) ** 2), 1e-300))


class CriticalBandAWAREEmbedder(StaircaseAWAREEmbedder):
    """AWARE writing into one critically-sampled strip, optimized against the
    round trip. `salient=False` is exp1a (all cells), `salient=True` is exp1b."""

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_embedder(cls, base, band_index=2, num_bands=10, sampling_rate=16000,
                      guard_hz=None, numtaps=None, chain_in_loop=True,
                      stft_center=True, tolerance_db=None, salient=False,
                      anchor="librosa_flux", anchor_rate=DEFAULT_ANCHOR_RATE,
                      region_ms=DEFAULT_REGION_MS):
        """Wrap an already-loaded AWAREEmbedder, copying all its state so the
        config/cards are never rebuilt.

        n_bands=1 makes the inherited staircase mask the full stripe on every
        frame, i.e. no frequency hopping -- hopping is orthogonal to what exp1
        is testing and would confound it. band_range spans the whole spectrum
        for the reason given in the module docstring.
        """
        from fsss.band_critical import GUARD_HZ, NUMTAPS
        nyq = sampling_rate / 2.0

        obj = super().from_embedder(base, key=b"unused", n_bands=1,
                                    band_range=(0, nyq),
                                    anchor=anchor, anchor_rate=anchor_rate,
                                    segment_mode="fixed",
                                    tolerance_db=tolerance_db)

        obj.plan = CriticalPlan(band_index, num_bands=num_bands, sr=sampling_rate,
                                guard_hz=GUARD_HZ if guard_hz is None else guard_hz,
                                numtaps=NUMTAPS if numtaps is None else numtaps)
        # Mask nothing: the strip occupies the whole relabelled spectrum, and the
        # perceptual bounds pin the empty bins for us (see module docstring).
        obj.embedding_bands = (0, nyq)
        obj.chain_in_loop = bool(chain_in_loop)
        obj.stft_center = bool(stft_center)
        obj.salient = bool(salient)
        obj.region_ms = float(region_ms)

        obj._taps_cache = {}
        obj._lo = None                 # the untouched remainder, x - strip
        obj._lo_dev = None             # _lo on the compute device (see _lo_on)
        obj._lo_dev_key = None
        obj._host_len = None
        obj._orig_audio = None         # anchors are computed on THIS, not the host
        obj._orig_sr = int(sampling_rate)
        obj.last_mask_stats = None
        obj.last_chain_stats = {}
        return obj

    # ---- helpers ----------------------------------------------------------- #
    def _taps(self, device, dtype):
        k = (str(device), str(dtype))
        if k not in self._taps_cache:
            self._taps_cache[k] = taps_for(self.plan, device=device, dtype=dtype)
        return self._taps_cache[k]

    def _set_lo(self, lo):
        """Store the untouched remainder and drop any cached device copy.

        Always go through this rather than assigning self._lo directly -- a
        stale _lo_dev would silently watermark the PREVIOUS clip's remainder
        into this clip, which produces plausible-looking audio and a watermark
        that does not detect.
        """
        self._lo = lo
        self._lo_dev = None
        self._lo_dev_key = None

    def _lo_on(self, device, dtype):
        """`lo` on the compute device, cached across iterations.

        The optimizer runs 400 iterations per file and `lo` is constant for all
        of them. Without this cache _through_chain paid a host-to-device copy
        AND a float64->float32 cast on every one, which is invisible on CPU
        (same memory, cast only) and dominant on GPU.
        """
        k = (str(device), str(dtype))
        if self._lo_dev is None or self._lo_dev_key != k:
            self._lo_dev = self._lo.to(device, dtype)
            self._lo_dev_key = k
        return self._lo_dev

    def _window(self, device, dtype):
        return torch.hann_window(self.frame_length, device=device, dtype=dtype)

    def _stft_mag(self, wav):
        S = torch.stft(wav, n_fft=self.frame_length, hop_length=self.hop_length,
                       window=self._window(wav.device, wav.dtype),
                       center=self.stft_center, return_complex=True)
        return S.abs()

    def _istft(self, magnitude, phase, length):
        S = torch.polar(magnitude, phase.to(magnitude.device, magnitude.dtype))
        return torch.istft(S, n_fft=self.frame_length, hop_length=self.hop_length,
                           window=self._window(magnitude.device, magnitude.dtype),
                           center=self.stft_center, length=length)

    # ---- salient gating, with anchors mapped from original time ------------ #
    def _build_staircase_mask(self, audio, sampling_rate, n_freq, n_frames):
        """Boolean [n_freq x n_frames] mask of writable cells.

        `audio` arrives as the HOST (inherited embed() passes what it was given),
        which is why we ignore it for anchor detection and use self._orig_audio.
        Using `audio` here is the drift bug chain_embed.py warned about.
        """
        if not self.salient:
            # n_bands=1 -> inherited mask is the full stripe on every frame.
            return super()._build_staircase_mask(audio, sampling_rate,
                                                 n_freq, n_frames)

        rows = self._writable_rows(sampling_rate, n_freq)

        if self._orig_audio is None:                     # no original stashed
            M = np.zeros((n_freq, n_frames), dtype=bool)
            M[rows, :] = True
            self.last_mask_stats = {"n_anchors": 0, "n_regions": 1,
                                    "coverage": 1.0, "cells": int(M.sum()),
                                    "cells_full_stripe": int(len(rows) * n_frames),
                                    "fell_back": True, "n_frames": int(n_frames)}
            return M

        # Shared with exp2b (fsss/chain_embed_salient.py) ON PURPOSE. exp1b vs
        # exp2b is meant to isolate the sampling strategy, so both arms must
        # build their masks with identical arithmetic -- only the time scale
        # differs. Decimation keeps every M-th sample and drops no time, so the
        # original->host map here is 1/M; band_steer passes up/down instead.
        M, stats = build_mapped_region_mask(
            self._orig_audio, self._orig_sr,
            1, self.plan.M,                              # <- the time scale, 1/10
            n_freq, n_frames, self.hop_length, rows,
            anchor=self.anchor, anchor_rate=self.anchor_rate,
            region_ms=self.region_ms)
        self.last_mask_stats = stats

        if self.verbose:
            from aware.utils.logger import logger
            if stats["fell_back"]:
                logger.warning("critical/salient: no anchors -- writing the "
                               "FULL stripe (equivalent to exp1a)")
            logger.info(f"critical/salient: {stats['n_anchors']} anchors -> "
                        f"{stats['n_regions']} regions, coverage "
                        f"{stats['coverage']*100:.1f}% of {n_frames} frames, "
                        f"{stats['cells']} writable cells "
                        f"(full stripe would be {stats['cells_full_stripe']})")
            if stats["coverage"] > COVERAGE_WARN:
                logger.warning(
                    f"critical/salient: coverage {stats['coverage']*100:.0f}% > "
                    f"{COVERAGE_WARN*100:.0f}% -- gating is effectively a NO-OP. "
                    f"Only {n_frames} host frames exist after decimating by "
                    f"{self.plan.M}; lower --anchor-rate or --region-ms.")
        return M

    def _writable_rows(self, sampling_rate, n_freq):
        """Rows the detector will actually read. Ask the base rather than assume,
        so the mask can never disagree with the detector band."""
        getter = getattr(self, "_get_embedding_frequency_indices", None)
        if getter is not None:
            try:
                rows = normalize_rows(getter(sampling_rate, self.frame_length)[0],
                                      n_freq)
                if rows.size:
                    return rows
            except Exception:                    # signature drift across AWARE versions
                pass
        return np.arange(n_freq, dtype=np.int64)

    # ---- the integration: one override ------------------------------------- #
    def _through_chain(self, magnitude, phase):
        """host magnitude -> what the detector will actually receive at read time.

        Differentiable end to end, so the 400-iteration loop above this sees the
        round trip and compensates for it. Mirrors
        BandSteerAWAREEmbedder._through_chain step for step.
        """
        n_host = self._host_len or magnitude.shape[-1] * self.hop_length
        taps = self._taps(magnitude.device, magnitude.dtype)

        wav = self._istft(magnitude, phase, n_host)          # host back to time
        lo = self._lo_on(wav.device, wav.dtype)              # cached, see _lo_on
        y = lo + t_from_model(wav, self.plan, lo.shape[-1], taps)   # rebuild full audio
        _, host = t_analyze(y, self.plan, taps)              # detector re-extracts
        return self._stft_mag(_fit(host, n_host))

    def _recompute_watermarked_magnitude(self, magnitude, phase):
        mag = super()._recompute_watermarked_magnitude(magnitude, phase)
        if not self.chain_in_loop or self._lo is None:
            return mag
        out = self._through_chain(mag, phase)
        return out if out.shape == mag.shape else _fit(out, mag.shape[-1])

    # ---- embed / detect ---------------------------------------------------- #
    def embed(self, audio: np.ndarray, sample_rate: int, watermark: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        taps = self._taps(x.device, x.dtype)

        lo, host = t_analyze(x, self.plan, taps)
        self._set_lo(lo)                       # never assign _lo directly
        self._host_len = int(host.shape[-1])
        self._orig_audio = np.asarray(audio, dtype=np.float32)   # anchors use this
        self._orig_sr = int(sample_rate)

        host_wm = super().embed(host.numpy().astype(audio.dtype), sample_rate, watermark)

        y = t_synthesize(lo, _fit(torch.as_tensor(np.asarray(host_wm, dtype=np.float64)),
                                  self._host_len), self.plan, taps)
        return y.numpy().astype(audio.dtype)

    def to_detector_input(self, audio, sample_rate=None):
        """Run the SAME analysis the embedder used, then feed the result to
        whatever AWARE detector entry point you normally call.

        For this experiment the detector is TOLD which strip to read -- there is
        no blind strip search. That is deliberate: it isolates the sampling
        question from the (harder, unsolved) question of recovering the strip
        index from attacked audio.
        """
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        _, host = t_analyze(x, self.plan, self._taps(x.device, x.dtype))
        return host.numpy().astype(np.asarray(audio).dtype)

    # ---- run this first ---------------------------------------------------- #
    def verify(self, audio, sample_rate, verbose=True):
        """Measure the four things that decide whether the wiring is right.

        stft_roundtrip_db : does our torch STFT/iSTFT agree with AWARE's own
                            magnitude convention? Low => `stft_center` is wrong,
                            or AWARE normalizes differently. Fix before anything.
        chain_null_db     : with no perturbation, does the chain return the
                            magnitude it was given? This is the identity the
                            optimizer implicitly assumes at iteration 0.
        decimation_snr_db : how exact the decimate/interpolate round trip is on
                            REAL audio. band_critical's T2 checks this on a
                            synthetic signal; speech has different spectral
                            occupancy, so measure it again here. This is the
                            number that would expose a too-narrow guard band.
        budget_gap_db     : how much quieter the host is than the original. The
                            inherited clamp is calibrated on the host, so this is
                            roughly how far off the watermark level is.

        Also reports host_len vs strip degrees of freedom. Their agreement is
        exp1's whole claim; band_steer's host fails it by construction.
        """
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        taps = self._taps(x.device, x.dtype)
        lo, host = t_analyze(x, self.plan, taps)
        self._set_lo(lo)                       # never assign _lo directly
        self._host_len = int(host.shape[-1])
        self._orig_audio = np.asarray(audio, dtype=np.float32)
        self._orig_sr = int(sample_rate)

        from aware.utils.utils import to_tensor
        h = to_tensor(host.numpy().astype(np.asarray(audio).dtype)).to(self.device)
        for processor in self.audio_preprocess_pipeline:
            h = processor(h)
        magnitude, phase = h

        wav = self._istft(magnitude, phase, self._host_len)
        stft_db = _snr_db(magnitude.detach().cpu().numpy(),
                          self._stft_mag(wav).detach().cpu().numpy())

        chain_db = _snr_db(magnitude.detach().cpu().numpy(),
                           self._through_chain(magnitude, phase).detach().cpu().numpy())

        # decimation round trip on the real strip, independent of AWARE
        _, hi = t_split(x, taps)
        hi2 = t_from_model(t_to_model(hi, self.plan), self.plan, hi.shape[-1], taps)
        dec_db = _snr_db(hi.numpy(), hi2.numpy(), trim=4096)

        rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2)) + 1e-20)
        budget_db = 20.0 * np.log10(rms(audio) / rms(host.numpy()))

        n = len(np.asarray(audio))
        self.last_chain_stats = {
            "plan": repr(self.plan),
            "stft_roundtrip_db": stft_db,
            "chain_null_db": chain_db,
            "decimation_snr_db": dec_db,
            "budget_gap_db": budget_db,
            "host_len": self._host_len,
            "slot_dof": self.plan.slot_dof(n),
            "strip_dof": self.plan.dof(n),
            "n_frames_host": 1 + self._host_len // self.hop_length,
        }
        if verbose:
            s = self.last_chain_stats
            print(f"plan                  {self.plan}")
            print(f"stft roundtrip        {stft_db:8.1f} dB   (want > 40; else stft_center wrong)")
            print(f"chain null            {chain_db:8.1f} dB   (want > 30)")
            # MEASURED 2026-08: at numtaps=4095 this reads ~33 dB on Emilia
            # speech and clean bit_acc is still 1.000 at 60/400 iterations, so
            # ~30 dB is demonstrably sufficient. The old ">40" was copied from
            # chain_embed's heuristic, not derived for this path -- do not chase
            # it. What limits this number is |H|^2 (analysis) vs |H|^4 (analysis
            # + synthesis) disagreeing inside the filter transitions; more taps
            # narrow them, ~6 dB per 4x. Only worry if clean bit_acc drops.
            print(f"decimation round trip {dec_db:8.1f} dB   (>30 ok; more taps = +6dB per 4x)")
            print(f"budget gap            {budget_db:8.1f} dB   (host is this much quieter)")
            print(f"host samples          {s['host_len']:8d}      "
                  f"(slot dof {s['slot_dof']:.0f} -- equal => critically sampled)")
            print(f"passband dof          {s['strip_dof']:8.0f}      "
                  f"(shortfall vs slot is guard band, not overcompleteness)")
            print(f"host frames           {s['n_frames_host']:8d}      "
                  f"(stock would be {1 + n // self.hop_length}; "
                  f"the BRH averages over these)")
        return self.last_chain_stats
