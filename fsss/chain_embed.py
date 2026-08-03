"""
fsss/chain_embed.py -- AWARE embedding with the band-steer chain IN THE LOOP.

Step 2 of the band-steering plan. Step 1 (fsss.band_steer / band_steer_torch)
showed the affine map lands an arbitrary band on AWARE's window and inverts.
This wires it into AWARE so the optimizer works against the trip the signal
actually makes, not against `host` in isolation.

The insertion point
-------------------
StaircaseAWAREEmbedder._optimize calls, every iteration:

    wm = self._recompute_watermarked_magnitude(wm, phase)
    wm[not_freq_indices] = 0.0
    predicted = self.detection_net(wm.unsqueeze(0))

`_recompute_watermarked_magnitude` is the last thing to touch the magnitude
before the detector, and it is called from `_record_stats` too. Overriding it is
therefore the whole integration: one method, applied consistently in both
places, with the 500-iteration loop and the perceptual clamp untouched.

Two things this deliberately does NOT do
----------------------------------------
* No salient-region gating and no key hopping. `n_bands=1` makes the inherited
  mask the full stripe, i.e. stock AWARE. Those come back in step 3, once the
  chain is known to be sound. Note that when they do, the inherited mask builder
  will be handed `host`, not the original audio -- anchors must be computed on
  the original and mapped through, or placement will drift.
* No perceptual re-budgeting. The inherited clamp is +/- tolerance_db around the
  HOST's magnitudes, and the host is one band in isolation, so it is quieter
  than the full mix and the budget is not calibrated to the original masker.
  Fixing that is its own step; `budget_gap_db` below measures how wrong it is so
  the decision is informed rather than guessed.

Run `verify()` BEFORE any real embedding
----------------------------------------
Two conventions cannot be checked without a live AWAREEmbedder: whether its STFT
is centred, and whether its magnitudes are scaled the way torch.stft scales
them. `verify()` measures both and prints dB. Do that first -- a convention
mismatch produces a silently wrong watermark, not an exception.

    from fsss.chain_embed import BandSteerAWAREEmbedder
    e = BandSteerAWAREEmbedder.from_embedder(base, key=b"k", target_band=(500, 4000))
    e.verify(audio, 16000)          # <- read the numbers before trusting anything
    y = e.embed(audio, 16000, watermark)
"""

import numpy as np
import torch

from fsss.band_steer import BandPlan
from fsss.band_steer_torch import (_fit, t_analyze, t_from_model, t_split,
                                   t_synthesize, t_to_model, taps_for)
from fsss.salient_region import normalize_rows
from fsss.staircase import StaircaseAWAREEmbedder

__all__ = ["BandSteerAWAREEmbedder", "native_window"]


def native_window(base, sampling_rate):
    """AWARE's real embedding window in Hz, read from the frozen model rather
    than assumed.

    The paper says 1000-4000 Hz; fsss/staircase.py documents the installed base
    as zeroing outside (500, 4000). Those disagree, and the answer decides
    whether a given target band needs steering at all -- so ask the object.
    """
    import librosa
    idx, _ = base._get_embedding_frequency_indices(sampling_rate, base.frame_length)
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=base.frame_length)
    rows = normalize_rows(idx, len(freqs))
    if rows.size == 0:
        raise RuntimeError("base returned an empty embedding band")
    return float(freqs[rows.min()]), float(freqs[rows.max()])


def _snr_db(ref, test, trim=0):
    ref, test = np.asarray(ref, dtype=float).ravel(), np.asarray(test, dtype=float).ravel()
    m = min(len(ref), len(test))
    ref, test = ref[:m], test[:m]
    if trim and m > 2 * trim:
        ref, test = ref[trim:-trim], test[trim:-trim]
    return 10.0 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - test) ** 2), 1e-300))


class BandSteerAWAREEmbedder(StaircaseAWAREEmbedder):
    """AWARE writing into an arbitrary band, optimized against the round trip."""

    @classmethod
    def from_embedder(cls, base, key=b"unused", target_band=(500, 4000),
                      sampling_rate=16000, chain_in_loop=True,
                      aware_band=None, stft_center=True, tolerance_db=None):
        obj = super().from_embedder(base, key, n_bands=1,
                                    band_range=(500, 4000),
                                    segment_mode="fixed",
                                    tolerance_db=tolerance_db)
        win = tuple(aware_band) if aware_band is not None else native_window(base, sampling_rate)
        obj.plan = BandPlan(target_band[0], target_band[1],
                            aware_band=win, sr=sampling_rate)
        obj.native_band = win
        obj.chain_in_loop = bool(chain_in_loop)
        obj.stft_center = bool(stft_center)
        obj._taps_cache = {}
        obj._lo = None
        obj._host_len = None
        obj.last_chain_stats = {}
        return obj

    # ---- helpers ----------------------------------------------------------- #
    def _taps(self, device, dtype):
        k = (str(device), str(dtype))
        if k not in self._taps_cache:
            self._taps_cache[k] = taps_for(self.plan, device=device, dtype=dtype)
        return self._taps_cache[k]

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

    # ---- the integration: one override ------------------------------------- #
    def _through_chain(self, magnitude, phase):
        """host magnitude -> what the detector will actually receive at read time.

        Differentiable end to end, so the 500-iteration loop above this sees the
        round trip and compensates for it.
        """
        n_host = self._host_len or magnitude.shape[-1] * self.hop_length
        wav = self._istft(magnitude, phase, n_host)
        lo = self._lo.to(wav.device, wav.dtype)
        y = lo + t_from_model(wav, self.plan, lo.shape[-1])          # rebuild full audio
        _, host = t_analyze(y, self.plan, self._taps(wav.device, wav.dtype))
        return self._stft_mag(_fit(host, n_host))                    # detector re-extracts

    def _recompute_watermarked_magnitude(self, magnitude, phase):
        mag = super()._recompute_watermarked_magnitude(magnitude, phase)
        if not self.chain_in_loop or self._lo is None:
            return mag
        out = self._through_chain(mag, phase)
        return out if out.shape == mag.shape else _fit(out, mag.shape[-1])

    # ---- embed / detect ---------------------------------------------------- #
    def embed(self, audio: np.ndarray, sample_rate: int, watermark: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        lo, host = t_analyze(x, self.plan, self._taps(x.device, x.dtype))
        self._lo, self._host_len = lo, int(host.shape[-1])

        host_wm = super().embed(host.numpy().astype(audio.dtype), sample_rate, watermark)

        y = t_synthesize(lo, _fit(torch.as_tensor(np.asarray(host_wm, dtype=np.float64)),
                                  self._host_len), self.plan)
        return y.numpy().astype(audio.dtype)

    def to_detector_input(self, audio, sample_rate=None):
        """Run the SAME analysis the embedder used. Feed the result to whatever
        AWARE detector entry point you normally call."""
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        _, host = t_analyze(x, self.plan, self._taps(x.device, x.dtype))
        return host.numpy().astype(np.asarray(audio).dtype)

    # ---- run this first ---------------------------------------------------- #
    def verify(self, audio, sample_rate, verbose=True):
        """Measure the three things that decide whether the wiring is right.

        stft_roundtrip_db : does our torch STFT/iSTFT agree with AWARE's own
                            magnitude convention? Low => `stft_center` is wrong,
                            or AWARE normalizes differently. Fix before anything.
        chain_null_db     : with no perturbation, does the chain return the
                            magnitude it was given? This is the identity the
                            optimizer is implicitly assuming when it starts.
        budget_gap_db     : how much quieter the host is than the original. The
                            inherited perceptual clamp is calibrated on the host,
                            so this is roughly how far off the watermark level is.
        """
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        lo, host = t_analyze(x, self.plan, self._taps(x.device, x.dtype))
        self._lo, self._host_len = lo, int(host.shape[-1])

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

        rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2)) + 1e-20)
        budget_db = 20.0 * np.log10(rms(audio) / rms(host.numpy()))

        self.last_chain_stats = {
            "native_band": self.native_band,
            "plan": repr(self.plan),
            "stft_roundtrip_db": stft_db,
            "chain_null_db": chain_db,
            "budget_gap_db": budget_db,
            "host_len": self._host_len,
            "payload_bits": self.plan.payload_bits(len(audio)),
        }
        if verbose:
            print(f"native AWARE band     {self.native_band[0]:.0f}-{self.native_band[1]:.0f} Hz")
            print(f"plan                  {self.plan}")
            print(f"stft roundtrip        {stft_db:8.1f} dB   (want > 40; else stft_center wrong)")
            print(f"chain null            {chain_db:8.1f} dB   (want > 30)")
            print(f"budget gap            {budget_db:8.1f} dB   (host is this much quieter)")
            print(f"payload               {self.plan.payload_bits(len(audio)):8.0f} bits")
        return self.last_chain_stats
