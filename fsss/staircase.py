"""
fsss/staircase.py -- key-driven subband-hopping embedder for AWARE.

StaircaseAWAREEmbedder subclasses AWARE's AWAREEmbedder and changes ONE thing:
instead of writing the watermark into a single static (500, 4000) Hz stripe on
every STFT frame, it writes into a KEY-CHOSEN sub-band that HOPS per anchor
segment. The anchor segments come from librosa spectral-flux onsets (our Exp C
winner); the per-segment band is chosen by HMAC(key, segment_index).

Everything else -- the adversarial optimization, the frozen detector, the loss,
the optimizer/scheduler, the +/- tolerance_db perceptual budget, the STFT
round-trip -- is inherited or copied verbatim from AWAREEmbedder. The only edits
are indexing: a static 1-D frequency-row mask (`freq_indices`) becomes a 2-D
boolean mask M[freq_bins x frames] that varies per frame.

Design note: the base zeroes everything OUTSIDE (500,4000) before the detector
(`watermarked_magnitude[not_freq_indices] = 0.0`), so the detector only sees that
band. We keep that untouched, so the staircase stays INSIDE (500,4000) -- our
n_bands sub-bands tile that range. With n_bands=1 the mask is the full stripe on
every frame => byte-identical to stock AWARE (the built-in equivalence check).

The stock `aware` install is never edited; all our code lives here.
"""

import time
import hmac
import hashlib
import numpy as np
import torch
import librosa

from aware.embedding.multibit_embedder import AWAREEmbedder
from aware.embedding.optimizers import get_optimizer
from aware.embedding.schedulers import get_scheduler
from aware.utils.utils import to_tensor
from aware.utils.logger import logger

try:
    from fsss.detectors import DETECTORS
except ImportError:
    from detectors import DETECTORS

__all__ = ["StaircaseAWAREEmbedder"]


class StaircaseAWAREEmbedder(AWAREEmbedder):
    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_embedder(cls, base, key, n_bands=4, band_range=(500, 4000),
                      min_seg_s=0.3, anchor_rate=3.5, anchor="librosa_flux",
                      segment_mode="librosa", segment_len_s=0.5, tolerance_db=None):
        """Wrap an already-loaded AWAREEmbedder, copying all its state (detector,
        pipelines, optimizer cfg, ...) so we never rebuild the config/cards.

        segment_mode: "librosa" (anchor-driven hops) or "fixed" (equal
        segment_len_s chunks, no content analysis). tolerance_db overrides the
        base perceptual budget (louder watermark) when provided.
        """
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.key = key.encode() if isinstance(key, str) else key
        obj.n_bands = int(n_bands)
        obj.band_range = tuple(band_range)
        obj.min_seg_s = float(min_seg_s)
        obj.anchor_rate = float(anchor_rate)
        obj.anchor = anchor
        obj.segment_mode = segment_mode
        obj.segment_len_s = float(segment_len_s)
        if tolerance_db is not None:
            obj.tolerance_db = float(tolerance_db)
        return obj

    # ---- the thesis contribution: the staircase mask ----------------------- #
    def _hmac_band(self, seg_index):
        digest = hmac.new(self.key, str(seg_index).encode(), hashlib.sha256).digest()
        return int.from_bytes(digest[:4], "big") % self.n_bands

    def _band_rows(self, sampling_rate, n_freq):
        """Row (freq-bin) indices for each of the n_bands equal-Hz sub-bands."""
        freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=self.frame_length)
        lo, hi = self.band_range
        edges = np.linspace(lo, hi, self.n_bands + 1)
        rows = []
        for b in range(self.n_bands):
            if b == self.n_bands - 1:                     # inclusive top => matches stock (<= hi)
                m = (freqs >= edges[b]) & (freqs <= edges[b + 1])
            else:
                m = (freqs >= edges[b]) & (freqs < edges[b + 1])
            rows.append(np.where(m)[0])
        return rows

    def _segments(self, audio, sampling_rate, n_frames):
        """Frame segments for the hop schedule. 'fixed' = equal segment_len_s
        chunks (no content analysis); 'librosa' = anchor-defined segments."""
        if getattr(self, "segment_mode", "librosa") == "fixed":
            L = max(1, int(self.segment_len_s * sampling_rate / self.hop_length))
            return [(s, min(s + L, n_frames)) for s in range(0, n_frames, L)]

        anchors = np.asarray(DETECTORS[self.anchor](audio, sampling_rate,
                                                     target_rate=self.anchor_rate))
        frames = sorted(set(int(a) // self.hop_length for a in anchors
                            if 0 < int(a) // self.hop_length < n_frames))
        min_frames = max(1, int(self.min_seg_s * sampling_rate / self.hop_length))
        starts = [0]
        for f in frames:                                  # keep a boundary only if far enough
            if f - starts[-1] >= min_frames:
                starts.append(f)
        segs = [(starts[i], starts[i + 1] if i + 1 < len(starts) else n_frames)
                for i in range(len(starts))]
        if len(segs) > 1 and (segs[-1][1] - segs[-1][0]) < min_frames:  # fold a short tail back
            segs[-2] = (segs[-2][0], segs[-1][1])
            segs.pop()
        return segs

    def _build_staircase_mask(self, audio, sampling_rate, n_freq, n_frames):
        rows = self._band_rows(sampling_rate, n_freq)
        segs = self._segments(audio, sampling_rate, n_frames)
        M = np.zeros((n_freq, n_frames), dtype=bool)
        for i, (s, e) in enumerate(segs):
            M[rows[self._hmac_band(i)], s:e] = True
        if self.verbose:
            logger.info(f"staircase: {len(segs)} segments, {self.n_bands} bands, "
                        f"{int(M.sum())} writable cells "
                        f"(stock would be ~{sum(len(r) for r in rows) // self.n_bands * n_frames})")
        return M

    # ---- embed: copy of AWAREEmbedder.embed with M instead of freq_indices - #
    def embed(self, audio: np.ndarray, sample_rate: int, watermark: np.ndarray) -> np.ndarray:
        x = to_tensor(audio).to(self.device)
        for processor in self.audio_preprocess_pipeline:
            x = processor(x)
        magnitude, phase = x
        watermark_pattern = to_tensor(watermark).to(self.device)

        # full-band indices (unchanged) are still used to ZERO outside (500,4000)
        freq_indices, not_freq_indices = self._get_embedding_frequency_indices(
            sample_rate, self.frame_length)

        # NEW: 2-D key-driven staircase mask over the writable cells
        M_np = self._build_staircase_mask(audio, sample_rate,
                                          magnitude.shape[0], magnitude.shape[1])
        M = torch.from_numpy(M_np).to(magnitude.device)

        watermark_coeffs = magnitude[M].flatten()
        magnitude_delta_threshold = watermark_coeffs * 10 ** (-self.tolerance_db / 20)
        bounds = [(max(0, c - d), c + d)
                  for c, d in zip(watermark_coeffs, magnitude_delta_threshold)]

        if self.verbose:
            logger.info(f"Starting optimization with {len(watermark_coeffs)} variables...")
        watermarked_coeffs = self._optimize(watermark_coeffs, magnitude, watermark_pattern,
                                            M, not_freq_indices, bounds, phase)

        watermarked_magnitude = magnitude.clone().detach().cpu()
        watermarked_magnitude[M.cpu()] = watermarked_coeffs
        phase = phase.detach().cpu() if phase.device != watermarked_magnitude.device else phase.detach()
        y = (watermarked_magnitude, phase)
        for processor in self.audio_postprocess_pipeline:
            y = processor(*y) if isinstance(y, tuple) and len(y) == 2 else processor(y)
        return y.detach().cpu().numpy()

    # ---- _optimize: copy with boolean-mask scatter instead of row-index ---- #
    def _optimize(self, initial_coeffs, stft_magnitude, watermark_pattern,
                  mask, not_freq_indices, bounds, stft_phase):
        start_time = time.time()
        for param in self.detection_net.parameters():
            param.requires_grad = False

        coeffs = to_tensor(initial_coeffs).to(self.device).requires_grad_(True)
        watermark_pattern = watermark_pattern.to(self.device)
        stft_magnitude = stft_magnitude.to(self.device)
        mask = mask.to(self.device)

        optimizer = get_optimizer(self.optimizer_name, [coeffs], **self.optimizer_params)
        scheduler = get_scheduler(self.scheduler_name, optimizer, **self.scheduler_params)
        lower_bounds = torch.FloatTensor([b[0] for b in bounds]).to(self.device)
        upper_bounds = torch.FloatTensor([b[1] for b in bounds]).to(self.device)

        best_loss = float("inf")
        best_coeffs = coeffs.clone()
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            watermarked_magnitude = stft_magnitude.clone()
            watermarked_magnitude[mask] = coeffs                     # <-- boolean scatter
            watermarked_magnitude = self._recompute_watermarked_magnitude(
                watermarked_magnitude, stft_phase)
            watermarked_magnitude[not_freq_indices] = 0.0            # zero outside (500,4000)
            predicted_pattern = self.detection_net(watermarked_magnitude.unsqueeze(0)).squeeze()
            loss = self.loss(predicted_pattern, watermark_pattern)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            with torch.no_grad():
                coeffs.data = torch.clamp(coeffs.data, lower_bounds, upper_bounds)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_coeffs = coeffs.clone().detach()
            if self.verbose and (iteration % 200 == 0 or iteration == self.num_iterations - 1):
                ber = torch.mean((torch.sign(predicted_pattern) != torch.sign(watermark_pattern)).float())
                logger.debug(f"Iter {iteration+1:3d}: loss={loss.item():.6f} BER={ber.item():.4f}")
        if self.verbose:
            logger.info(f"Optimization done in {time.time()-start_time:.1f}s, final loss {best_loss:.6f}")
        return best_coeffs.detach().cpu()
