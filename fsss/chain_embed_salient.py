"""
fsss/chain_embed_salient.py -- band-steer embedding restricted to salient regions.

This is exp2b: fsss/chain_embed.py's BandSteerAWAREEmbedder with the salient
gating its own docstring deferred.

    chain_embed.py: "No salient-region gating and no key hopping. [...] Those
    come back in step 3, once the chain is known to be sound. Note that when
    they do, the inherited mask builder will be handed `host`, not the original
    audio -- anchors must be computed on the original and mapped through, or
    placement will drift."

The chain is now known to be sound (verify() on 3 Emilia clips: stft roundtrip
133.9 dB, chain null 28.7 dB), so this is step 3. The drift warning is honoured
by routing through fsss/salient_mapped.py, which detects on the original and
maps sample indices through the plan's rational time scale.

THE TIME SCALE IS THE ONLY DIFFERENCE FROM exp1b
------------------------------------------------
band_steer resamples by plan.up/plan.down, so an anchor at original sample `a`
lands at host sample a*up/down. For the 1600-2400 Hz strip that is 4/15.
chain_embed_critical.py (exp1b) passes (1, M) instead -- 1/10 for a 10-band
split. Same mask builder, same merge rule, same half-width arithmetic; only the
fraction changes. That is deliberate: exp1b vs exp2b is meant to isolate the
sampling strategy, so anything else differing between them would be a confound.

Practical consequence of 4/15 vs 1/10: exp2 keeps ~163 host frames on a 10 s
clip where exp1 keeps only ~61, so the same anchor_rate produces a somewhat
lower coverage here. Both land in the 25-40% band at anchor_rate 1.2, which is
what the comparison needs.

WHY THIS SUBCLASSES RATHER THAN EDITS chain_embed.py
----------------------------------------------------
chain_embed.py is the operational exp2a path and has already produced published
numbers. Adding a flag there would put exp2a and exp2b on the same code object
and make it possible to move exp2a's behaviour by accident. One override in a
subclass keeps the baseline frozen.

    from fsss.chain_embed_salient import SalientBandSteerAWAREEmbedder
    e = SalientBandSteerAWAREEmbedder.from_embedder(base, target_band=(1600, 2400))
    e.verify(audio, 16000)
    y = e.embed(audio, 16000, watermark)
"""

import numpy as np

from fsss.chain_embed import BandSteerAWAREEmbedder
from fsss.salient_mapped import (COVERAGE_WARN, DEFAULT_ANCHOR_RATE,
                                 DEFAULT_REGION_MS, build_mapped_region_mask)
from fsss.salient_region import normalize_rows

__all__ = ["SalientBandSteerAWAREEmbedder"]


class SalientBandSteerAWAREEmbedder(BandSteerAWAREEmbedder):
    """exp2b -- band-steer host, embedding restricted to salient time regions."""

    @classmethod
    def from_embedder(cls, base, key=b"unused", target_band=(1600, 2400),
                      sampling_rate=16000, chain_in_loop=True, aware_band=None,
                      stft_center=True, tolerance_db=None,
                      anchor="librosa_flux", anchor_rate=DEFAULT_ANCHOR_RATE,
                      region_ms=DEFAULT_REGION_MS):
        obj = super().from_embedder(base, key=key, target_band=target_band,
                                    sampling_rate=sampling_rate,
                                    chain_in_loop=chain_in_loop,
                                    aware_band=aware_band,
                                    stft_center=stft_center,
                                    tolerance_db=tolerance_db)
        obj.anchor = anchor
        obj.anchor_rate = float(anchor_rate)
        obj.region_ms = float(region_ms)
        obj._orig_audio = None          # anchors come from THIS, never the host
        obj._orig_sr = int(sampling_rate)
        obj.last_mask_stats = None
        return obj

    # ---- the one override --------------------------------------------------- #
    def _build_staircase_mask(self, audio, sampling_rate, n_freq, n_frames):
        """`audio` arrives as the HOST -- the inherited embed() passes whatever
        it was given, and by this point we are inside the steered band. Using it
        for anchor detection is precisely the drift bug chain_embed.py warned
        about, so it is ignored in favour of self._orig_audio."""
        rows = self._writable_rows(sampling_rate, n_freq)

        if self._orig_audio is None:                    # nothing stashed: be stock
            M = np.zeros((n_freq, n_frames), dtype=bool)
            M[rows, :] = True
            self.last_mask_stats = {"n_anchors": 0, "n_regions": 1,
                                    "coverage": 1.0, "cells": int(M.sum()),
                                    "cells_full_stripe": int(len(rows) * n_frames),
                                    "fell_back": True, "n_frames": int(n_frames)}
            return M

        M, stats = build_mapped_region_mask(
            self._orig_audio, self._orig_sr,
            self.plan.up, self.plan.down,               # <- the time scale, 4/15
            n_freq, n_frames, self.hop_length, rows,
            anchor=self.anchor, anchor_rate=self.anchor_rate,
            region_ms=self.region_ms)
        self.last_mask_stats = stats

        if self.verbose:
            from aware.utils.logger import logger
            if stats["fell_back"]:
                logger.warning("chain_embed_salient: no anchors -- writing the "
                               "FULL stripe (equivalent to exp2a)")
            logger.info(f"chain_embed_salient: {stats['n_anchors']} anchors -> "
                        f"{stats['n_regions']} regions, coverage "
                        f"{stats['coverage']*100:.1f}% of {n_frames} frames, "
                        f"{stats['cells']} writable cells "
                        f"(full stripe would be {stats['cells_full_stripe']})")
            if stats["coverage"] > COVERAGE_WARN:
                logger.warning(
                    f"chain_embed_salient: coverage {stats['coverage']*100:.0f}% "
                    f"> {COVERAGE_WARN*100:.0f}% -- gating is effectively a NO-OP. "
                    f"Lower --anchor-rate or --region-ms.")
        return M

    def _writable_rows(self, sampling_rate, n_freq):
        """Rows the detector will actually read. Prefer the base's own indices so
        the mask cannot drift from the detector band; fall back to everything."""
        getter = getattr(self, "_get_embedding_frequency_indices", None)
        if getter is not None:
            try:
                rows = normalize_rows(getter(sampling_rate, self.frame_length)[0],
                                      n_freq)
                if rows.size:
                    return rows
            except Exception:                # signature drift across AWARE versions
                pass
        return np.arange(n_freq, dtype=np.int64)

    # ---- stash the original before the inherited embed() runs --------------- #
    def embed(self, audio: np.ndarray, sample_rate: int, watermark: np.ndarray) -> np.ndarray:
        self._orig_audio = np.asarray(audio, dtype=np.float32)
        self._orig_sr = int(sample_rate)
        return super().embed(audio, sample_rate, watermark)
