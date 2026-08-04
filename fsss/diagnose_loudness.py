"""
fsss/diagnose_loudness.py -- WHY is the strip watermark louder than AWARE's?

THE QUESTION
------------
Measured on 3 clips at tolerance 6, the watermark-to-signal ratio was:

    aware  -16.7 dB        exp2a  -9.4 dB        exp1a  -6.1 dB

Same tolerance_db for all three, yet the strip configs put 7-11 dB more energy
into the file. PESQ follows exactly that ordering. So the PESQ gap is explained
by loudness -- but WHY they are louder has never been measured, only argued from
reading the code.

This script measures it. It runs ONE clip through ONE config and reports every
scale factor along the way, so the answer is a number rather than an argument.

THE HYPOTHESIS BEING TESTED
---------------------------
AWARE's embed pipeline ends with WaveformNormalizer, so `super().embed(host)`
returns a signal scaled to peak 1.0 regardless of how quiet the input host was.
Meanwhile `lo` -- the rest of the audio, which we set aside -- is still at its
original scale. chain_embed.py then adds them:

    y = lo + t_from_model(host_wm, ...)          (chain_embed.py:160)

If that is right, the strip is amplified by roughly 1 / peak(host_raw) before
being glued back, and since an isolated 800 Hz band is much quieter than the
full mix, that factor is large.

WHAT DECIDES IT
---------------
  boost = peak(host_wm) / peak(host_raw)

    ~1.0  ->  hypothesis WRONG. AWARE returned the host at its original level,
              nothing was amplified, and the loudness gap comes from somewhere
              else -- look at the strip energy numbers below instead.

    >>1.0 ->  hypothesis RIGHT, and `boost` in dB should roughly match the
              measured strip-energy increase.

Two independent cross-checks are printed alongside it:

  strip energy change    original vs output, INSIDE the band. The perceptual
                         clamp allows each value to move by at most
                         +/- tolerance_db, so a correctly scaled watermark
                         cannot raise the band's energy by much more than that.
                         A far larger number means the band was rescaled, not
                         watermarked.

  outside energy change  original vs output, OUTSIDE the band. Should be ~0 dB.
                         `lo` is never touched, so anything else here is a bug
                         -- and it is what makes figure 1's difference plot show
                         a faint copy of the whole original.

`aware` is measured the same way as a control. If IT also shows a non-zero
outside-change, the effect is inherited from stock AWARE and is not something
the strip methods introduced.

    conda activate wmcompare
    python -m fsss.diagnose_loudness
    python -m fsss.diagnose_loudness --config exp1a
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark

from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

STRIP_INDEX, NUM_BANDS = 2, 10
STRIP_LOW = STRIP_INDEX * (WORK_SR / 2 / NUM_BANDS)      # 1600 Hz
STRIP_HIGH = (STRIP_INDEX + 1) * (WORK_SR / 2 / NUM_BANDS)   # 2400 Hz
TOLERANCE_DB = 4.0
DURATION_S = 10.0


def peak(v):
    return float(np.max(np.abs(np.asarray(v, dtype=float))))


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2)))


def band_energy_db(signal, sample_rate, low_hz, high_hz, inside=True):
    """Energy in dB inside (or outside) a frequency range."""
    v = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(v)
    frequencies = np.fft.rfftfreq(len(v), 1.0 / sample_rate)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not inside:
        mask = ~mask
    return 10.0 * np.log10(np.sum(np.abs(spectrum[mask]) ** 2) + 1e-30)


def report_energies(original, output, sample_rate, low_hz, high_hz, label):
    """How much the band, and everything else, changed from input to output."""
    n = min(len(original), len(output))
    a, b = original[:n], output[:n]

    inside = (band_energy_db(b, sample_rate, low_hz, high_hz, True)
              - band_energy_db(a, sample_rate, low_hz, high_hz, True))
    outside = (band_energy_db(b, sample_rate, low_hz, high_hz, False)
               - band_energy_db(a, sample_rate, low_hz, high_hz, False))

    print(f"  {label}")
    print(f"    energy change INSIDE  {low_hz:.0f}-{high_hz:.0f} Hz : {inside:+7.2f} dB")
    print(f"    energy change OUTSIDE {low_hz:.0f}-{high_hz:.0f} Hz : {outside:+7.2f} dB")
    return inside, outside


def main(argv):
    which = argv[argv.index("--config") + 1] if "--config" in argv else "exp2a"

    clips = pick_clips(EMILIA_CSV, 1)
    if not clips:
        print("no audio found; check EMILIA_CSV")
        return
    audio = load_16k(clips[0])[:int(round(DURATION_S * WORK_SR))]

    print("=" * 74)
    print(f"LOUDNESS DIAGNOSIS -- {which}")
    print("=" * 74)
    print(f"clip          : {os.path.basename(clips[0])}  ({len(audio)} samples)")
    print(f"tolerance_db  : {TOLERANCE_DB}")
    print(f"strip         : {STRIP_LOW:.0f}-{STRIP_HIGH:.0f} Hz\n")

    base_embedder, _ = load()
    base_embedder.tolerance_db = TOLERANCE_DB
    bits = np.random.default_rng(0).integers(0, 2, size=20, dtype=np.int32)

    # ---- 1. the control: stock AWARE -------------------------------------- #
    print("-" * 74)
    print("CONTROL -- stock AWARE (no strip, no chain)")
    print("-" * 74)
    aware_out = embed_watermark(audio, sample_rate=WORK_SR,
                                watermark_bits=bits, model=base_embedder)
    print(f"  peak in {peak(audio):.4f}   peak out {peak(aware_out):.4f}"
          f"   ratio {peak(aware_out) / max(peak(audio), 1e-12):.4f}")
    print(f"  rms  in {rms(audio):.4f}   rms  out {rms(aware_out):.4f}"
          f"   ratio {rms(aware_out) / max(rms(audio), 1e-12):.4f}")
    report_energies(audio, aware_out, WORK_SR, 1000, 4000, "stock, over its own band:")

    # ---- 2. build the strip config ---------------------------------------- #
    print("\n" + "-" * 74)
    print(f"TEST -- {which}")
    print("-" * 74)

    if which.startswith("exp1"):
        from fsss.band_critical import CriticalPlan, t_analyze, taps_for
        from fsss.chain_embed_critical import CriticalBandAWAREEmbedder
        embedder = CriticalBandAWAREEmbedder.from_embedder(
            base_embedder, band_index=STRIP_INDEX, num_bands=NUM_BANDS,
            sampling_rate=WORK_SR, numtaps=4095, tolerance_db=TOLERANCE_DB,
            salient=which.endswith("b"))
        plan = embedder.plan
        taps = taps_for(plan)
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        lo, host_raw = t_analyze(x, plan, taps)
    else:
        from fsss.band_steer_torch import t_analyze, taps_for
        if which.endswith("b"):
            from fsss.chain_embed_salient import SalientBandSteerAWAREEmbedder as Cls
        else:
            from fsss.chain_embed import BandSteerAWAREEmbedder as Cls
        embedder = Cls.from_embedder(
            base_embedder, target_band=(STRIP_LOW, STRIP_HIGH),
            sampling_rate=WORK_SR, tolerance_db=TOLERANCE_DB)
        plan = embedder.plan
        taps = taps_for(plan)
        x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
        lo, host_raw = t_analyze(x, plan, taps)

    host_raw_np = host_raw.numpy()

    # ---- 3. what AWARE gives back for that host --------------------------- #
    # Call the inherited embed exactly the way chain_embed.embed does, so the
    # returned host can be inspected before it is glued back on.
    embedder._lo = lo
    embedder._host_len = int(host_raw.shape[-1])
    if hasattr(embedder, "_lo_dev"):          # exp1 caches lo on the device
        embedder._lo_dev = None
    embedder._orig_audio = np.asarray(audio, dtype=np.float32)
    embedder._orig_sr = WORK_SR

    from fsss.staircase import StaircaseAWAREEmbedder
    host_wm = StaircaseAWAREEmbedder.embed(
        embedder, host_raw_np.astype(audio.dtype), WORK_SR, bits)

    boost = peak(host_wm) / max(peak(host_raw_np), 1e-12)

    print(f"  peak(original audio)        {peak(audio):10.6f}")
    print(f"  peak(host, raw)             {peak(host_raw_np):10.6f}"
          f"   <- the isolated band, before AWARE")
    print(f"  peak(host, watermarked)     {peak(host_wm):10.6f}"
          f"   <- what AWARE handed back")
    print()
    print(f"  BOOST = peak(wm) / peak(raw) {boost:9.3f}x"
          f"   = {20 * np.log10(max(boost, 1e-12)):+.2f} dB")
    print()
    if boost > 2.0:
        print("  >> AWARE returned the band at a DIFFERENT level than it was given.")
        print("     Its pipeline ends in WaveformNormalizer, so the output is")
        print("     scaled to peak 1.0 whatever went in, while `lo` stayed at its")
        print("     original scale. Adding them re-balances the mix.")
    else:
        print("  >> AWARE returned the band at essentially the level it was given,")
        print("     so the normalisation hypothesis is WRONG and the loudness gap")
        print("     has another cause. Read the energy numbers below.")

    # ---- 4. the end-to-end result ----------------------------------------- #
    print()
    strip_out = embed_watermark(audio, sample_rate=WORK_SR,
                                watermark_bits=bits, model=embedder)
    print(f"  peak in {peak(audio):.4f}   peak out {peak(strip_out):.4f}"
          f"   ratio {peak(strip_out) / max(peak(audio), 1e-12):.4f}")
    inside, outside = report_energies(audio, strip_out, WORK_SR,
                                      STRIP_LOW, STRIP_HIGH,
                                      f"{which}, over the strip:")

    # ---- 5. verdict -------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  The clamp allows each spectrogram value to move by at most")
    print(f"  +/- {TOLERANCE_DB} dB, so a correctly scaled watermark cannot raise the")
    print(f"  strip's total energy by much more than that.")
    print()
    print(f"  measured strip energy change : {inside:+.2f} dB")
    if inside > TOLERANCE_DB + 3:
        print(f"  -> FAR above the budget. The band was RESCALED, not just")
        print(f"     watermarked. That is the cause of the PESQ gap, and the fix")
        print(f"     is to restore the watermarked host to the raw host's level")
        print(f"     before synthesis (chain_embed.py:160, chain_embed_critical")
        print(f"     .py's embed) rather than to change tolerance_db.")
    else:
        print(f"  -> within the budget. The band was NOT rescaled, so the PESQ gap")
        print(f"     is not a scaling bug. Next suspect: the strip is only 800 Hz")
        print(f"     wide, so the same 20 bits must be carried by ~4x fewer")
        print(f"     coefficients than stock AWARE uses, forcing each one harder")
        print(f"     against the clamp.")
    print()
    print(f"  measured outside energy change : {outside:+.2f} dB")
    print(f"  -> should be ~0. Compare against the stock control above: if BOTH")
    print(f"     are non-zero, it is inherited from AWARE, not from the strip.")


if __name__ == "__main__":
    main(sys.argv[1:])
