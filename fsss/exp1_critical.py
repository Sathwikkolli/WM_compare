"""
fsss/exp1_critical.py -- EXPERIMENT 1: critically-sampled subband watermarking.

WHAT THIS COMPARES
------------------
Three ways of putting a 20-bit watermark into the same speech file:

  aware   Original AWARE, exactly as shipped. Writes across its whole native
          band (1000-4000 Hz). This is the baseline everything is measured
          against.

  exp1a   Cut out ONE narrow strip of the spectrum (1600-2400 Hz), shrink it by
          keeping every 10th sample, watermark the shrunken strip, then grow it
          back and glue it to the untouched rest of the audio.

  exp1b   Exactly exp1a, except the watermark is only written during "salient"
          moments -- the loud onsets that librosa's spectral flux detector
          finds. Loud moments hide a watermark better than quiet ones, so the
          question is whether writing in fewer, better-chosen places sounds
          better without losing too much robustness.

WHY SHRINKING THE STRIP IS ALLOWED
----------------------------------
A signal needs roughly 2 samples per second for every Hz of BANDWIDTH -- not
for every Hz of frequency. The strip 1600-2400 Hz is only 800 Hz wide, so it
needs 1600 samples/second, one tenth of the original 16000. Keeping every 10th
sample throws away nine samples that were completely predictable from the one
we kept. Nothing is lost, and the strip lands neatly on 0-800 Hz.

The DSP that does this lives in fsss/band_critical.py, and its own smoke test
(`python -m fsss.band_critical`) proves the shrink/grow round trip is exact.

WHAT IS MEASURED
----------------
For clean audio and for every attack:
  BER   bit error rate, in percent. 0 = every bit recovered. Lower is better.
  conf  the detector's confidence that a watermark is present, 0 to 1.
        A file counts as "detected" when conf >= 0.5.
And for clean audio only, the perceptual cost:
  PESQ  speech quality, roughly 1 (bad) to 4.5 (transparent). Higher is better.
  SNR   how far below the audio the watermark sits, in dB. Higher is better.

RELATIONSHIP TO fsss/exp2_bandsteer.py
--------------------------------------
That file runs the SAME comparison for experiment 2, which reaches the strip a
different way (a frequency shift plus a resample, instead of decimation). The
two files are kept deliberately near-identical so that

    diff fsss/exp1_critical.py fsss/exp2_bandsteer.py

shows exactly what differs between the two experiments and nothing else.

HOW TO RUN
----------
    conda activate wmcompare
    python -m fsss.exp1_critical

Everything adjustable is in the SETTINGS block below. Results are written to
fsss_out/exp1_critical.csv and five PNG figures.
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save figures to disk; no display on a compute node
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.metrics.audio import BER, PESQ, SNR

import vox_attacks
from fsss.chain_embed_critical import CriticalBandAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR


# =========================================================================== #
#  SETTINGS -- everything you might want to change lives here
# =========================================================================== #

N_CLIPS = 3                 # how many speech files to test
DURATION_S = 10.0           # seconds taken from the start of each file

# The perceptual budget. AWARE lets each spectrogram value move by at most this
# many dB. LOWER means a louder, stronger, more audible watermark.
# It is deliberately the SAME for all three configs and is never changed, so any
# difference between them comes from the method and not from the strength.
TOLERANCE_DB = 4.0

WATERMARK_LENGTH = 20       # AWARE carries 20 bits
SEED = 0                    # fixes the random watermark, so runs are repeatable

# The strip. 10 bands across 0-8000 Hz makes each one 800 Hz wide, so band 2 is
# 1600-2400 Hz. Even-numbered bands come out the right way up after shrinking;
# odd ones come out mirrored, which is harmless but harder to think about.
STRIP_INDEX = 2
NUM_BANDS = 10

# Derived: the strip's real frequency edges. 10 bands across 0-8000 Hz makes
# each 800 Hz wide, so band 2 spans 1600-2400 Hz.
STRIP_LOW = STRIP_INDEX * (WORK_SR / 2 / NUM_BANDS)
STRIP_HIGH = (STRIP_INDEX + 1) * (WORK_SR / 2 / NUM_BANDS)

# Length of the band-pass filter. Longer = sharper edges = less leakage when we
# shrink the strip. 4095 was measured to be ~6 dB cleaner than 1023.
NUM_TAPS = 4095

# Salient-moment settings for exp1b.
# ANCHOR_RATE is deliberately low. After shrinking, a 10 s clip has only ~61
# spectrogram frames, so the usual 3.5 anchors/second would cover every frame
# and the gating would do nothing at all.
ANCHOR = "librosa_flux"
ANCHOR_RATE = 1.2           # salient points per second
REGION_MS = 250.0           # how much audio around each point to write into

DETECT_THRESHOLD = 0.5      # conf >= this counts as "watermark detected"

# Attacks. Same set as fsss/exp_v12_metapxyl_compare.py so the numbers here can
# be compared with the earlier experiments in this repo.
ATTACKS = [
    # the METAPXYL mastering-chain proxy
    "dynamic_compression", "echo", "mp3", "quantization", "lowpass",
    "gaussian_noise",
    # the VoxWatermark attacks that matter most for this design
    "time_stretch", "time_jitter", "highpass", "encodec",
    "background_noise", "opus",
]

OUT_DIR = os.path.join(BASE, "fsss_out")
TAG = "exp1"                # prefix for every output file


# =========================================================================== #
#  Small helpers
# =========================================================================== #

def attack_grid(attack_name):
    """The list of (label, parameter) strengths to try for one attack.

    Note the opus special case. cascade/vox_attacks.py builds opus bitrates as
    b*16 for b in [1,2,4,8,16,31], so the last one is 496 kbps -- above the
    256 kbps that libopus accepts. ffmpeg refuses it and floods the log with
    errors. We drop that one strength here instead of editing the shared attack
    module, which other experiments depend on.
    """
    grid = vox_attacks.VOX_GRID.get(attack_name, [])
    if attack_name == "opus":
        grid = [(label, p) for label, p in grid if float(p) <= 256]
    return grid


def band_snr(watermarked, original, sample_rate, low_hz, high_hz):
    """How much of the audio changed INSIDE the intended band, and outside it.

    Returns (snr_inside_dB, snr_outside_dB), each = 10*log10(original energy /
    changed energy) in that frequency range. Higher means less was changed.

    The outside number is the interesting one. A config that only touches its
    own band should leave everything else bit-identical, giving a huge value
    (60 dB or more). A small value means something is altering audio it was
    never supposed to touch -- a global gain error will do this, and it shows up
    in figure 1 as a faint copy of the whole original underneath the watermark.

    Worth watching for stock AWARE too: aware/src/aware/service/embed.py
    rescales the finished file by np.max(audio), the SIGNED maximum, while the
    embedder normalised by max(|audio|). When a waveform's largest excursion is
    negative -- common in speech -- those differ and the whole file comes back
    slightly quiet.
    """
    n = min(len(watermarked), len(original))
    x = np.asarray(original[:n], dtype=float)
    difference = np.asarray(watermarked[:n], dtype=float) - x

    spectrum_original = np.fft.rfft(x)
    spectrum_difference = np.fft.rfft(difference)
    frequencies = np.fft.rfftfreq(n, 1.0 / sample_rate)
    inside = (frequencies >= low_hz) & (frequencies <= high_hz)

    def ratio(mask):
        signal = np.sum(np.abs(spectrum_original[mask]) ** 2)
        changed = np.sum(np.abs(spectrum_difference[mask]) ** 2)
        return 10.0 * np.log10(signal / max(changed, 1e-30))

    return ratio(inside), ratio(~inside)


class Config:
    """One column of the results tables.

    Holds everything needed to watermark a file and then read the mark back:
      embedder       the object that writes the watermark
      detector_band  the frequency range the detector must look at (see below)
      to_detector    turns a finished audio file into whatever the detector
                     expects to be handed
      expected_band  where this config is SUPPOSED to change the audio, used by
                     band_snr to check that it did not change anything else
    """

    def __init__(self, name, embedder, detector_band, to_detector, description,
                 expected_band):
        self.name = name
        self.embedder = embedder
        self.detector_band = detector_band
        self.to_detector = to_detector
        self.description = description
        self.expected_band = expected_band

    def embed(self, audio, sample_rate, bits):
        return embed_watermark(audio, sample_rate=sample_rate,
                               watermark_bits=bits, model=self.embedder)

    def detect(self, audio, sample_rate, detector):
        # IMPORTANT: the AWARE detector blanks out everything outside its own
        # embedding_bands before reading. That range is different for each
        # config, and one detector object is shared by all of them, so it has to
        # be set immediately before every read. Earlier experiments in this repo
        # that forgot this step reported artificially weak results.
        detector.embedding_bands = self.detector_band
        return detect_watermark(self.to_detector(audio), sample_rate, detector)


def build_configs(base_embedder):
    """Create the three configs, all at the same TOLERANCE_DB."""
    configs = []

    # ---- aware: the original, untouched -------------------------------- #
    base_embedder.tolerance_db = TOLERANCE_DB
    native_band = tuple(getattr(base_embedder, "embedding_bands", (1000, 4000)))
    configs.append(Config(
        name="aware",
        embedder=base_embedder,
        detector_band=native_band,          # its own native range
        to_detector=lambda audio: audio,    # detector reads the file directly
        description=f"original AWARE, band {native_band[0]:.0f}-{native_band[1]:.0f} Hz",
        expected_band=native_band))         # it should only change its own band

    # ---- exp1a and exp1b: the shrunken strip --------------------------- #
    # No try/except here, unlike exp2_bandsteer.py. Shrinking works for every
    # band of a uniform split by construction, so there is no configuration
    # that can legitimately fail. If it ever does, that is a bug and should
    # stop the run loudly rather than be swallowed.
    for name, use_salient in (("exp1a", False), ("exp1b", True)):
        embedder = CriticalBandAWAREEmbedder.from_embedder(
            base_embedder,
            band_index=STRIP_INDEX,
            num_bands=NUM_BANDS,
            sampling_rate=WORK_SR,
            numtaps=NUM_TAPS,
            tolerance_db=TOLERANCE_DB,
            salient=use_salient,
            anchor=ANCHOR,
            anchor_rate=ANCHOR_RATE,
            region_ms=REGION_MS)

        # After shrinking, the strip fills the ENTIRE spectrum the detector
        # sees, so the detector must not blank anything out. Empty bins take
        # care of themselves: AWARE's budget is a percentage of each value, and
        # a percentage of zero is zero, so silent bins simply cannot move.
        configs.append(Config(
            name=name,
            embedder=embedder,
            detector_band=(0, WORK_SR / 2),
            to_detector=embedder.to_detector_input,   # cut + shrink, same as embedding
            description=("strip only" if not use_salient
                         else f"strip + salient ({ANCHOR}, {ANCHOR_RATE}/s)"),
            expected_band=(STRIP_LOW, STRIP_HIGH)))   # only the strip should move
    return configs


# =========================================================================== #
#  Running the experiment
# =========================================================================== #

def run_clean(configs, clips, audios, bits, detector, writer):
    """Watermark every clip with every config and measure the undamaged result.

    Returns
      results    {(config, "clean"): [(ber, conf), ...]}
      quality    {config: {"pesq": [...], "snr": [...]}}
      marked     {(clip_index, config): watermarked audio}  reused by the attacks
    """
    ber_metric, pesq_metric, snr_metric = BER(), PESQ(), SNR()
    results, quality, marked = {}, {}, {}

    print("=" * 74)
    print("STEP 1 -- clean audio (no attacks)")
    print("=" * 74)

    for clip_index, (path, audio) in enumerate(zip(clips, audios)):
        print(f"\nclip {clip_index + 1}/{len(audios)}: {os.path.basename(path)}")

        for config in configs:
            watermarked = config.embed(audio, WORK_SR, bits)
            marked[(clip_index, config.name)] = watermarked

            recovered, confidence = config.detect(watermarked, WORK_SR, detector)
            ber = float(ber_metric(recovered, bits))
            confidence = float(confidence)

            # PESQ can fail on clips with very little speech in them; that is a
            # property of the clip, not of the watermark, so skip rather than stop.
            try:
                pesq_score = float(pesq_metric(watermarked, audio, WORK_SR))
            except Exception:
                pesq_score = float("nan")
            snr_score = float(snr_metric(watermarked, audio))

            # Did it change only what it was supposed to change?
            snr_in, snr_out = band_snr(watermarked, audio, WORK_SR,
                                       config.expected_band[0],
                                       config.expected_band[1])

            # every result is stored as (BER, confidence, PESQ) so the clean row
            # and the attack rows can share one table
            results.setdefault((config.name, "clean"), []).append(
                (ber, confidence, pesq_score))
            q = quality.setdefault(config.name,
                                   {"pesq": [], "snr": [], "in": [], "out": []})
            if not np.isnan(pesq_score):
                q["pesq"].append(pesq_score)
            q["snr"].append(snr_score)
            q["in"].append(snr_in)
            q["out"].append(snr_out)

            # coverage only exists for the salient config
            stats = getattr(config.embedder, "last_mask_stats", None) or {}
            coverage = stats.get("coverage", float("nan"))

            writer.writerow([clip_index, config.name, TOLERANCE_DB, "clean", "",
                             round(ber, 2), round(confidence, 4),
                             int(confidence >= DETECT_THRESHOLD),
                             round(pesq_score, 3), round(snr_score, 2),
                             round(snr_in, 1), round(snr_out, 1),
                             round(coverage, 3) if coverage == coverage else ""])

            line = (f"  {config.name:6s}  BER {ber:6.2f}%   conf {confidence:5.3f}"
                    f"   PESQ {pesq_score:4.2f}   SNR {snr_score:6.2f} dB")
            if coverage == coverage:                     # not NaN
                line += f"   coverage {coverage * 100:.0f}%"
            print(line)

    return results, quality, marked


def run_attacks(configs, clips, audios, marked, bits, detector, results, writer):
    """Damage every watermarked file in every way, and try to read it back.

    PESQ is measured here too, against the ORIGINAL clean audio. So it scores
    watermark damage AND attack damage together, which means the attack usually
    dominates -- a heavy mp3 sounds bad whether or not it carries a watermark.
    That is still useful, because every config gets the identical attack, so
    comparing configs within one row isolates the watermark's share of it.
    """
    ber_metric, pesq_metric = BER(), PESQ()

    print("\n" + "=" * 74)
    print("STEP 2 -- attacks")
    print("=" * 74)

    for clip_index, (path, audio) in enumerate(zip(clips, audios)):
        print(f"\nclip {clip_index + 1}/{len(clips)}: {os.path.basename(path)}")

        for config in configs:
            watermarked = marked[(clip_index, config.name)]

            for attack in ATTACKS:
                for label, parameter in attack_grid(attack):
                    try:
                        damaged = vox_attacks.apply(
                            attack, parameter, watermarked.astype("float32"), WORK_SR)
                    except Exception:
                        continue                 # attack needs a missing package
                    if damaged is None:
                        continue

                    recovered, confidence = config.detect(damaged, WORK_SR, detector)
                    ber = float(ber_metric(recovered, bits))
                    confidence = float(confidence)

                    # PESQ refuses some badly mangled inputs; that is a fact
                    # about the attack, so record it as missing and carry on.
                    try:
                        pesq_score = float(pesq_metric(damaged, audio, WORK_SR))
                    except Exception:
                        pesq_score = float("nan")

                    results.setdefault((config.name, attack), []).append(
                        (ber, confidence, pesq_score))
                    # the last three columns are clean-audio-only measurements
                    writer.writerow([clip_index, config.name, TOLERANCE_DB,
                                     attack, label, round(ber, 2),
                                     round(confidence, 4),
                                     int(confidence >= DETECT_THRESHOLD),
                                     round(pesq_score, 3), "", "", "", ""])
            print(f"  {config.name:6s}  done")


# =========================================================================== #
#  Printing the tables
# =========================================================================== #

def average(results, config_name, attack):
    """(mean BER, mean confidence, mean PESQ, number detected, number tried).

    PESQ is averaged over the cases where it succeeded; if it failed everywhere
    the mean is NaN and the table prints a dash.
    """
    rows = results.get((config_name, attack), [])
    if not rows:
        return float("nan"), float("nan"), float("nan"), 0, 0
    bers = [r[0] for r in rows]
    confs = [r[1] for r in rows]
    pesqs = [r[2] for r in rows if not np.isnan(r[2])]
    detected = sum(1 for r in rows if r[1] >= DETECT_THRESHOLD)
    return (float(np.mean(bers)), float(np.mean(confs)),
            float(np.mean(pesqs)) if pesqs else float("nan"),
            detected, len(rows))


def print_tables(results, configs, quality):
    names = [c.name for c in configs]
    rows = ["clean"] + ATTACKS

    # ---- table 1: BER, confidence and PESQ, per attack ------------------ #
    width = 22 + 22 * len(names)
    print("\n" + "=" * width)
    print("TABLE 1 -- bit error rate, detector confidence and audio quality")
    print("           BER%  percent of bits wrong. Lower is better, 0 = perfect.")
    print("           conf  detector confidence 0-1. Higher is better, >= 0.5 = detected.")
    print("           PESQ  quality of the attacked file vs the clean original.")
    print("                 Higher is better. The attack dominates this number, so")
    print("                 compare configs ACROSS a row, not rows against each other.")
    print("=" * width)
    print(f"  {'attack':20s}" + "".join(f"{n:>22s}" for n in names))
    print(f"  {'':20s}" + "".join(f"{'BER%':>7s}{'conf':>7s}{'PESQ':>8s}" for _ in names))
    print("  " + "-" * (width - 2))
    for attack in rows:
        line = f"  {attack:20s}"
        for name in names:
            ber, conf, pesq, _, tried = average(results, name, attack)
            if tried == 0:
                line += "-".rjust(22)
            else:
                pesq_text = "   -" if np.isnan(pesq) else f"{pesq:8.2f}"
                line += f"{ber:>7.2f}{conf:>7.3f}{pesq_text:>8s}"
        print(line)
    print("=" * width)

    # ---- table 2: how often the watermark was found at all -------------- #
    print("\n" + "=" * width)
    print("TABLE 2 -- detection rate (files where confidence >= 0.5)")
    print("=" * width)
    print(f"  {'attack':20s}" + "".join(f"{n:>22s}" for n in names))
    print("  " + "-" * (width - 2))
    for attack in rows:
        line = f"  {attack:20s}"
        for name in names:
            _, _, _, detected, tried = average(results, name, attack)
            line += "-".rjust(22) if tried == 0 else f"{f'{detected}/{tried}':>22s}"
        print(line)
    print("=" * width)

    # ---- table 3: what the watermark costs in audio quality ------------- #
    print("\n" + "=" * 78)
    print("TABLE 3 -- audio quality on clean files")
    print("=" * 78)
    print(f"  {'config':10s}{'PESQ':>10s}{'SNR dB':>10s}"
          f"{'in-band dB':>14s}{'out-band dB':>15s}")
    print("  " + "-" * 76)
    for name in names:
        q = quality.get(name, {"pesq": [], "snr": [], "in": [], "out": []})
        mean = lambda key: np.mean(q[key]) if q.get(key) else float("nan")
        print(f"  {name:10s}{mean('pesq'):>10.2f}{mean('snr'):>10.2f}"
              f"{mean('in'):>14.1f}{mean('out'):>15.1f}")
    print("=" * 78)
    print("  PESQ        speech quality, higher is better")
    print("  SNR         whole-file watermark loudness, higher = quieter mark")
    print("  in-band     how much was changed inside the intended band")
    print("              (lower = a stronger watermark went in there)")
    print("  out-band    how much was changed OUTSIDE it. Should be very large")
    print("              (60 dB+): a config that only touches its own band must")
    print("              leave the rest of the audio bit-identical. A small")
    print("              value means audio is being altered that never should")
    print("              have been -- see figure 1, where it appears as a faint")
    print("              copy of the whole original underneath the watermark.")
    print(f"\n  All configs used tolerance_db = {TOLERANCE_DB} -- the same perceptual")
    print("  budget -- so differences come from the method, not the strength.")
    print("  But note that the budget is applied to whatever each config hands")
    print("  AWARE, and for the strip configs that is a quieter, isolated band.")
    print("  The strip watermarks therefore still end up louder in absolute")
    print("  terms; the SNR column is the honest measure of that.")


# =========================================================================== #
#  Figures
# =========================================================================== #

def spectrogram_db(signal, n_fft=1024, hop=256):
    """Magnitude spectrogram in dB, floored 80 dB below the peak so the colour
    scale is not dominated by numerical dust."""
    import librosa
    S = np.abs(librosa.stft(np.asarray(signal, dtype=np.float32),
                            n_fft=n_fft, hop_length=hop))
    db = 20.0 * np.log10(S + 1e-10)
    return np.maximum(db, db.max() - 80.0)


def figure_placement(original, marked_by_config, configs):
    """FIGURE 1 -- where each config actually put the watermark.

    Shows the difference between the watermarked file and the original. For the
    strip configs this should be a bright horizontal band at 1600-2400 Hz and
    near-black everywhere else, because the rest of the audio is never touched.
    """
    strip_low = STRIP_INDEX * (WORK_SR / 2 / NUM_BANDS)
    strip_high = (STRIP_INDEX + 1) * (WORK_SR / 2 / NUM_BANDS)

    fig, axes = plt.subplots(1, len(configs) + 1, figsize=(4.4 * (len(configs) + 1), 3.5))
    axes[0].imshow(spectrogram_db(original), origin="lower", aspect="auto",
                   cmap="magma", extent=[0, len(original) / WORK_SR, 0, WORK_SR / 2])
    axes[0].set_title("original audio", fontsize=10)
    axes[0].set_ylabel("frequency (Hz)", fontsize=9)

    for ax, config in zip(axes[1:], configs):
        wm = marked_by_config[config.name]
        n = min(len(wm), len(original))
        difference = np.asarray(wm[:n], dtype=float) - np.asarray(original[:n], dtype=float)
        ax.imshow(spectrogram_db(difference), origin="lower", aspect="auto",
                  cmap="magma", extent=[0, n / WORK_SR, 0, WORK_SR / 2])
        ax.set_title(f"watermark only: {config.name}", fontsize=10)

    for ax in axes:
        ax.axhline(strip_low, color="cyan", lw=1.0)
        ax.axhline(strip_high, color="cyan", lw=1.0)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"{TAG.upper()} figure 1 -- where the watermark lives "
                 f"(cyan lines mark the {strip_low:.0f}-{strip_high:.0f} Hz strip)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "f1_placement")


def figure_bars(results, configs, value_index, title, ylabel, filename,
                reference_line=None, invert_note=""):
    """FIGURES 2 and 3 -- one grouped bar per attack, one bar per config.

    value_index picks what to plot: 0 = BER, 1 = confidence.
    """
    names = [c.name for c in configs]
    rows = ["clean"] + ATTACKS
    x = np.arange(len(rows))
    bar_width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    for i, name in enumerate(names):
        values = []
        for attack in rows:
            stats = average(results, name, attack)
            values.append(stats[value_index])
        ax.bar(x + i * bar_width - 0.4 + bar_width / 2, values, bar_width, label=name)

    if reference_line is not None:
        ax.axhline(reference_line, color="0.35", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(rows, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{TAG.upper()} {title}{invert_note}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, filename)


def figure_quality(quality, configs):
    """FIGURE 4 -- what the watermark costs, on clean audio."""
    names = [c.name for c in configs]
    pesq_values = [np.mean(quality[n]["pesq"]) if quality[n]["pesq"] else 0 for n in names]
    snr_values = [np.mean(quality[n]["snr"]) if quality[n]["snr"] else 0 for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(names, pesq_values, color="#2a7fbf")
    ax1.set_title("PESQ -- speech quality (higher is better)", fontsize=10)
    ax1.set_ylim(1, 4.6)
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(names, snr_values, color="#c0742a")
    ax2.set_title("SNR -- watermark loudness (higher = quieter mark)", fontsize=10)
    ax2.set_ylabel("dB", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{TAG.upper()} figure 5 -- cost on clean audio "
                 f"(all configs at tolerance_db = {TOLERANCE_DB})", fontsize=11)
    fig.tight_layout()
    save(fig, "f5_quality")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{TAG}_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    print("=" * 74)
    print("EXPERIMENT 1 -- critically-sampled subband watermarking")
    print("=" * 74)

    # ---- audio --------------------------------------------------------- #
    clips = pick_clips(EMILIA_CSV, N_CLIPS)
    if not clips:
        print("No audio found. Check the EMILIA_CSV environment variable.")
        return
    wanted_samples = int(round(DURATION_S * WORK_SR))
    audios = [load_16k(path)[:wanted_samples] for path in clips]

    # one fixed random watermark, reused everywhere, so every config is asked
    # to carry exactly the same message
    bits = np.random.default_rng(SEED).integers(
        0, 2, size=WATERMARK_LENGTH, dtype=np.int32)

    strip_low = STRIP_INDEX * (WORK_SR / 2 / NUM_BANDS)
    strip_high = (STRIP_INDEX + 1) * (WORK_SR / 2 / NUM_BANDS)
    print(f"clips        : {len(audios)} x {DURATION_S:.0f} s at {WORK_SR} Hz")
    print(f"strip        : band {STRIP_INDEX} of {NUM_BANDS} "
          f"= {strip_low:.0f}-{strip_high:.0f} Hz")
    print(f"tolerance_db : {TOLERANCE_DB}  (identical for all configs, never changed)")
    print(f"watermark    : {WATERMARK_LENGTH} bits, seed {SEED}")
    print(f"attacks      : {len(ATTACKS)} families, each at several strengths")

    try:
        import torch
        print(f"device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception:
        pass

    # ---- model and configs --------------------------------------------- #
    base_embedder, detector = load()
    configs = build_configs(base_embedder)
    print("\nconfigs:")
    for config in configs:
        print(f"  {config.name:6s}  {config.description}")

    # ---- sanity check before spending time on the full run -------------- #
    # A wiring mistake in the cut/shrink/grow chain produces a silently weak
    # watermark rather than an error, so measure it rather than assume it.
    for config in configs:
        if hasattr(config.embedder, "verify"):
            print(f"\n--- chain check: {config.name} " + "-" * 32)
            config.embedder.verify(audios[0], WORK_SR)

    # ---- run ------------------------------------------------------------ #
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{TAG}_critical.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "config", "tolerance_db", "attack", "strength",
                         "ber_percent", "confidence", "detected",
                         "pesq", "snr_db", "snr_in_band_db", "snr_out_band_db",
                         "coverage"])

        results, quality, marked = run_clean(
            configs, clips, audios, bits, detector, writer)
        run_attacks(configs, clips, audios, marked, bits, detector, results, writer)

    # ---- report ---------------------------------------------------------- #
    print_tables(results, configs, quality)

    print("\nfigures:")
    figure_placement(audios[0],
                     {c.name: marked[(0, c.name)] for c in configs},
                     configs)
    figure_bars(results, configs, 0,
                "figure 2 -- bit error rate by attack", "BER (%)",
                "f2_ber", invert_note="   (lower is better)")
    figure_bars(results, configs, 1,
                "figure 3 -- detector confidence by attack", "confidence",
                "f3_confidence", reference_line=DETECT_THRESHOLD,
                invert_note="   (dashed line = detection threshold 0.5)")
    figure_bars(results, configs, 2,
                "figure 4 -- PESQ by attack", "PESQ",
                "f4_pesq", invert_note="   (higher is better)")
    figure_quality(quality, configs)

    print(f"\nwrote {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
