# informed — does an aligned original recover watermarks blind detection loses?

Design, registered predictions and the validity constraint live in
`results/2026-08-28_informed-detection/README.md`. **Read that first.**
This directory is Phase A-1 only: the music sweep.

## What Phase A-1 answers

We know AWARE's confidence crosses 0.50 near **+4.9 dB** speech-to-music SNR.
We do not know where the *audio* stops being usable. Without both numbers the
crossing cannot be located, and the parent plan's lead candidate is unsupported.

> At the SNR where detection fails, is the audio still usable?

Positive answer ⇒ a band of usable audio with an undetectable watermark, which
is the target informed detection exists to attack. Negative ⇒ the watermark
outlives usability here as it does for high-pass and quantisation, and the
parent plan loses its lead candidate.

## Run it

```bash
conda activate wmcompare
export WM_COMPARE_BASE=$HOME/wm_compare    # a login shell does not have this set
cd $WM_COMPARE_BASE/informed

# once: extra packages, with a rollback point
pip freeze > $HOME/wmcompare_freeze_before_informed.txt
pip install --dry-run -r requirements_extra.txt
pip install -r requirements_extra.txt

# sanity checks before burning cluster time
python quality.py       # MUST PASS -- validates the quality backend
python clips.py         # -> clips.json, the frozen 50-clip set

# time one clip before submitting 50 of them
time python music_sweep.py --clip 0

sbatch music_sweep.sbatch          # array 0-49

# after the array finishes, on a login node
python score_music_sweep.py        # -> summary_phase_a1.md + data/music_metrics.csv
python plots_music_sweep.py        # -> figures/crossing.png, metric_disagreement.png
```

Set `--account` in `music_sweep.sbatch` if `hafiz1` is not right.

## The attack screen (Phase A) — 27 attacks, 118 configs

`music_sweep.py` measures one attack precisely. The screen measures all of them
and places them relative to each other.

```bash
python attacks_screen.py          # print the grid; what is installed
python attacks_screen.py --check  # actually run each attack once
time python screen_sweep.py --clip 0     # TIME IT before submitting 50
sbatch screen.sbatch                      # array 0-49
python score_screen.py                    # -> summary_screen.md + selected_attacks.json
```

| category | attacks | why it is here |
|---|---|---|
| additive | `music_bed`, `gaussian_noise`, `noise_babble`, `noise_factory`, `noise_machinegun` | watermark is **masked**, damage is linear — where informed detection should win |
| codec | `mp3`, `aac`, `opus`, `encodec`, `resample_roundtrip`, `platform_reencode` | the waveform is **rebuilt**, so subtraction's additive assumption breaks |
| filter | `highpass`, `lowpass`, `smooth` | a band is **deleted** — what is gone cannot be recovered |
| dynamics | `quantization`, `dynamic_compression`, `dynamic_expansion`, `volume`, `mastering_chain` | level and envelope, including the client's chain |
| acoustic | `reverb`, `echo`, `stereo_widen` | realistic and quality-preserving |
| enhancement | `denoise` | the one attack where quality may go **up** while the mark is stripped as noise |
| temporal | `time_stretch` ⚠, `time_jitter` | ⚠ no aligner handles time-stretch, so Phase B cannot use it |
| control | `inverse_polarity`, `phase_shift` | must not move detection; if they do, the harness is wrong |

**17 families are delegated to `cascade/vox_attacks.py`, not rewritten.** Eight
are new, and two inherited ones are fixed:

- `background_noise` in vox always reads `wavs[0]`, so only `babble.wav` was ever
  used and `factory1.wav` / `machinegun.wav` were dead files. Split here into
  three separate attacks.
- `music_bed` is absent from vox entirely, despite being the one case with
  existing evidence of failure in usable audio.

`score_screen.py` classifies each attack at **s\***, the strongest setting whose
audio is still usable:

| verdict | meaning |
|---|---|
| **VULNERABLE** | detection already fails at s\* — usable audio, no detection |
| **SECURE** | detection holds at s\*; the attacker must wreck the audio to win |
| **NO_FLOOR** | quality never fell below the floor in the swept range |
| **UNAVAILABLE** | the attack could not run — not a result either way |

Only VULNERABLE, alignment-compatible attacks are written to
`selected_attacks.json`. **Phase B reads that file rather than hardcoding a
list** — choosing the attacks by evidence is the screen's whole purpose.

## Files


| File | Does |
|---|---|
| `quality.py` | DNSMOS (no-reference) + PESQ/STOI via `cascade_lib`. **Self-test is a gate, not decoration** |
| `clips.py` | The frozen 50 clips, selection copied from `cascade/emilia_bench.py` |
| `music_sweep.py` | Slurm-array runner, one task per clip |
| `score_music_sweep.py` | Crossings and the five predictions. **Metric meanings at the top of the file** |
| `plots_music_sweep.py` | `crossing.png` (the headline) and `metric_disagreement.png` |
| `attacks_screen.py` | The 27-attack grid; delegates to `vox_attacks`, adds 8 |
| `screen_sweep.py` | Slurm-array runner for the full screen |
| `score_screen.py` | Verdicts -> `summary_screen.md` + **`selected_attacks.json`** |
| `requirements_extra.txt` | The extra packages + snapshot/rollback procedure |

## Three things that make this trustworthy

**1. It fixes an n = 1 result.** `real_attacks_experiment.py` uses one LibriVox
clip and one music track, with no trial loop anywhere in the file — despite
`real_attacks_summary.csv` labelling its rows "mean over trials". Masking depends
heavily on what the speech and music are, so 4.9 dB from a single pair cannot be
a threshold. This uses 50 speaker-stratified clips and reports the spread.

The `3.8 dB` grid point from that script is deliberately dropped: it was chosen
to bracket the n=1 result and would bias a re-measurement toward confirming it.

**2. The clip set is not new.** Selection is copied from
`cascade/emilia_bench.py:stage_select()` — same filters, same seed, same
speaker-stratified round robin — so this run uses the *same 50 clips* as the
cascade benchmark. `clips.py` asserts the constants still match and fails on
drift rather than silently diverging.

**3. Predictions were registered before the run.** Five of them, in
`results/2026-08-28_informed-detection/PHASE_A_PLAN.md`. Two can stop the project
early, which is the point of running this first.

## The quality metric, and why it is a gate

PESQ measures **fidelity to the original**. An attacker does not need fidelity —
they need the result to **sound acceptable**. Speech over a music bed has poor
PESQ and sounds completely normal; screening on PESQ would file it under
"destroyed audio" alongside `highpass_0.2` and throw away the best candidate we
have. So the primary metric is **DNSMOS**, no-reference.

DNSMOS specifically, not any no-reference score: `cascade/emilia_bench.py:98`
already filters source clips with `dnsmos >= 3.0`. Using the same metric makes
the usability floor here **the same 3.0** the project already uses to choose
clips, rather than a number invented for this run.

`quality.py` implements a torchaudio SQUIM fallback, and `music_sweep.py`
**refuses to run on it** without `--force`. Two reasons: its scale is not
comparable to the manifest, and it is unvalidated — on both signals it could be
tested against locally it scored noisy audio *higher* than clean (tones: 3.92
clean vs 3.95 noisy; 8 kHz speech: 2.57 clean vs 3.98 noisy). Both inputs were
out-of-distribution, so that does not condemn SQUIM — but a backend that has
never been shown to decrease with damage cannot be used to locate a crossing.

`python quality.py` checks the backend is **monotone in added noise**, on real
wideband speech. The sbatch runs it and aborts on failure.

## Known limitations

- **One music track** in the main sweep. Its spectrum may mask AWARE's bands
  unusually well or badly. The diversity probe (`--clips 0-9 --music all`)
  exists to check that, but needs three contrasting tracks added to
  `MUSIC_SOURCES` first — left empty rather than guessed, because a broken URL
  discovered inside a 50-task array is expensive.
- **10-second clips**, matching `emilia_bench`'s `SECONDS`. Longer audio gives a
  detector more evidence, so the crossing may sit differently for real uploads.
- **DNSMOS is a model, not a listening test.** It is a proxy for "sounds
  acceptable", and it has never been validated on speech-plus-music mixes
  specifically. If section 1 hinges on a small margin, a listening check on the
  saved mixes (`--save-audio`) is worth the hour.

## Not built yet

Phase B (informed vs blind) is designed in the parent README but not
implemented. It should read the attack set chosen by Phase A rather than
hardcoding one — that selection is Phase A's entire purpose.
