# Phase A-1 — the music sweep: locate the quality/detection crossing

**Status:** planned, not run. Predictions registered below.
**Parent:** [README.md](README.md) — the informed-detection run.
**Code:** extends `real_attacks_experiment.py` (or a new `informed/music_sweep.py`)

---

## The question

`real_attacks_summary.csv` shows AWARE's confidence crossing 0.50 at about
**+4.9 dB** speech-to-music SNR, while bit accuracy is still 0.925. So we know
where the *watermark* dies.

We do not know where the *audio* dies. Without both numbers the crossing point
cannot be located, and the parent plan's central claim — that the music bed is a
vulnerability in usable audio — is unsupported.

> **At the SNR where detection fails, is the audio still usable?**

---

## Two problems with the existing evidence

**1. It is n = 1.** `real_attacks_experiment.py` uses one LibriVox speech clip
(Poe, 25 s) and one music track (*March For Honor*). There is no trial loop in
the code, despite `real_attacks_summary.csv` labelling rows "mean over trials".
The +4.9 dB figure rests on a single speech/music pair.

Masking depends heavily on what the speech and the music actually are, so a
single pair cannot establish a threshold. This must be fixed before the number
is used to justify anything.

**2. It measures no quality at all.** Confidence and bit accuracy only. The
script already writes each mix to disk (`{tag}_mix_{snr}dB.wav`), so quality can
be scored without regenerating audio — but nothing computes it today.

---

## Prerequisite: there is no DNSMOS scorer in this repo

Worth knowing before any code is written. `cascade/emilia_bench.py:98` uses
DNSMOS as a **manifest column** to filter source clips
(`df['dnsmos'] >= DNSMOS_MIN`, with `DNSMOS_MIN = 3.0`) — it reads a
precomputed value, it does not compute one. Nothing in the repo can score new
audio.

Two options:

| option | pro | con |
|---|---|---|
| **DNSMOS P.835** (ONNX) | scores are directly comparable to the Emilia manifest and to `emilia_bench`'s existing 3.0 filter | needs `onnxruntime` plus model files |
| **torchaudio SQUIM** | ships with torchaudio, already a dependency | different scale; not comparable to the manifest |

**Recommendation: DNSMOS**, specifically for the comparability. The parent plan's
usability floor of 3.0 is then *the same 3.0* that `emilia_bench` already uses to
select clips, which makes the floor defensible rather than arbitrary.

Install it into `wmcompare` the same way `align_bench` did — snapshot
`pip freeze` first, `--dry-run` before installing.

---

## Design

### Clips — 50, not 1

Reuse `cascade/emilia_bench.py`'s selection logic verbatim: Emilia only,
`dnsmos >= 3.0`, `duration_s >= 10`, speaker-stratified round-robin for voice
diversity, `SEED = 1234`, `N_CLIPS = 50`. Same 50 clips as the parent plan.

Reusing the existing selector rather than writing a new one keeps this
comparable to `emilia_bench` and removes a whole class of selection-bias
argument.

### Music — one canonical track, plus a diversity probe

Main sweep uses the existing *March For Honor* track, so results stay comparable
to `real_attacks_summary.csv`.

**But one track is a confound**: its particular spectrum may mask AWARE's bands
unusually well or unusually badly. So add a smaller probe — **10 clips × 3
additional tracks** of contrasting character (orchestral / electronic /
sparse-acoustic) — purely to check whether the threshold moves with the music.
If it does, the main sweep's number is track-specific and must be reported that
way.

### Grid

SNR: `20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 0, -3, -6, -10` dB.

Keeps the existing fine spacing near the crossing. Drops the `3.8` point, which
was chosen to bracket the n=1 result and would bias a 50-clip re-measurement.

### Arms

| arm | what it is | why |
|---|---|---|
| **watermarked + music** | the measurement | detection and quality |
| **unwatermarked + music** | control | separates damage caused by the music from damage caused by the watermark |

The control follows `2026-08-17_attack-damage-control`, which found the
watermark itself contributes only **0.06 PESQ** across 15 conditions. If that
holds here, the music is doing all the damage and the control can be dropped
from later phases.

### Measured per mix

| quantity | role |
|---|---|
| **DNSMOS** | primary quality — no-reference, "does this sound acceptable" |
| **PESQ** vs clean speech | secondary — fidelity to the original |
| detection confidence | at threshold 0.50 per `THRESHOLD_DECISION.md` |
| bit accuracy | is the watermark still readable even when undetected |

---

## Outputs

Following `results/README.md`:

- `data/music_sweep.csv` — one row per (clip, music, SNR, arm)
- `data/music_sweep_diversity.csv` — the 3-track probe
- `figures/crossing.png` — **the headline figure**: DNSMOS and detection
  confidence against SNR on one chart, with the 0.50 detection threshold and the
  3.0 DNSMOS floor drawn in. The gap between where the two lines cross their
  thresholds *is* the vulnerability window.
- `figures/metric_disagreement.png` — DNSMOS vs PESQ across the sweep

---

## Registered predictions

1. **A crossing exists.** DNSMOS stays above 3.0 at SNRs where detection has
   already failed, so there is a band of usable audio with no detectable
   watermark. If DNSMOS falls below 3.0 *before* detection fails, the music bed
   is not a vulnerability and the parent plan loses its lead candidate.
2. **PESQ and DNSMOS disagree sharply.** PESQ against clean speech sits below
   2.0 at every SNR where music is clearly audible, while DNSMOS stays above
   3.0. This is the empirical test of the parent plan's metric argument; if they
   agree, that argument is wrong and the screen can use PESQ.
3. **The arms are indistinguishable** — mean DNSMOS gap between watermarked and
   unwatermarked mixes below 0.1, consistent with damage-control's 0.06 PESQ.
4. **The n = 1 threshold does not survive.** The 50-clip mean crossing lands
   within ±3 dB of 4.9 dB, but the per-clip spread is several dB, because masking
   depends on the speech content. A single number will not describe it, and any
   later work quoting "4.9 dB" as *the* threshold is over-claiming.
5. **Bit accuracy stays high where confidence fails** — at least 0.85 at the
   detection crossing, confirming the watermark is present but unreadable *by
   the detector*, which is precisely the condition informed detection should fix.

---

## Why this is the right first task

It is small, it reuses existing audio generation and clip selection, and it is
decisive: prediction 1 either confirms the music bed as a genuine vulnerability
or removes it. Either outcome determines whether Phase B has a target worth
attacking.

Prediction 5 matters just as much — if the bits are already gone at the
crossing, informed detection has nothing to recover and the parent plan needs
rethinking before any code is written for it.

---

## Conclusions

*(empty until the run completes)*
