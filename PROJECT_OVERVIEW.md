# Audio Watermarking — Project Overview

*A single front-door summary of the whole project: what it is, the three systems
studied, what was measured, the key results, and the deployable system built on
top. Written to be read in ~10 minutes and to serve as a NotebookLM source.*

---

## 1. What this project is (one paragraph)

Audio watermarking hides an inaudible, machine-readable signal inside a sound file
so it can later be identified — useful for provenance, tracing AI-generated audio,
and content authentication. This project **compares three state-of-the-art audio
watermarking systems**, measures how well each survives real-world "attacks"
(compression, noise, cropping, pitch/tempo changes, etc.), studies what happens
when **all three are stacked into one file**, and then **packages them into a
deployable service** that embeds a *per-user ID* and reads it back — so a web app
can watermark audio and trace it to a specific user.

---

## 2. The three systems (and their papers)

| System | Origin / paper | How it works (plain terms) | Payload |
|---|---|---|---|
| **AudioSeal** | Meta AI (San Roman et al., 2024) | A trained neural encoder adds the mark in one fast pass; a neural detector reads it. | 16 bits |
| **AWARE** | DeepMark (Pavlović et al., 2026) — *paper included* | No trained encoder. For each file it runs an **adversarial optimization** (~400 steps) in the time–frequency domain to plant the mark; a desync-robust detector reads it. | 20 bits |
| **Timbre** | TimbreWatermarking (Liu et al., 2023) — *paper included* | A trained neural encoder embeds in the spectrogram; robust decoder reads it back. | 10 bits |

*(AudioSeal is the well-known baseline; AWARE and Timbre are the two research
papers to read in depth.)*

---

## 3. What was done — research (three experiments)

**A. Solo robustness benchmark.** Each watermark embedded alone, then hit with
~25 attacks (MP3/AAC/Opus compression, additive noise, resampling, low/high-pass
filtering, cropping, pitch/tempo shift, echo, denoise). For each we recorded
detection confidence, **bit-accuracy** (fraction of bits recovered correctly), and
audio quality (PESQ). Detection threshold: `bit_acc ≥ 0.8`.

**B. Cascade study.** All three watermarks embedded into one file, across **all 15
orderings** (3 singles + 6 pairs + 6 triples). Measured (1) *interference* — how
much stacking one mark damages the others, (2) *robustness* of the stacked file
under 23 attacks, and (3) *audio quality* (PESQ/SNR/SI-SNR/STOI) at each stack
depth.

**C. VoxWatermark no-box attack sweep.** 17 attack families at swept strengths
(from the VoxWatermark benchmark) run against each model solo and combined,
producing robustness-vs-strength curves.

---

## 4. Key findings (the numbers that matter)

**Solo robustness — who survives what** (bit-accuracy, 1.0 = perfect):

| Attack | AudioSeal | AWARE | Timbre |
|---|---|---|---|
| MP3/AAC/Opus, resample, filters | 1.0 | 1.0 | 1.0 |
| Noise (20 dB) | 1.0 | 1.0 | 1.0 |
| Noise (5 dB, harsh) | 0.69 | 0.95 | 0.80 |
| Tempo change ±10% | 0.63 | 1.0 | 1.0 |
| Denoise filter | 0.56 | 1.0 | 1.0 |
| Cropping (3–10 s) | ~0.94 | 0.6–0.75 | 1.0 |
| Pitch shift up | 0.81 | 0.6 | 0.2 |

**Takeaways:**
- **Timbre is the most robust overall** — near-perfect on almost every attack,
  including cropping and tempo, where the others struggle.
- **AWARE excels at noise, tempo, and denoising** (its design goal — robustness
  without attack-simulation training) but is **weakest under cropping**.
- **AudioSeal is solid on compression** but degrades under tempo, denoise, and
  harsh noise.
- **Pitch-shifting breaks all three** — the common failure mode.

**Cascade — stacking all three works, but costs audio quality:**
- *Interference is minimal*: after embedding all three, each watermark is still
  recovered at **bit-accuracy 1.0** on the clean file — they don't erase each other.
- *Quality drops with depth*: a single mark keeps PESQ high (AudioSeal 4.18,
  AWARE 4.30, Timbre 3.49). Stacking all three lowers PESQ to ~3.1–3.3 and SNR to
  ~10 dB. **AWARE embedding is the most audible** (drops SNR the most), so ordering
  affects final quality.

**Vox sweep:** confirms the same picture across swept attack strengths — see the
interactive report (`vox_out/vox_report.html`).

---

## 5. What was done — deployment (the usable system)

Beyond benchmarking, the three models were wrapped into **one uniform service** so
a web app can use them without touching each model's internals:

- **Two functions:** `embed(audio, user_id, models)` and `detect(audio)`.
- **Per-user watermarking:** a user ID (0–1023) is turned into the three models'
  bit-messages and embedded — the *same* ID in all three for redundancy. Any one
  detector can recover it. Capacity is capped at 1024 users by Timbre's 10 bits.
- **Pick one, two, or all three** models per request (all three = cascade).
- **Verified end-to-end:** embedding user 42 and detecting it returns the correct
  ID from all three watermarks (3/3 agreement), on both **Python 3.10 and 3.12.10**.
- **Handoff-ready:** full setup, model-acquisition, and API contract in
  `HANDOFF.md`; delivered to the deployment team.

**Operational note:** AudioSeal and Timbre embed instantly (one pass); **AWARE is
the slow step** (per-file 400-iteration optimization) — so embedding should run
asynchronously / on GPU in production.

---

## 6. Where everything lives

| Item | Location |
|---|---|
| **Code + all results** | GitHub: https://github.com/Sathwikkolli/WM_compare |
| Deployment guide | `HANDOFF.md` |
| This overview | `PROJECT_OVERVIEW.md` |
| Solo benchmark results | `bench_*.csv`, `extra_*.csv` |
| Cascade study | `cascade/README.md`, `cascade_out/*.csv`, `cascade_out/report.html` |
| Vox attack sweep | `cascade/README_vox.md`, `vox_out/vox_report.html` |
| Service API | `cascade/service.py`, `cascade/user_key.py`, `cascade/cascade_lib.py` |
| Papers (AWARE, Timbre) | Drive folder (see below) |
| Example audio (clean + watermarked + attacked) | Drive folder / `audio/` |

---

## 7. Glossary (for quick reference)

- **Watermark:** hidden, inaudible data embedded in audio.
- **Bit-accuracy:** fraction of embedded bits read back correctly. ≥ 0.8 = detected.
- **PESQ / STOI:** standard audio quality / speech-intelligibility scores (higher =
  better, PESQ max ~4.5).
- **SNR:** signal-to-noise ratio in dB (higher = the watermark is less audible).
- **Attack:** any processing of the audio (compression, noise, cropping…) that a
  watermark must survive.
- **Cascade:** embedding multiple watermarks into one file.
- **Payload:** how many bits a watermark can carry (AudioSeal 16, AWARE 20, Timbre 10).
