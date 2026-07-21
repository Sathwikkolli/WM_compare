# Audio Watermarking — Capability Brief

**For:** FakeXpose marketing team
**From:** Sathwik Kolli
**Date:** 15 July 2026

---

## 1. What it does

The system hides an **inaudible per-user identifier** inside an audio file, and reads that identifier back later — even after the audio has been compressed, re-encoded, re-recorded, or edited.

The practical use: if a piece of audio leaks or is misused, the embedded ID says **which user's account it came from**.

Two operations:

- **Embed** — take an audio file plus a user ID, produce an audio file that sounds identical but carries the hidden ID.
- **Detect** — take an audio file, return the user ID found inside it.

The ID is written **three separate times using three independent watermarking methods**. If processing destroys one, the other two still carry it. This redundancy is the core of the design.

---

## 2. Specifications

| Property | Value |
|---|---|
| Watermarks embedded per file | 3 independent, all carrying the same ID |
| Audio output | 22.05 kHz, 16-bit PCM WAV |
| Audio input | Any common format/rate (converted internally) |
| Detect speed | Fast — roughly one model pass per watermark |
| Embed speed | Two methods near-instant; one takes seconds-to-minutes per file (see §5) |
| Audible difference | None intended — the mark sits below the hearing threshold |

---

## 3. What it survives

Tested against 17 families of audio distortion at multiple strengths, plus real-world conditions (music beds, reverb, stereo conversion).

**Holds up under:**

- **Compression / re-encoding** — MP3, AAC, Opus across the bitrate ranges tested
- **Resampling** and sample-rate conversion
- **Volume changes**
- **Added noise** — both synthetic and real background noise
- **Echo and reverb**
- **Dynamic range compression / expansion** (broadcast-style processing)
- **Low-pass filtering, smoothing, time jitter, phase shift**
- **Polarity inversion, time-stretching, quantization** — these each defeat *one* of the three watermarks, but the other two recover the ID

**The headline:** across the distortion families tested, **no single one defeated all three watermarks at once**. That is exactly what the three-way redundancy is for — each method's weakness is covered by the other two.

**Known weakness — pitch-shifting.** Aggressive pitch-shifting degrades all three watermarks. If someone deliberately pitch-shifts audio to strip the mark, it can work. This is a known limitation across the field, not specific to this system.

---

## 4. Published benchmark results

The figures below are the results **each model's authors published in their own peer-reviewed papers**. They are the citable numbers, with a source anyone can check.

### AudioSeal — Meta AI, ICML 2024

| Metric | Reported |
|---|---|
| Detection AUC, averaged over 15 edit types | **0.97** (vs 0.84 for WavMark, the prior state of the art) |
| Perfect detection (AUC 1.00) under | MP3, AAC, resampling, echo, pink noise, bandpass, volume boost/duck |
| Audio quality — PESQ | **4.47** |
| Audio quality — STOI | **0.997** |
| Audio quality — ViSQOL | **4.83** |
| Detection speed | **3.25 ms** per sample — **485× faster** than WavMark |

### AWARE — 2025

| Metric | Reported (bit error rate, lower is better) |
|---|---|
| Low-pass filter, high-pass filter, resampling | **0.00%** |
| MP3 @ 64 kbps | **0.71%** |
| Pitch shift | **0.92%** |
| PCM 8-bit requantization | **1.43%** |
| Pink noise | **1.61%** |
| Neural vocoder re-synthesis | **1.61%** |
| Audio quality — PESQ | **4.08** |
| Audio quality — STOI | **0.97** |

### Timbre Watermarking — NDSS 2024

| Metric | Reported |
|---|---|
| Detection accuracy vs. voice-cloning attacks (Tacotron2, FastSpeech2, VITS) | **99.88% – 100%** |
| Design | Frequency-domain embedding with repeated-embedding strategy for robustness |

**Sources:**
- AudioSeal — *Proactive Detection of Voice Cloning with Localized Watermarking*, ICML 2024 — arxiv.org/abs/2401.17264
- AWARE — *Audio Watermarking with Adversarial Resistance to Edits* — arxiv.org/abs/2510.17512
- Timbre — *Detecting Voice Cloning Attacks via Timbre Watermarking*, NDSS 2024 — arxiv.org/abs/2312.03410

**One framing note that matters.** These are each model's results measured **standalone**, by its own authors. Our system runs all three **stacked in one file**, which trades some audio quality for redundancy — three marks in one waveform measures lower on quality than any single mark alone. So these figures are best presented as *"the published results of the methods we build on,"* not as measurements of our combined system. Corpus-level numbers for our own stacked configuration are in progress.

---

## 5. Deployment note — embedding is not instant

Two of the three methods embed in milliseconds. The third works differently: it runs a per-file optimization loop, taking **seconds to minutes per file** on CPU.

For product framing:

- **Embedding is a background job, not a live request.** Users should see "processing," not a blocked spinner.
- **A GPU speeds this up substantially** if embed latency becomes a bottleneck.
- **A faster two-watermark mode exists** — near-instant, still gives two independent marks carrying the same ID. A real product option if latency matters more than maximum redundancy.

Detection is fast in all configurations. The cost sits in embedding, not detection — which is the right way round, since files are embedded once and checked many times.

---

## 6. What the models are

The three watermarking methods are **published third-party research systems**:

- **AudioSeal** — Meta AI
- **AWARE** — deepmarkpy
- **TimbreWatermarking** — the TimbreWatermarking research group

**Our engineering contribution:** the unified embed/detect layer, the per-user ID keying scheme, the three-way redundancy design, and the benchmarking that establishes how the three behave together and where they break.

Worth being precise about this in copy — the models are public research systems, and describing them as in-house would be checkable in a minute by anyone in the field. The integration and attribution layer is genuinely ours; the watermark models are not.

---

## 7. Safe to say

- "Embeds an inaudible, per-user identifier into audio and recovers it later."
- "Three independent watermarks carry the same ID — if one is destroyed, the others recover it."
- "Survives common real-world audio processing, including MP3/AAC/Opus compression, resampling, volume changes, background noise, and echo."
- "Built on published, peer-reviewed watermarking research, integrated behind a single API with per-user attribution."
- "The underlying methods report detection AUC of 0.97 across 15 edit types, and sub-2% bit error under compression, filtering, noise, and pitch shift." *(cite the papers — §4)*

---

Happy to walk anyone through any of this, or to review copy before it goes out.
</content>
