# Phase B — does informed detection extend robustness?

**Status:** planned, not run. Predictions registered below.
**Code:** `informed/` (Phase B files not yet written)
**Supersedes:** the Phase B sketch in `README.md`, which gated on a quality floor.

---

## The question, and why it needs no quality threshold

> For each attack, at what strength does blind detection fail, and at what
> strength does **informed** detection fail? The gap is the benefit.

Both detectors see the **identical attacked file**. Same music, same bitrate,
same everything — only the detector changes, and only in whether it is handed
the unwatermarked original.

So audio quality is **constant across the two arms**. It cannot influence which
one wins, and including it in the core claim imports a threshold argument that
has already cost this project several days and produced one refuted
justification (see README, usability floor).

Quality is still measured and reported per condition. It is not a gate.

**What this design gives up:** it measures whether informed detection extends
robustness, not whether the extension lands in a region anyone cares about. That
is a scope statement, not a defect, and it belongs in the write-up verbatim.

---

## Why the Phase A design was replaced

`summary_screen.md` classified 27 attacks against a usability floor. That floor
was the weakest part of the whole experiment:

- The first version (absolute DNSMOS 3.0) was refuted by measurement — clean
  Emilia clips score 2.86–3.36, so it failed a quarter of *undamaged* audio.
- The replacement (relative drop ≤ 0.5 MOS) was a convention with no application
  justification, on a metric (DNSMOS P.835) built to score *noise suppression*,
  whose `bak` term penalises a deliberate music bed as intrusive noise.
- Listening confirmed the mismatch: audio the floor called unusable sounded fine.

The literature does not solve this either. Boato et al. (IEEE TIFS 2009) needs a
quality threshold too and states plainly it is *"chosen dependently on the
application scenario"* — then **sweeps it and reports several values**, with the
verdict flipping between 40 dB and 35 dB on the same data.

Phase A's screen is retained as a **supporting result**: it establishes which
attacks defeat blind detection at all, and its detection numbers are unaffected
by the floor argument. Only its verdicts depend on the floor.

---

## Design

### The measurement, per attack

1. **Bracket.** Verify at the weakest setting that **both** detectors succeed,
   and at the strongest that **both** fail. If informed still survives at the
   strongest setting the attack is not bracketed, and the result is reported as
   *"gain ≥ X"* rather than a number. Bisection is invalid without this and the
   check is cheap.
2. **Bisect** the strength parameter, 10 iterations (~0.1% precision), separately
   for each detector and **each clip**.
3. **Gain** = blind crossing − informed crossing, in that attack's own units.

Strength axes, and which direction is stronger:

| attack | knob | stronger |
|---|---|---|
| `music_bed`, `gaussian_noise`, `noise_babble/factory/machinegun` | SNR dB | lower |
| `mp3`, `aac`, `opus`, `platform_reencode` | bitrate | lower |
| `resample_roundtrip` | target rate | lower |
| `highpass`, `lowpass` | cutoff ratio | higher |
| `smooth` | window | higher |
| `reverb` | RT60 | higher |
| `echo` | decay | higher |
| `quantization` | levels | lower |
| `stereo_widen` | delay ms | higher |
| `denoise` | nr dB | higher |
| `dynamic_compression` / `dynamic_expansion` | ratio | higher |
| `volume_down` / `volume_up` | gain | away from 1.0 — **split in two** |

**Excluded.** `encodec` and `mastering_chain` have only 3–5 discrete settings and
cannot be bisected; they are reported at step resolution. `inverse_polarity`,
`phase_shift` and `time_jitter` have no strength axis and serve as harness
controls. `time_stretch` is excluded entirely — the bake-off found **no aligner
handles it**, so subtraction is impossible regardless of the result.

### The informed detector

```
1. align      gcc_phat(org, attacked)  -> offset, including the fractional part
2. shift      resample attacked to the sub-sample offset
3. gain match a = <attacked, org> / <org, org>
4. residual   w_obs   = attacked - a * org
5. reference  w_clean = wm - org            (exact: we made the file)
6. statistic  windowed normalised correlation of w_obs against w_clean
```

**`gcc_phat`, not `aof`.** Subtraction needs sample-exact alignment.
`2026-08-18_frame-align-null` §5 measured `aof` as quantised to a ~7.7 ms grid
(~123 samples) — at that error the host does not cancel, it smears and doubles.
`gcc_phat` is sample-exact whenever it succeeds. That finding is what makes this
experiment possible.

Rows where `gcc_phat` does not report a trusted alignment are **excluded and
counted**, never scored as detection failures.

**Why correlation.** Under Neyman–Pearson the optimal detector is the likelihood
ratio test, and for an additive watermark in Gaussian noise the LRT reduces
exactly to a correlation detector (matched filter). Normalised correlation is
the standard practical form: bounded in [−1, 1] and immune to the level changes
many attacks introduce.

**Why not reconstruct and reuse AWARE's detector.** Tempting, because it would
put both arms on one scale. It is circular: for an additive attack
`attacked = wm + n`, so the residual is `w + n`, and adding it back to `org`
returns `wm + n` — the file we started with. **The gain lives in the statistic,
not in reconstruction.** Blind AWARE must find `w` inside `org + w + n` where the
host is the dominant interference; the correlation detector sees `w + n` with the
host removed entirely.

### Three decisions that could have flattered the result — registered here

**Windowed correlation is primary.** AWARE hops in 42 ms steps, so a global
correlation over 10 s dilutes a locally-surviving watermark. Windowed is more
sensitive and will very likely look better — which is exactly why it is fixed
**before** the run. Window = 42 ms (one AWARE hop), 50% overlap, aggregate = mean.
**Global correlation is computed and reported as secondary**, so the effect of
this choice is visible rather than hidden.

**Thresholds are per attack strength.** As an attack strengthens the residual
gets noisier and the null distribution moves, so a threshold calibrated on clean
nulls becomes wrong. Every point on every sweep gets its own null calibration.
This applies to the **blind arm equally** — the same objection would otherwise
undermine the baseline.

**Host removal is scalar gain, primary.** A short FIR filter (64 taps, 4 ms) is
implemented and reported as secondary, because a scalar cannot undo an EQ curve
and the filtering attacks would otherwise be scored against contaminated
residuals. Both remain "generic inverse parameters" — no signal is injected.

### False-positive matching — the load-bearing control

The two detectors output different quantities on different scales. AWARE gives a
confidence in [0,1]; informed gives a correlation. **Comparing "where each
crosses 0.50" is meaningless**, and informed could be made to look arbitrarily
good simply by being more willing to say yes.

So at every strength point, on unwatermarked audio put through the *identical*
pipeline:

- **blind null:** attacked clean speech → AWARE detect → confidence
- **informed null:** attacked clean speech → align, gain-match, subtract →
  correlate against that clip's `w_clean` (the watermark it *would* have carried)

Both thresholds are then set to the **same false-positive rate**. Primary **1%**,
with **0.1%** reported alongside.

`2026-08-18_frame-align-null` §3 measured the cost of skipping this: a
matched-data-only calibration admitted 30–41% of unrelated audio while every
matched-data metric looked healthy.

**Null size.** 50 clips give at best 2% resolution — not enough for a 1% rate.
The null arm therefore uses **200 clips**, drawn from the same Emilia pool and
disjoint from the 50 used for the positive arm. The null needs no embedding, so
it is the cheap half.

### Corpus and message

Same **50 clips**, 10 s, from `informed/clips.json` — the `emilia_bench`
selection, so results stay comparable across every run in this project.

**A random 20-bit message per clip**, seeded and recorded. `cascade_lib` embeds a
fixed `AWARE_BITS` for every clip, which is a confound: results could be specific
to one bit pattern. Randomising removes the objection at negligible cost.

### Statistical treatment

**Crossings are found per clip, then reported as a distribution** — never
averaged first. Clip variance is large: the Phase A music sweep found a **22 dB
range** across 50 clips (−5.19 to +16.84), so an averaged curve would hide the
thing most worth knowing.

Per-clip crossings also allow a **paired test**: for each clip, did informed beat
blind on that same clip? A paired Wilcoxon signed-rank test on 50 paired
differences is far stronger evidence than comparing two means, and it is the
right test because each clip is its own control.

---

## Registered predictions

Written before the run. Two of them can end the project.

1. **Informed shows a positive gain on additive attacks** — music bed, gaussian,
   babble, factory, machinegun. This is the central claim; if it fails, informed
   detection does not work and Phase C is cancelled.
2. **The music-bed gain is ≈ 5 dB.** At the blind crossing (~4.4 dB SNR) the
   speech carries ~2.5× the music's power, so removing it should drop total
   interference by ~5.5 dB. Predicted crossing shift: 4.4 dB → ≈ −1 dB.
3. **Codec attacks show ≈ zero gain.** `mp3`, `aac`, `opus`,
   `resample_roundtrip`, `platform_reencode` rebuild the waveform, so the
   additive assumption behind subtraction does not hold.
4. **Filtering attacks show zero or negative gain** with scalar host removal, and
   improve measurably with the FIR variant. `highpass` and `lowpass` delete a
   band — what is gone cannot be recovered — and a scalar cannot undo an EQ
   curve, so leftover host should contaminate the residual.
5. **The paired test is significant on additive attacks**: informed wins on
   ≥ 80% of individual clips, not merely on the mean.

**A stated possible outcome:** informed may perform **worse** than blind where
alignment or gain estimation is imperfect, because subtraction amplifies whatever
they get wrong. The design permits this answer and the summary bar chart is drawn
with a zero line so negative bars are visible rather than hidden.

---

## Figures

**1. Gain curve — `gain_music.png`.** Attack strength on x, detection score on y,
one line per detector, each detector's 1%-FPR threshold as a dotted horizontal,
and the band between the two crossings shaded. The shaded band *is* the result.
Drawn for `music_bed` as the explainer.

**2. Summary — `gain_summary.png`.** One horizontal bar per attack, gain in that
attack's own units, sorted, zero line marked. Positive = informed helps, negative
= informed hurts. This is the abstract figure. If it splits by category —
additive positive, codec at zero — the mechanism is confirmed visually.

**3. Score distributions — `null_separation.png`.** Per detector, overlapping
histograms of scores on watermarked and unwatermarked audio, with the 1%-FPR
threshold drawn in. This is the figure that makes the comparison credible; without
it the first question at a defence is *"how do we know that was fair?"*

**4. Optional — `residual_spectrograms.png`.** Clean watermark fingerprint,
residual after a mild attack, residual after a strong one, amplified. Intuition
only, not evidence. `cascade/make_fail_spectrograms.py` already computes
residuals, so it is nearly free.

---

## Known limitations

- **Strength is not comparable across attacks.** "5 dB of music" and "8 kbps MP3"
  share no units, so gains are reported per attack and never averaged across them.
- **Single attacks only.** Boato et al. search *combinations and ordering*, and
  found order alone worth >1 dB. A real attacker chains attacks. Our 27 may each
  look survivable while a chain of three defeats AWARE — untested here, and the
  most likely place a genuine vulnerability hides.
- **AWARE only.** AudioSeal and Timbre are Phase C, and only if Phase B works.
- **`time_stretch` cannot be evaluated at all** — no aligner handles it.
- **Informed detection needs the original**, so this measures a capability
  available only in verify-mode, not in open-set search.

---

## Validity: what may and may not be claimed

Full subtraction uses the original waveform in the detection statistic and does
**not** pass the litmus test registered in `2026-08-05_align-method-bakeoff`
(*"could this correction be applied if an oracle gave us the attack parameters,
without the audio?"*). That is the treatment, not an oversight — FakeXpose
verify-mode holds the original by construction.

- **Valid:** informed vs blind on the same files, same clips, same attacks,
  matched false-positive rates.
- **Not valid:** comparing these numbers to AudioMarkBench, VoxWatermark,
  SoK-2025 or any published blind benchmark. Different detector class.

Informed detection and host-interference cancellation are **classical, not
novel** (Cox et al.; *Selective Host-Interference Cancellation*, IEICE;
*Informed Detection Revisited*, Springer). The claim here is the **measurement on
a modern neural audio watermark**, where every published benchmark evaluates
blind. A systematic literature check remains an open item before any novelty
claim is made.

---

## Conclusions

*(empty until the run completes — do not fill this in from expectation)*
