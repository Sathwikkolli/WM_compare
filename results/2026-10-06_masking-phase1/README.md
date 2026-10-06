# 2026-10-06 — masking-phase1: masking budget vs AWARE budget (E2–E6)

Phase 1 of masking-threshold-in-AWARE. Compares the **shape** of AWARE's budget
with a psychoacoustic masking threshold **without** AWARE's embedder.

**Why a noise stand-in is valid:** [masking-e0-e1](../2026-10-06_masking-e0-e1/)
showed AWARE pushes ~90 % of 1–4 kHz bins to exactly ±budget with ~50/50 signs.
So AWARE's watermark ≈ random signs × budget. Here: seeded random ±1 signs × a
budget, added to |X| with the phase kept — the same edit AWARE makes.

Code: `masking/psy.py` (model), `masking/run_phase1.py`, `masking/aggregate_phase1.py`,
`masking/phase1.sbatch`. Same 30 clips as E0/E1.

## Shapes

| name | budget per bin |
|---|---|
| `aware` | \|X\| · 10^(−6/20) — AWARE's rule (code default tolerance 6.0 dB) |
| `mask` | Johnston (1988) critical-band threshold: Schroeder spreading, per-band spectral-flatness tonality (tone 14.5+b dB, noise 5.5 dB, SFM −30 dB = fully tonal), post-masking 300 dB/s, pre-masking 20 dB/frame, Terhardt hearing floor; full-scale sine = 96 dB SPL |
| `flat` | constant (control: no shaping) |

## Experiments

| ID | Question | How |
|---|---|---|
| E2 | Where do the budgets differ? | per-bin ratio mask/aware; quiet vs loud frames; tonal vs noisy bands; share of bins where AWARE exceeds the threshold |
| E3 | Equal energy: which shape sounds better? | scale mask & flat to AWARE's total energy; PESQ / STOI |
| E4 | Equal quality: how much more energy does mask carry? | bisect mask gain until PESQ = PESQ(aware); report energy gain in dB |
| E5 | Is AWARE risky on tonal parts? | E3 restricted to the most tonal vs most noisy third of speech frames |
| E6 | Do codecs strip the mask shape more? | MP3 64/32, Opus 32/16: correlation and energy of the change before vs after the codec |

## Gate 2 (fixed before the run)

- **A (E3):** median PESQ(mask) − PESQ(aware) > 0 and Wilcoxon p < 0.05
- **B (E4):** median energy gain ≥ +1 dB at equal PESQ
- **C (E6):** median corr(mask_eq_quality) / corr(aware) ≥ 0.9 at mp3_64 and opus_32
- **PASS = (A or B) and C.** Fail → redesign the budget (mixes, E12) before E10.

## Predictions (written before the run)

From a model sanity check on 3 local speech clips (budgets only, no audio scoring):
1. E2: mask allows **more** room in ~60 % of bins (median +2 dB) but **~2 dB less
   total energy** — it takes from spectral peaks and gives to the valleys and
   neighbours of loud bins.
2. E3: small PESQ gain for mask (+0.05 to +0.2); flat clearly worst.
3. E4: +1 to +3 dB.
4. E5: AWARE exceeds the threshold more often in tonal bands; the mask advantage is
   larger in tonal frames.
5. E6: the main risk — energy moved into valleys is what codecs quantise away. Expect
   corr ratio 0.8–0.95, i.e. C is the gate most likely to fail.

## Results (30 clips) — **Gate 2: FAIL**

| | result | gate |
|---|---|---|
| E2 | mask allows more room in 75 % of bins (median +4.6 dB) but **3 dB less total energy**; **+8.6 dB in quiet frames**, +3.1 dB in loud frames; AWARE exceeds the threshold in 25 % of bins | — |
| E3 | equal energy: PESQ aware **4.13**, mask **3.85** (Δ −0.25, 0/30 clips better, p = 2e−9), flat 1.96 | A ✗ |
| E4 | at AWARE's PESQ the mask shape carries **−5.6 dB** energy (all 30 clips negative, range −11 to −2 dB); it must sit 2.3 dB *below* its own modelled threshold | B ✗ |
| E5 | mask worse in both frame types: noisy Δ −0.04, tonal Δ −0.16 | — |
| E6 | at **equal energy** the mask shape survives codecs almost as well (corr mp3_64 0.978 vs 0.983, opus_32 0.871 vs 0.904). At equal quality it survives worse (opus_32 ratio 0.77) — a consequence of E4's lower energy, not of codecs targeting the shape | C ✗ |

Predictions 1 (direction of E2) and 5 (codec risk, partly) held; 2, 3, 4 refuted.

## Why it failed (from the budget maps, `figures/e2_budget_maps.png`)

1. **Room in silence.** The mask budget is +15–20 dB above AWARE's in pauses
   (start and end of the clip). That comes from the hearing floor (96 dB SPL
   calibration) and the 300 dB/s post-masking tail. PESQ — and listeners at a
   normal level — hear noise in pauses. AWARE's rule puts ~nothing there.
2. **Smearing between harmonics.** The threshold is computed per critical band, so
   it is flat across a band and ignores the harmonic peaks and valleys of voiced
   speech. Energy moves from the peaks (where AWARE hides it under the speech
   itself) into the valleys (where it is exposed).
3. **Tonality detection does not fire.** Only 0.1 % of 1–4 kHz bins are classed
   tonal, so almost every band gets the generous noise offset (5.5 dB) instead of
   the tonal one (14.5 + b dB) — voiced speech is treated as noise.
4. **AWARE's own rule is a strong baseline.** Per-bin self-masking (change ∝ the
   bin's own magnitude) hides energy exactly under the spectral peaks; the flat
   control (−2.2 PESQ) shows how much shaping matters.

## Caveats

- The only quality judge is PESQ (no ODG / listening test). PESQ is a speech-quality
  model, not a masking-exact one, and it punishes noise in low-level frames hard.
- This is one masking model with one parameter set. The failure is of *this model
  as a drop-in replacement*, not of masking in general.

## Next (proposed)

Phase 2 (E7 ablation + E9 calibration), same noise framework, to find which part
hurts: switch off the hearing floor / temporal tails; keep the per-bin peak structure
(e.g. threshold = min(band threshold, per-bin self-masking), or MPEG model 1 tonal
maskers per bin); test calibration 86/96/106 dB. And E12 mixes:
`min(mask, aware)` (only remove AWARE's above-threshold 25 %) and `aware` boosted
where the mask allows more, capped at the mask.

## Status

complete — Gate 2 failed; redesign before touching AWARE.
