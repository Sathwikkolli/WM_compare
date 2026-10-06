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

## Status

planned — code pushed, not yet run.
