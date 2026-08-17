# Detector null test — clean audio through AudioSeal and AWARE

**Date:** 2026-08-14 · **Status:** complete · **Job:** Great Lakes, `standard`,
8 CPU, ~1 h 23 m wall · **Code:** `cascade/null_test.py` @ `fb86c8d`

## Why this was run

A production watermark detection request (id 1374, "without watermark") returned
`watermark_detected = true` on a file that carried no watermark. The cause was a
presence test that compared decoded bits against a key rebuilt *from those same
bits* — always a match, so every input looked watermarked. That has been
replaced by a two-stage rule: presence from detector confidence, identity from a
registry lookup.

The replacement rests on a threshold, and the threshold had never been measured.
It had been set three times by three people:

| Value | Origin | Outcome |
|---|---|---|
| `BIT_ACC_PRESENT = 0.8` | picked as "reasonable" | ~100% false positives |
| `0.0007–0.0021` | in `config.py` | far below any real score |
| `0.5` | AudioSeal's documented default; AWARE's sigmoid midpoint (`x0=0.041`) | **unverified for this cascade** |

Two incompatible observations existed, both n≈1. This run measures the actual
distribution so the next threshold decision is evidence, not another guess.

## Method

Score clean audio with the two models that have a real presence detector. No
embedding, no attacks, no database, no API — one forward pass per model per file.

**Excluded: Timbre.** It has no detector. `TimbreAdapter.detect` returns
bit-accuracy against `self.truth`, which is left over from whatever the process
last embedded, so the number is neither a presence test nor deterministic. This
also means the run needs no Timbre repo and no checkpoint.

### Inputs — 312 files, grouped by provenance

| Group | n | Source | Provenance |
|---|---|---|---|
| `emilia` | 300 | `emilia_curated.csv`, ≥5 s, seed 0 | source corpus, never watermarked |
| `osr_speech` | 6 | [Open Speech Repository](https://www.voiptroubleshooter.com/open_speech/american.html), 3F/3M, 8 kHz | public domain, published for reuse |
| `synthetic` | 3 | silence, white noise, 440 Hz tone | clean by construction |
| `non_speech` | 1 | Windows system audio, 1.3 s | clean |
| `uncertain` | 2 | mp3s labelled "original" in this repo | **not proven** — reported separately |

Provenance is tracked because the whole experiment depends on the inputs being
genuinely unmarked. The two files that cannot be proven clean are grouped apart
so they can never quietly contaminate the conclusion.

### Pass/fail, committed before the data was seen

| Outcome | Condition | Action |
|---|---|---|
| PASS | 0 detected **and** max ≤ 0.35 | keep 0.5, document the margin |
| MARGINAL | 0 detected, max in 0.35–0.50 | raise to 0.65 — passing without headroom is luck |
| FAIL | any detected | change same day, notify before more customer traffic |
| INVALID | HTTP/decode errors, or passes disagree | fix first; an unstable detector cannot be calibrated |

The margin criterion was written down in advance so that "zero false positives"
with a top score of 0.49 could not be read as success.

## Results

See [`summary.md`](summary.md). In short: **PASS on speech** (0/300, max 0.2723,
headroom +0.23), **keep 0.5**. One synthetic file failed — a 440 Hz tone at
AWARE 0.9679 — logged as a separate defect and not addressed here.

## Files

```
data/null_emilia.csv     300 Emilia clips, per-model confidence
data/null_curated.csv    12 curated files, scored twice (stability check)
params.json              reproduction config
summary.md               headline numbers and what they mean
```

## Reproduce

```bash
cd ~/wm_compare && sbatch cascade/null_test.sbatch
```

Or directly:

```bash
conda activate wmcompare
export WM_COMPARE_BASE=$HOME/wm_compare TORCHDYNAMO_DISABLE=1
python cascade/null_test.py --dir ../audio/clean_set --repeat 2
python cascade/null_test.py --emilia 300
```

## Open items

1. **Positive control.** A null test alone can be passed by a detector that never
   fires. Embed, then detect at 0.5, and confirm real watermarks are still found.
   Until that runs, this result is only half the picture.
2. **Tonal failure.** Map the boundary — sine sweep in and out of the 1000–4000 Hz
   band, DTMF, hold music, single instruments.
3. **Non-speech coverage.** One 1.3 s music file is not enough to say anything
   about music, and customers upload music.
4. **Production distribution.** This is a lab sample. Logging per-model scores on
   live traffic would give the real distribution across real customer audio.
