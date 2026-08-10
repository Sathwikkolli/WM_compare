# Alignment bake-off — analysis

30 Emilia clips x 76 attack configs x 6 methods = **2,280 rows per method**,
13,680 total. All 30 Slurm tasks completed. `opus/496k` skipped everywhere
(libopus caps at 256 kbps), hence 76 configs not 77.

---

## 1. The `ALL` column is nearly useless — read families instead

Family sizes are lopsided, and 73% of rows are families where every method
scores ~100%:

| family | configs | rows | what it is |
|---|---|---|---|
| `null` | 39 | 1,170 (51%) | offset really is 0 |
| `codec` | 17 | 510 (22%) | small unknown encoder delay |
| `multiseg` | 9 | 270 (12%) | splice / insert |
| `warp` | 7 | 210 (9%) | time_stretch, time_jitter |
| `shift` | 4 | 120 (5%) | crop_head |

So the `ALL` hit rates (88.1–90.4% across the top five) are mostly measuring
*"can you return 0 when nothing moved"*. The real separation is in `multiseg`
and `warp`. **Do not rank on `ALL`.**

---

## 2. The false-shift gate disqualifies exactly one method

Pre-registered rule: false-shift > 5% disqualifies, regardless of hit rate.

| method | false-shift rate |
|---|---|
| `gcc_phat` | **0.00%** |
| `aof` | **0.00%** |
| `audalign_spec` | **0.00%** |
| `audalign_corr` | 0.09% |
| `audalign_fp` | 0.77% |
| `dtw_subseq` | **22.82%** ← DISQUALIFIED |

`dtw_subseq` invents an offset on nearly a quarter of audio that never moved,
with a p95 error of **6,321 ms** on the `null` family. This is visible as the
vertical stripe at `true = 0` in `scatter.png`, spanning −12 s to +14 s.

Subsequence DTW is free to slide the query anywhere in the reference, and with
no penalty for doing so it happily finds a "better" path several seconds away.
It is out, and its otherwise-excellent confidence AUC (0.988) does not save it.

---

## 3. `shift` (crop_head) — solved, but only two methods are exact

| method | median err | hit@1ms | hit@20ms | hit@50ms |
|---|---|---|---|---|
| `gcc_phat` | **0.000 ms** | **100%** | 100% | 100% |
| `audalign_corr` | **0.000 ms** | **100%** | 100% | 100% |
| `aof` | 1.812 ms | 35% | 100% | 100% |
| `dtw_subseq` | 6.5 ms | 17% | 100% | 100% |
| `audalign_fp` | 10.08 ms | 12% | 89% | 100% |
| `audalign_spec` | 10.08 ms | 12% | 91% | 100% |

Everything clears 50 ms. But `gcc_phat` and `audalign_corr` are **sample-exact**
(median 0.000 ms, 100% at the 1 ms bar) while the fingerprint and spectrogram
recognizers sit at their ~10–25 ms resolution floor, exactly as calibration
predicted.

Crop is the attack that motivated this whole project, and it is comprehensively
solved.

---

## 4. `multiseg` — where methods actually differ, and the aggregate lies

| method | hit@50ms | p95 err |
|---|---|---|
| `audalign_corr` | 73.7% | 3,156 ms |
| `audalign_spec` | 73.3% | 3,109 ms |
| `audalign_fp` | 68.2% | 3,644 ms |
| `aof` | 67.8% | 3,644 ms |
| `gcc_phat` | 53.7% | 3,000 ms |
| `dtw_subseq` | 3.0% | 1,962 ms |

But the aggregate hides **two opposite stories**. From `heatmap.png`:

| attack | gcc_phat | audalign_corr | audalign_spec | audalign_fp | aof | dtw |
|---|---|---|---|---|---|---|
| `splice_cut` | **41%** | 28% | 27% | 11% | 10% | 0% |
| `insert_foreign` | 60% | **97%** | **97%** | **97%** | **97%** | 3% |
| `insert_silence` | 60% | **97%** | **97%** | **97%** | **97%** | 1% |

`gcc_phat` is the **best** method on true splices and the **worst** on
insertions. The two cancel in the aggregate, which is why `gcc_phat` looks
mediocre at `multiseg` while actually being the strongest on the harder case.

Likely mechanism: on `insert_silence`/`insert_foreign` the first half is
untouched (true offset 0) and the second half is shifted. PHAT whitening splits
correlation energy between the two candidate lags, so `gcc_phat` lands on the
wrong one ~40% of the time. Plain correlation keeps the amplitude information
that makes the dominant (unshifted) half win.

**Caveat that applies to this whole section:** segment F1 is still not
implemented, so `multiseg` is scored on *primary offset* — the offset of the
longest segment. For a 1-cut splice the two halves are near-equal, so this is
partly a coin flip. Read these numbers as "found a plausible piece", not
"mapped the file". The p95 values near 3,000 ms confirm that when methods miss
here, they miss by whole segments.

---

## 5. `warp` — a universal failure, and the number is a giveaway

Every non-DTW method scores **exactly 0.2857** on `warp`. That is `2/7` — the
two `time_jitter` configs, and none of the five `time_stretch` configs.

**No method aligns time-stretched audio. Not one, at any strength.**

`dtw_subseq` reaches 49% (29% on `time_stretch` at 20 ms), which is the one
place its warping machinery earns anything — but it is already disqualified on
false-shift.

`time_jitter` is passed by everyone because vox's jitter is zero-mean, so the
true offset stays 0; it is effectively a `null` test in disguise.

If time-stretch matters, **none of these six tools solve it**. That needs a
resample-factor grid search (estimate the speed factor, resample, then align),
which is a different piece of work.

---

## 6. `null` and `codec` — solved

`null`: `aof`, `audalign_spec`, `gcc_phat` all 100%; `audalign_corr` 99.9%;
`audalign_fp` 98.2%; `dtw_subseq` 77.3%.

`codec`: everything 100% except `audalign_fp` at 97.5%. mp3/aac/opus/encodec
delays are all comfortably inside the 100 ms bound. `encodec` — a neural codec
that fully resynthesises the waveform — did not trouble any method except
`audalign_fp`, which drops to 83% at 1.5 kbps and recovers to 100% by 24 kbps.

`inverse_polarity` (the trap): passed 100% by every method. Using `argmax(|cc|)`
in `gcc_phat` was the right call, and the third-party tools handle sign too.

---

## 7. Confidence — two methods have confidence you must not trust

| method | conf AUC |
|---|---|
| `dtw_subseq` | 0.988 (disqualified) |
| `gcc_phat` | **0.888** |
| `aof` | 0.772 |
| `audalign_fp` | 0.749 |
| `audalign_corr` | 0.646 |
| `audalign_spec` | **0.500** |

`audalign_spec` at exactly 0.500 means its confidence carries **zero
information** — it is constant. It does not even appear in `reliability.png`
because every prediction lands in one bin.

Worse, `reliability.png` shows `audalign_corr` is **anti-calibrated at the top
end**: 100% observed accuracy in the 0.35–0.75 confidence bins, then collapsing
to 13% at 0.85 and **0% at 0.95**. Its highest-confidence predictions are its
*wrongest*. AUC 0.646 masks this because the inversion is confined to the top
bins. Any fallback logic keyed on `audalign_corr`'s confidence would fire
exactly backwards.

`gcc_phat`'s peak-to-sidelobe ratio is the best-behaved score of the survivors
and rises monotonically with accuracy.

---

## 8. Runtime

| method | median s/pair | relative |
|---|---|---|
| `dtw_subseq` | 0.038 | 1.0x |
| `gcc_phat` | 0.061 | 1.6x |
| `aof` | 0.109 | 2.9x |
| `audalign_corr` | 0.509 | 13x |
| `audalign_spec` | 0.668 | 18x |
| `audalign_fp` | 1.229 | **32x** |

The audalign recognizers pay a heavy price for writing temp WAVs and
re-fingerprinting on every call. Nobody is vetoed on runtime at this scale, but
`gcc_phat` being 20x faster than `audalign_fp` matters if this ever runs
per-request in the service.

---

## Decision

Applying the pre-registered order — false-shift gate, then hit rate, then
confidence AUC, then runtime veto:

### Primary: `gcc_phat`

- 0.00% false-shift, 100% on `null`
- **Sample-exact on crop** (median 0.000 ms, 100% @ 1 ms) — the attack that
  motivated this work
- **Best on `splice_cut`** (41%, next best 28%)
- **Best confidence of any survivor** (AUC 0.888, monotonic), so it can be
  trusted to flag its own failures and hand off
- 2nd fastest, 20x faster than `audalign_fp`
- ~15 lines of scipy, no dependency

### Partner for insertions: `audalign_corr`

`gcc_phat`'s one real weakness is mid-file insertion (60% vs 97%). Run
`audalign_corr` when `gcc_phat` reports low confidence and the length delta
suggests inserted material.

**But do not use `audalign_corr`'s confidence to make that decision** — it is
inverted at the top end (§7). Gate on `gcc_phat`'s PSR and on the file-length
difference instead.

### Not recommended

- `dtw_subseq` — disqualified, 22.8% false-shift
- `audalign_fp` — 32x the cost of `gcc_phat`, no family where it leads, and the
  multi-segment capability that justified including it is not measurable until
  segment F1 exists
- `audalign_spec` — highest `ALL` score (90.4%) but that is the null family
  talking; confidence is pure noise (AUC 0.500)

---

## Open items

1. **Segment F1 is still unimplemented.** All `multiseg` conclusions rest on a
   primary-offset proxy. `audalign_fp` may look better once its `candidates`
   list is turned into real segment maps.
2. **time_stretch is unsolved by every method.** Needs a resample-factor grid
   search if it matters for the threat model.
3. **No clip exceeds 27.8 s.** The 3-minute regime is untested, and correlation
   methods scale differently with length.
4. **Emilia is speech only.** The repetitive-music case — where several
   correlation peaks tie — is untested, and the Metapyxl client chain includes
   a music bed.
5. **`gcc_phat`'s insertion weakness has a plausible fix**: search the top-N
   correlation peaks rather than the single argmax, and prefer the one nearest
   zero lag. Cheap to test, might remove the need for a second method entirely.
