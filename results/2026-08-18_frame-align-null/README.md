# Frame-level alignment: resolution limit and false-alarm floor

**Status:** predictions registered, run not yet executed.
**Code:** `frame_align/`
**Predecessor:** `results/2026-08-05_align-method-bakeoff/`

---

## The question

The bake-off aligned **whole clips** and picked `gcc_phat` (primary) with
`audalign_corr` as an insertion partner. Two things it never asked:

1. **How short can a piece of audio be and still be alignable?**
   AWARE(20bps) slides its detector in 42 ms steps. If alignment needs 500 ms of
   audio to work, that is a hard floor on how short a key-hop dwell can be before
   sync — not the watermark — becomes the bottleneck.

2. **What does the aligner do when the audio does not match at all?**
   The bake-off's `null` family is *audio that did not move*. That is not the same
   as *audio that does not belong*. A method can be 0.00% false-shift on the first
   and still return a confident offset for unrelated speech. Nothing in the
   existing data rules this out.

The deliverable is one number per method:

> at the confidence threshold τ where the false-accept rate on mismatched audio
> is ≤ 1%, what fraction of genuine frames still align within 50 ms?

That is the number that goes into the detector. Everything else here is support.

---

## Design

**Two experiments over the same frame draws.**

| | reference | truth | measures |
|---|---|---|---|
| **A — localization** | the frame's own clip | known exactly (we chose the cut point) | usable resolution vs frame length |
| **B — mismatched null** | a *different* clip | none exists | false-alarm rate, confidence null distribution |

**Methods:** `gcc_phat` and `aof`, imported from `align_bench/methods.py` — not
copied, so the two runs stay comparable.

`audalign_corr` is the designated production partner but is **deliberately not
used here.** The output of this experiment is a confidence threshold, and
`audalign_corr`'s confidence is anti-calibrated at the top end (bake-off §7: 100%
accurate in the 0.35–0.75 bins, 0% at 0.95). A threshold derived from an inverted
score is worse than no threshold. `aof` has a usable unbounded `standard_score`
and an independent-ish MFCC front-end.

**Frame lengths:** 50, 100, 250, 500, 1000, 2000 ms.
**Trials:** 20 random start positions per clip per length.
**Clips:** the same 30 Emilia clips as the bake-off (`align_bench/clips.json`).
The 5 held-out `foreign` clips stay held out — null partners come from the main
30 by deterministic rotation, so every clip pairs with a spread of partners and
never with itself.

**Starts are drawn uniformly, not snapped to the 42 ms AWARE hop grid.** Snapping
would let a method score by landing on a grid point by luck, and real desync is
not grid-aligned.

**Silence is stratified, not dropped.** Emilia is speech with pauses; a random
50 ms frame can land in one, where alignment is genuinely impossible and the
method is not at fault. `run_frames.py` records per-frame RMS as a column and
`score_frames.py` splits voiced / low-energy / silent. Dropping silent frames
would flatter the short lengths; pooling them would move the knee for the wrong
reason.

**Reference lengths:** native (~9–27.8 s), 60 s, 180 s. False-alarm rate should
scale with the number of candidate positions in the search space, and open item
#3 of the bake-off records that nothing above 27.8 s has ever been tested.

---

## Declared deviation: long references are synthetic

`align_bench/make_clips.py` refuses concatenation on purpose. Emilia clips are
~10 s, so the 60 s and 180 s conditions **cannot** be native. `refs.py` probes the
manifest for genuinely long clips first; if none exist, it builds padded
references by joining other clips around the home clip.

A concatenated reference is **not** a real 3-minute recording. Speaker and channel
change at every junction, which is not what a long single-source recording looks
like. Junctions use a 10 ms raised-cosine crossfade specifically so they do *not*
produce discontinuity clicks — a click is a strong landmark that would hand the
aligner free help and make the long-reference numbers look better than they are.

The 60 s and 180 s rows are labelled `ref_kind=padded` in the raw data. If they
disagree with `native`, the concatenation is a candidate explanation and must be
ruled out before the result is believed.

---

## Registered predictions

Written before the run, per the convention in `align_bench/README.md` §3. The
point is that the result can contradict this instead of being rationalised after.

1. **`gcc_phat`'s knee for hit@50 ms falls between 200 and 500 ms** on native
   references. Below 100 ms it degrades sharply.
2. **Null false-alarm rate at fixed τ rises monotonically with reference length**
   (native → 60 s → 180 s). More candidate positions, more chances at a spurious
   peak.
3. **`gcc_phat`'s PSR separates matched from null better than `aof`'s
   `standard_score`**, consistent with the bake-off's 0.888 vs 0.772 confidence AUC.
4. **Silent and low-energy frames account for the majority of short-frame
   failures** — the voiced-only knee sits at a shorter frame length than the
   pooled knee.
5. **`aof` never reaches the 1 ms bar at any frame length.** It is MFCC-frame-rate
   limited, consistent with its 1.812 ms median on `crop_head`.

---

## Known threat to validity: PSR may not be comparable across conditions

`gcc_phat`'s confidence is a peak-to-sidelobe ratio computed against the sidelobe
population of the whole correlation. A 100 ms frame against a 180 s reference has
a completely different sidelobe population than a whole clip against a whole clip.

If raw PSR is not comparable across reference lengths, **it corrupts the ROC in
exactly the place this experiment is trying to measure** (prediction 2). This is
checked before the full run. If it fails, PSR is reported z-scored within each
condition and the raw value is kept alongside it. Either way the choice is
recorded here and in `params.json`.

---

## Phasing

- **Phase 1** — A + B, native references only. Smallest run that produces the
  threshold number. Stop and read before continuing.
- **Phase 2** — extend to 60 s and 180 s.
- **Phase 3** — layer the attack grid. Separate results directory, not this one.

---

## Conclusions

*(empty until the run completes — do not fill this in from expectation)*
