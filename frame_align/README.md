# frame_align — how short can a frame be, and when is an alignment fake?

Follow-on to `align_bench/`. That run aligned **whole clips** and picked
`gcc_phat`. This one asks the two questions it left open:

1. **How short can a piece of audio be and still align?** AWARE(20bps) hops its
   detector in 42 ms, so this sets a floor on how short a key-hop dwell can be
   before sync — not the watermark — is the limiting factor.
2. **What does an aligner do on audio that does not match at all?** The
   bake-off's `null` family is *audio that did not move*, which is a different
   thing. Nothing on record rules out confident alignment of unrelated speech.

The deliverable is one number per method: **at the threshold where false accepts
on unrelated audio sit at 1%, what fraction of genuine frames still align within
50 ms?**

Full design, registered predictions, and the declared deviation on long
references: `results/2026-08-18_frame-align-null/README.md`. Read that first.

## Run it

```bash
conda activate wmcompare        # same env as the bake-off

# 0. clip selection is reused verbatim from align_bench
cd $WM_COMPARE_BASE/align_bench && python make_clips.py     # if clips.json absent

cd $WM_COMPARE_BASE/frame_align
python refs.py                  # SELF-TEST: home_start must be recoverable
python refs.py --probe          # does Emilia have real 60s/180s clips?
python make_frames.py           # -> frames.json, prints the row budget

sbatch frame_align.sbatch       # array 0-29, phase 1 (native refs)

python score_frames.py          # -> summary.md + data/metrics.csv
```

`refs.py` self-test is not optional. It verifies that a frame cut at a known
position is found at that position in a built reference. If the crossfade
bookkeeping is off by one junction, every padded ground truth is wrong by ~10 ms
— which sails past the 50 ms bar and silently destroys the 1 ms results.

## Files

| File | Does |
|---|---|
| `make_frames.py` | Fixes every random draw up front → `frames.json` |
| `refs.py` | Builds native / 60 s / 180 s references; self-test; manifest probe |
| `run_frames.py` | Slurm-array runner, one task per clip. Both experiments |
| `score_frames.py` | Metrics → `summary.md`. **Metric meanings at the top of the file** |
| `frame_align.sbatch` | Phase 1 job. Phase 2 needs re-timing first |

`align_bench/methods.py` is **imported, not copied** — same aligners, same sign
calibration, so the two runs stay comparable.

## Design decisions worth knowing

**Draws are a file, not an RNG call in the runner.** A requeued Slurm task would
otherwise draw different frames than the one it replaced, and the run would
quietly stop being one experiment.

**Starts are uniform, not snapped to the 42 ms hop grid.** Snapping would let a
method score by landing on a grid point by luck. Real desync is not grid-aligned.

**Silence is stratified, never dropped.** A random 50 ms frame can land in an
Emilia pause where alignment is impossible and the method is not at fault.
`run_frames.py` writes per-frame RMS as a column; the voiced/low/silent cut is a
*scoring* decision, revisable without re-running the cluster job.

**`audalign_corr` is excluded despite being the production partner.** The output
here is a confidence threshold, and its confidence is anti-calibrated at the top
end (bake-off §7: 100% accurate in the 0.35–0.75 bins, **0% at 0.95**). A
threshold derived from an inverted score is worse than no threshold. `aof` runs
instead — usable unbounded `standard_score`, independent-ish MFCC front-end.

**`raw_score` is kept alongside `confidence`.** `methods.py` squashes each
method's native statistic into [0,1]. Squashing is monotone so it cannot change a
ROC, but it destroys the ability to check whether the statistic is comparable
across reference lengths — which is the main threat to validity here (see below).
The un-squashed value is parsed back out of the method's note.

## The threat to validity, stated up front

`gcc_phat`'s confidence is a **peak-to-sidelobe ratio**, and the sidelobe
population is completely different for a 100 ms frame against 180 s than for a
whole clip against a whole clip. If the null score distribution shifts with
reference length, **one global threshold is invalid** and τ has to be set per
condition.

`score_frames.py` §5 checks exactly this and prints a verdict. Do not quote the
headline number across reference lengths until that section says the spread is
small.

## Status

Phase 1 (native references) is built and smoke-tested end to end on synthetic
audio; **it has not been run on Emilia.** No real numbers exist yet.

Phase 2 (60 s / 180 s) is implemented but needs `frame_align.sbatch` re-timed
from phase 1's logs before submitting — 180 s references are ~20x the correlation
length.

Phase 3 (attacked frames) is **not** implemented and belongs in its own results
directory.

## Not implemented

- **Plots.** `score_frames.py` writes tables only. The knee is a curve and wants
  a figure, but nothing should be plotted before there are real numbers to plot.
- **Attacked frames.** Clean-only by design: without the clean floor you cannot
  tell a frame-length failure from a distortion failure.
