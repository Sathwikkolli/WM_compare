# align_bench -- alignment method bake-off

Picks which audio alignment method to use for **non-blind (informed) watermark
detection**: we hold the *unwatermarked* original, so lining it up against a
distorted watermarked copy should let the detector do better than it can blind.

Results land in `results/2026-08-05_align-method-bakeoff/` per the archive
convention in `results/README.md`.

## Run it

```bash
# once, on Great Lakes -- installs into the existing wmcompare env
conda activate wmcompare
pip freeze > $HOME/wmcompare_freeze_before_align.txt   # rollback point
pip install --dry-run -r requirements_extra.txt        # check what pip would change
pip install -r requirements_extra.txt

# confirm the watermark models still load
python -c "from aware.utils.models import load; load(name='AWARE(20bps)'); print('aware ok')"

# sanity check before burning cluster time
python attacks_align.py        # prints the 20 attacks / 77 configs by family
python methods.py              # calibrates sign conventions; shows what's installed

# the real run
sbatch align_bench.sbatch      # array 0-29, one task per clip

# after the array finishes
python score.py                # -> summary.md + data/metrics.csv
python plots.py                # -> figures/*.png
```

Set `--account` in `align_bench.sbatch` first; it is still a placeholder.

## Files

| File | Does |
|---|---|
| `make_clips.py` | Picks 30 Emilia clips + 5 held-out foreign clips -> `clips.json` |
| `attacks_align.py` | The 20-attack / 77-config grid **with ground truth** |
| `methods.py` | 6 aligners behind one interface + sign calibration |
| `run_bench.py` | Slurm-array runner, one task per clip |
| `score.py` | Metrics -> `summary.md`. **Metric meanings documented at the top.** |
| `plots.py` | 5 figures |
| `requirements_extra.txt` | the 2 extra packages + snapshot/rollback procedure |

## The three things that make this benchmark trustworthy

**1. Ground truth is written at generation time, not inferred.**
Every attack in `attacks_align.py` returns the exact segment map alongside the
audio. Nothing is reverse-engineered from file sizes.

**2. Sign conventions are calibrated, not assumed.**
Third-party aligners disagree about which direction an offset points. A silent
sign flip would poison every number and still look plausible. `methods.calibrate()`
feeds each method a synthetic clip with a known 1.0 s head crop and derives the
multiplier. The result is recorded in `params.json`.

**3. The prediction was written down before the run.**
See the run's README. Recording the expected outcome up front means the result
can contradict us instead of being rationalised afterwards.

## Two things we found reading the existing code

**VoxWatermark has no cropping attack.** `VOX_GRID` in `cascade/vox_attacks.py`
covers stretch, noise, codecs, filters, echo, jitter, polarity, phase -- but
nothing that removes audio. Since cropping is the whole reason an aligner is
needed, `crop_head`, `crop_tail`, `splice_cut`, `insert_silence` and
`insert_foreign` are implemented here.

**`phase_shift` does not shift.** `vox_attacks.a_phase` builds
`[zeros(shift), y[shift:]]`, so content stays at its original index -- it mutes
the head. True offset is **0**, so it is scored in the `null` family. Worth
knowing if it was ever read as a desync test elsewhere.

## Attack families and how they're scored

| Family | Attacks | Scored against |
|---|---|---|
| `null` | crop_tail, echo, lowpass, highpass, quantization, gaussian_noise, background_noise, dynamic_compression, inverse_polarity, phase_shift | exact 0. **False-shift rate lives here.** |
| `shift` | crop_head | one known constant offset |
| `multiseg` | splice_cut, insert_silence, insert_foreign | per-segment map; primary = longest segment |
| `warp` | time_stretch, time_jitter | offset at clip midpoint (approximate by construction) |
| `codec` | mp3, aac, opus, encodec | encoder delay is real but unknown -> bounded at 100 ms. Catches hallucination without pretending we know the delay. Raw predictions are kept, so the true delay is visible in the data. |

## Not yet implemented -- segment F1

`score.py` scores every family by **primary offset** (the offset of the longest
matched segment). Segment-level precision/recall is specified in the run README
but **not computed yet**, because `methods._dig_audalign` cannot reliably pull
per-segment matches out of audalign without seeing its actual return shape on
this version. The wrapper currently returns `segments: []` for every method.

Consequence: on the `multiseg` family, single-offset methods are scored on
whether they find the *dominant* segment. For a 1-cut splice the two halves are
near-equal, so they can score ~50% by coin flip. Read `multiseg` numbers as
"did it find a plausible piece", not "did it map the file".

Fix: run one clip, print the raw audalign dict, patch `_dig_audalign`, then add
the F1 computation. ~30 minutes once we can see real output.

## Known limitation

Emilia clips are ~10 s (`MIN_DUR = 9.0`). We take them as-is -- no concatenation,
no truncation -- so **the 3-minute regime is not covered**, and correlation-based
methods scale differently with length. Emilia is also speech only, so the
repetitive-music failure mode (several correlation peaks tie, alignment picks
the wrong one) goes untested. The Metapyxl client chain includes a music bed, so
this may matter in production.
