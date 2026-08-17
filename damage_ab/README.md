# damage_ab — is the attack destroying the audio, or is AWARE fragile?

The A/B (`results/2026-08-10_aware-detection-ab`) found two failing cells,
`highpass_0.2` at 0/20 and `quantize_8lvl` at 1/20. It scored **only watermarked
audio at one strength each**, so it cannot distinguish two very different
explanations:

1. AWARE is fragile to these distortions, or
2. these distortions destroy the audio outright, and no watermark could survive
   in a file nobody would listen to.

This run separates them by putting **unwatermarked audio through the identical
attack** and sweeping the strength.

Results land in `results/2026-08-17_attack-damage-control/`. **Read that README
first** — the design, the pre-registered criterion, and the limits live there.

## Run it

```bash
conda activate wmcompare
cd $WM_COMPARE_BASE/damage_ab

# sanity checks before burning cluster time
python sweep.py                 # prints the 15 conditions, checks the A/B anchors,
                                # probes ffmpeg

# login node -- NOT in a batch job
python make_pairs.py            # -> clips.json      (5 Emilia clips, >=9 s)
python embed_pairs.py           # -> work/           (embeds + round-trip check)

# the run
sbatch damage_ab.sbatch         # single job, ~25 min

# after it finishes
python analyze.py               # -> summary.md + figures/ + data/by_condition.csv
```

`embed_pairs.py` warns loudly if any clip is undetected **before** any attack.
If that fires, stop — every sweep row from that clip is noise.

## Listening

The point of saving audio is to check by ear what PESQ is claiming. PESQ
saturates near 1.0, so it cannot rank two destroyed files; your ears can.

```bash
# from your laptop, not the cluster
scp -r greatlakes:~/wm_compare/damage_ab/work/listen ./listen
```

`work/listen/` holds **one clip across all 15 conditions in both arms**, flatly
named — `hp_3200hz_src.wav` vs `hp_3200hz_wm.wav`, etc. That pairing is the
listening test. `work/att/` has the same for all 5 clips if you want more.

The `_src` file is unwatermarked and the `_wm` file is watermarked. Under a
destructive attack they should sound equally ruined; if `_wm` sounds clearly
worse, that is the fragility hypothesis and the numbers should show it too.

## Files

| File | Does |
|---|---|
| `sweep.py` | The 15-condition grid. Cutoffs in **Hz**; asserts at import that its two anchor cells match `ab_aware/attacks_ab.py` |
| `make_pairs.py` | Picks 5 Emilia clips -> `clips.json`. **Matched pairs**, unlike ab_aware |
| `embed_pairs.py` | Writes both arms + payloads -> `work/` |
| `run_damage.py` | The run. **Four PESQ measurements per row, documented at the top of the file** |
| `analyze.py` | Applies the pre-registered criterion -> `summary.md` + figures |
| `damage_ab.sbatch` | Great Lakes job, single task |

## Why this is not just ab_aware with fewer clips

**1. Matched arms.** Every clip appears both unwatermarked and watermarked. The
A/B used disjoint arms because it needed an honest FPR; this run needs a
within-clip quality comparison, and at n=5 clip variance would otherwise swamp
the effect. The cost is that **this run cannot produce a false-positive rate** —
its `src` arm is a paired control, not an independent negative sample.

**2. Four PESQ references, not one.** `src→attack(src)` is the control the A/B
never had. `wm→attack(wm)` isolates attack damage from watermark cost. See the
table at the top of `run_damage.py`.

**3. Band retention.** The AWARE band is 1000–4000 Hz (`fsss/band_steer.py`).
Every row records how much of that band's energy survived, which separates
"emptied by a filter" (`<<1`) from "buried in switching noise" (`>1`). PESQ
cannot tell those apart and they are different failures.

**4. The criterion was fixed before the run**, in the results README, along with
both of its constants.

## Known gotchas

- **`mp3_32k` needs ffmpeg.** It is the quality anchor, not an optional extra: it
  is the condition AWARE survives 20/20 in the A/B, and without it the PESQ scale
  on this clip set has no reference point. `python sweep.py` checks for it.
- **`quant_256lvl` is not identical to the bench's `quantize_8bit`.** The bench
  quantiser (`run_bench.py:88`) is zero-centred; `vox_attacks.a_quantization`
  anchors its grid on the file's own min/max, so zero is generally not a level.
  That difference is the whole story at 8 levels — do not treat the two as
  interchangeable.
- **`work/` is gitignored.** The audio is regenerated on the cluster; only the
  CSVs, `summary.md` and figures are archived under `results/`.
