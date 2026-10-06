# masking — masking threshold inside AWARE

Experiments for replacing AWARE's fixed per-bin budget with a psychoacoustic
masking threshold. Each run archives to `results/<date>_masking-*/`.

| Script | Does |
|---|---|
| `common.py` | paths (ASVspoof19 LA eval on Great Lakes), AWARE import, `_optimize` capture, attacks, quality metrics |
| `make_clips.py` | picks 30 bona fide clips, one per speaker -> `clips.json` (login node) |
| `run_e0_e1.py` | one clip: E0 baseline + E1a budget usage + E1b window readability |
| `e0_e1.sbatch` | Slurm array 0-29 |
| `aggregate.py` | tables, gate verdicts, figures -> `results/<run>/summary.md` |
| `psy.py` | masking threshold (Johnston-style) and AWARE budget on AWARE's STFT grid |
| `run_phase1.py` / `phase1.sbatch` / `aggregate_phase1.py` | Phase 1, E2-E6 (budget shape tests, no embedder) |

```bash
conda activate wmcompare && cd $WM_COMPARE_BASE/masking
python make_clips.py
python run_e0_e1.py --index 0     # smoke test
sbatch e0_e1.sbatch
python aggregate.py               # after the array finishes
```

ODG needs the GstPEAQ `peaq` binary on PATH (or `PEAQ_BIN=/path/to/peaq`);
without it the column stays empty and everything else runs.
