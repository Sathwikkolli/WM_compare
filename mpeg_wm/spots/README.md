# spots — where CASP sees what MPEG misses, and where MPEG does better

**Question:** in which speech situations does the CASP ear model say the MPEG
watermark could be louder (room MPEG misses), and in which does it say MPEG already
allows too much? Listening (Phase 3, later) decides which model is right.

## The measurement (Phase 1)

| Piece | What |
|---|---|
| **Probe** | One 32 ms block of the **deployed** MPEG watermark: MPEG mask + MPEG time envelope + Hann window, keyed PN noise. Either full-band (250 Hz–8 kHz) or one of 12 ERB bands. By construction, MPEG rates every probe "at its limit". |
| **CASP verdict** | d′ (JND units) of the host segment with vs without the probe. The segment includes 400 ms before (so CASP's adaptation memory is right) and 250 ms after. |
| **Headroom H (dB)** | Each cell is probed at −6, 0 and +6 dB re MPEG's level, with 3 PN seeds. Fit 20·log10(d′) against level, then solve for the level where d′ equals the clip's reference: the median d′ of full-band probes at MPEG's level (band cells use the median of band probes). **H > 0**: CASP says MPEG leaves room. **H < 0**: CASP says MPEG allows too much. Only *relative* judgements are used, so CASP's absolute strictness does not matter. |
| **Cells** | Full-band: every MPEG block (16 ms steps). Per band: every 4th block × 12 bands. |

## The labels (Phase 2)

`contexts.py` labels every block from the original audio:

| Label | Rule |
|---|---|
| `long_pause` | below −40 dB re clip peak for more than 150 ms |
| `short_gap` | a dip under 80 ms between loud speech |
| `tail` | within 200 ms after loud speech, falling |
| `onset` | rises more than 10 dB inside the block |
| `voiced` | speech level with pitch periodicity |
| `fricative` | speech level, mostly above 3 kHz |
| `decay` | speech level, falling |
| `other` | none of the above |

Every band also gets `peak` (within 6 dB of the block's strongest band), `above`
(above the frequency holding 90% of the block's energy) or `valley`. The thresholds
are constants at the top of the file.

## Files

| File | Role |
|---|---|
| `probes.py` | probe builder, CASP d′, headroom fit |
| `contexts.py` | speech-context and band labels |
| `scan.py` | one clip → `results/<run>/<clip>/cells.csv`; parallel over cells (`--workers`) |
| `spots.sbatch` | 30 clips (every 4th of `clips120.json`), 8 CPUs each |
| `aggregate.py` | per-context and per-(context, band) headroom, median of clip medians with a bootstrap interval over clips; heat map; `listening_candidates.csv` for Phase 3 |

## Run on Great Lakes

```bash
cd $WM_COMPARE_BASE && git pull && conda activate wmcompare && cd mpeg_wm/spots
python scan.py --index 0 --quick --workers 8 --out /tmp/spots_quick
sbatch spots.sbatch
python aggregate.py
```

## Known limits

- **6.3–8 kHz** is at CASP's 16 kHz edge: low confidence, and kept out of the listening candidates.
- **One probe at a time** is not the full watermark everywhere. Phase 4 tests the real watermark.
- **Short gaps are rare.** The per-band scan (every 4th block) samples them sparsely; the full-band scan covers every block.
