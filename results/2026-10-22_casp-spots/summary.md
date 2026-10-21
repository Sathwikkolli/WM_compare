# Where CASP and the deployed MPEG watermark disagree

Run `2026-10-22_casp-spots`: **30** clips. Each cell: one 32 ms block of the deployed MPEG watermark (mask + time envelope), probed at -6/0/+6 dB with 3 PN seeds; CASP d' interpolated to the clip's reference (median d' of full-band probes at MPEG's level).

**Headroom (dB):** how much louder (+) or quieter (-) than MPEG's level the probe can be before CASP rates it as noticeable as a typical MPEG-level probe. Median of per-clip medians, 95% bootstrap interval over clips. Verdict needs the interval to clear +-1 dB.

## Full-band probes, by speech context

| context | clips | cells | headroom_db | ci_lo | ci_hi | share_room | verdict |
|---|---|---|---|---|---|---|---|
| long_pause | 30 | 3041 | -0.87 | -1.42 | -0.62 | 0.13 | agree (within +-1 dB) |
| short_gap | 29 | 183 | 1.53 | 1.09 | 3.19 | 0.83 | CASP sees room MPEG misses |
| tail | 30 | 412 | 1.31 | 1.00 | 1.98 | 0.87 | CASP sees room MPEG misses |
| onset | 30 | 244 | 0.15 | -1.60 | 2.16 | 0.50 | agree (within +-1 dB) |
| voiced | 30 | 1904 | 2.40 | 1.96 | 2.89 | 0.90 | CASP sees room MPEG misses |
| fricative | 29 | 350 | -1.32 | -1.62 | -0.46 | 0.24 | agree (within +-1 dB) |
| decay | 30 | 240 | -0.08 | -0.86 | 0.39 | 0.47 | agree (within +-1 dB) |
| other | 30 | 705 | -0.54 | -1.07 | -0.33 | 0.23 | agree (within +-1 dB) |

## Per band: groups where CASP and MPEG clearly disagree

6.3-8 kHz is at CASP's 16 kHz edge: treat it as low confidence.

| context | band_hz | clips | cells | headroom_db | ci_lo | ci_hi | share_room | verdict |
|---|---|---|---|---|---|---|---|---|
| voiced | 6264-8000 | 30 | 481 | 18.00 | 18.00 | 18.00 | 1.00 | CASP sees room MPEG misses |
| decay | 6264-8000 | 26 | 56 | 18.00 | 18.00 | 18.00 | 0.96 | CASP sees room MPEG misses |
| voiced | 4894-6264 | 30 | 481 | 15.74 | 13.99 | 17.82 | 1.00 | CASP sees room MPEG misses |
| other | 6264-8000 | 30 | 189 | 15.44 | 11.00 | 18.00 | 1.00 | CASP sees room MPEG misses |
| onset | 6264-8000 | 28 | 55 | 13.87 | 7.57 | 18.00 | 0.86 | CASP sees room MPEG misses |
| short_gap | 6264-8000 | 24 | 46 | 10.84 | 7.65 | 15.24 | 0.83 | CASP sees room MPEG misses |
| tail | 6264-8000 | 30 | 89 | 9.63 | 6.79 | 14.52 | 0.77 | CASP sees room MPEG misses |
| long_pause | 6264-8000 | 30 | 771 | 9.28 | 8.77 | 10.07 | 1.00 | CASP sees room MPEG misses |
| decay | 4894-6264 | 26 | 56 | 7.28 | 2.65 | 9.18 | 0.81 | CASP sees room MPEG misses |
| voiced | 1007-1337 | 30 | 481 | 5.79 | 4.36 | 6.80 | 0.97 | CASP sees room MPEG misses |
| voiced | 1337-1756 | 30 | 481 | 4.93 | 2.95 | 7.18 | 0.90 | CASP sees room MPEG misses |
| short_gap | 250-378 | 24 | 46 | 4.78 | 3.68 | 6.13 | 0.92 | CASP sees room MPEG misses |
| voiced | 3813-4894 | 30 | 481 | 4.28 | 3.25 | 6.74 | 0.93 | CASP sees room MPEG misses |
| voiced | 1756-2287 | 30 | 481 | 2.79 | 2.39 | 4.10 | 0.90 | CASP sees room MPEG misses |
| short_gap | 378-540 | 24 | 46 | 2.74 | 1.11 | 5.91 | 0.83 | CASP sees room MPEG misses |
| long_pause | 378-540 | 30 | 771 | 1.90 | 1.53 | 2.30 | 0.97 | CASP sees room MPEG misses |
| long_pause | 250-378 | 30 | 771 | 1.39 | 1.08 | 2.26 | 0.97 | CASP sees room MPEG misses |
| long_pause | 746-1007 | 30 | 771 | -1.46 | -2.12 | -1.04 | 0.03 | MPEG allows too much (per CASP) |
| long_pause | 3813-4894 | 30 | 771 | -1.54 | -1.93 | -1.25 | 0.10 | MPEG allows too much (per CASP) |
| voiced | 540-746 | 30 | 481 | -1.74 | -2.34 | -1.42 | 0.10 | MPEG allows too much (per CASP) |
| long_pause | 2960-3813 | 30 | 771 | -2.06 | -2.35 | -1.61 | 0.03 | MPEG allows too much (per CASP) |
| voiced | 378-540 | 30 | 481 | -2.12 | -3.34 | -1.35 | 0.17 | MPEG allows too much (per CASP) |
| fricative | 746-1007 | 28 | 93 | -2.29 | -3.18 | -1.73 | 0.07 | MPEG allows too much (per CASP) |
| decay | 378-540 | 26 | 56 | -2.60 | -4.01 | -1.00 | 0.23 | MPEG allows too much (per CASP) |
| decay | 2287-2960 | 26 | 56 | -3.40 | -6.90 | -1.84 | 0.15 | MPEG allows too much (per CASP) |
| long_pause | 2287-2960 | 30 | 771 | -3.50 | -3.89 | -3.05 | 0.00 | MPEG allows too much (per CASP) |
| other | 2287-2960 | 30 | 189 | -3.69 | -4.94 | -2.64 | 0.07 | MPEG allows too much (per CASP) |
| other | 2960-3813 | 30 | 189 | -3.71 | -4.90 | -2.64 | 0.10 | MPEG allows too much (per CASP) |
| other | 540-746 | 30 | 189 | -3.80 | -5.05 | -2.72 | 0.17 | MPEG allows too much (per CASP) |
| tail | 2287-2960 | 30 | 89 | -3.93 | -4.67 | -1.35 | 0.27 | MPEG allows too much (per CASP) |
| decay | 540-746 | 26 | 56 | -4.26 | -6.30 | -1.28 | 0.12 | MPEG allows too much (per CASP) |
| fricative | 2960-3813 | 28 | 93 | -4.58 | -5.28 | -4.03 | 0.04 | MPEG allows too much (per CASP) |
| fricative | 2287-2960 | 28 | 93 | -5.18 | -5.97 | -3.97 | 0.07 | MPEG allows too much (per CASP) |
| fricative | 3813-4894 | 28 | 93 | -6.09 | -7.30 | -5.12 | 0.00 | MPEG allows too much (per CASP) |
| fricative | 6264-8000 | 28 | 93 | -7.18 | -8.30 | -6.65 | 0.04 | MPEG allows too much (per CASP) |
| fricative | 4894-6264 | 28 | 93 | -10.34 | -11.36 | -9.40 | 0.00 | MPEG allows too much (per CASP) |

Heat map: `fig_context_band.png`. Full tables: `contexts_full.csv`, `contexts_band.csv`. Stimuli for the listening test: `listening_candidates.csv`.
