"""
spots/scan.py -- one clip -> CASP headroom relative to the deployed MPEG watermark, per cell.

    python scan.py --index 0                     # clip 0 of the clip list (sbatch array task)
    python scan.py --wav some16k.wav --quick     # local test on any 16 kHz wav

Cells:
  full-band   every FULL_EVERY-th MPEG block (default every block = 16 ms steps), 250 Hz - 8 kHz
  per-band    every BAND_EVERY-th block (default every 4th = 64 ms), each of 12 ERB bands
Each cell: CASP d' at -6 / 0 / +6 dB re MPEG's limit (3 PN seeds), headroom H, slope, context.

Writes results/<run>/<clip_id>/cells.csv and meta.json.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MPEG = os.path.dirname(HERE)
ROOT = os.path.dirname(MPEG)
sys.path.insert(0, HERE)
import probes as P          # noqa: E402
import contexts as C        # noqa: E402

RUN = os.environ.get("SPOTS_RUN", "2026-10-22_casp-spots")
CLIPS_JSON = os.path.join(MPEG, "clips120.json")
COLS = ["clip_id", "kind", "block", "t_s", "band", "band_lo_hz", "band_hi_hz", "context", "band_context",
        "level_db", "d_m6", "d_0", "d_p6", "d_ref", "headroom_db", "slope", "clipped"]


def load_clip(a):
    if a.wav:
        from scipy.io import wavfile
        sr, y = wavfile.read(a.wav)
        y = y.astype(np.float64) / 32768.0 if y.dtype == np.int16 else y.astype(np.float64)
        if sr != P.FS:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, P.FS)
            y = resample_poly(y, P.FS // g, sr // g)
        return os.path.splitext(os.path.basename(a.wav))[0], y
    sys.path.insert(0, os.path.join(ROOT, "casp"))
    import bm_core as B                                  # same loading as the benchmark
    spec = json.load(open(a.clips))["clips"][a.index]
    host22 = B.pcm16(B.load_master(spec["path"]))
    return spec["clip_id"], B.resample(host22, B.SR, B.SR16).astype(np.float64)


_HOST = None


def _init_worker(x):
    global _HOST
    _HOST = P.Host(x)


def _measure(job):
    """job = (key, block, band or None, seeds) -> (key, d' by level or None)."""
    key, b, band, seeds = job
    return key, P.measure_cell(_HOST, b, band, seeds)


def measure_all(x, jobs, workers, t0, label):
    """Run measure_cell over jobs, in parallel when workers > 1."""
    out = {}
    if workers <= 1:
        _init_worker(x)
        results = map(_measure, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(x,))
        results = ex.map(_measure, jobs, chunksize=4)
    for i, (key, d) in enumerate(results):
        if d is not None:
            out[key] = d
        if (i + 1) % 50 == 0:
            print(f"   {label} {i + 1}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    if workers > 1:
        ex.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--clips", default=CLIPS_JSON)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--full-every", type=int, default=1)     # every block: short gaps are only ~1-3 blocks long
    ap.add_argument("--band-every", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="a few cells only (plumbing test)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cid, x = load_clip(a)
    host = P.Host(x)
    feat = C.block_features(x)
    ctx = C.block_contexts(feat)
    seeds = tuple(range(a.seeds))
    edges = P.band_edges()
    n_blocks = len(host.starts)
    full_blocks = list(range(0, n_blocks, a.full_every))
    band_blocks = list(range(0, n_blocks, a.band_every))
    if a.quick:
        full_blocks, band_blocks = full_blocks[::10][:6], band_blocks[::10][:2]
    out_dir = a.out or os.path.join(ROOT, "results", RUN, cid)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"[{cid}] {len(x) / P.FS:.2f} s, {n_blocks} blocks; full-band cells {len(full_blocks)}, "
          f"band cells {len(band_blocks) * len(edges[:-1])}, seeds {len(seeds)}", flush=True)

    # ---- measure: full-band cells, then per-band cells (parallel over cells) ----------
    full = measure_all(x, [(b, b, None, seeds) for b in full_blocks], a.workers, t0, "full-band")
    d_ref_full = float(np.median([d[0.0] for d in full.values()]))
    band = measure_all(x, [((b, k), b, (edges[k], edges[k + 1]), seeds)
                           for b in band_blocks for k in range(len(edges) - 1)], a.workers, t0, "band")
    d_ref_band = float(np.median([d[0.0] for d in band.values()])) if band else float("nan")

    # ---- write ------------------------------------------------------------------------
    with open(os.path.join(out_dir, "cells.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        def row(kind, b, k, d, d_ref):
            h, slope, clipped = P.headroom(d, d_ref)
            lo, hi = (edges[k], edges[k + 1]) if k is not None else (P.BAND_LOW_HZ, P.BAND_HIGH_HZ)
            w.writerow([cid, kind, b, f"{(host.starts[b] + P.N / 2) / P.FS:.4f}", "" if k is None else k,
                        f"{lo:.0f}", f"{hi:.0f}", ctx[b], "" if k is None else C.band_label(feat, b, (lo, hi)),
                        f"{feat['level'][b]:.1f}", f"{d[-6.0]:.4f}", f"{d[0.0]:.4f}", f"{d[6.0]:.4f}",
                        f"{d_ref:.4f}", f"{h:.2f}", f"{slope:.3f}", int(clipped)])
        for b, d in full.items():
            row("full", b, None, d, d_ref_full)
        for (b, k), d in band.items():
            row("band", b, k, d, d_ref_band)
    json.dump(dict(clip_id=cid, seconds=len(x) / P.FS, d_ref_full=d_ref_full, d_ref_band=d_ref_band,
                   levels_db=list(P.LEVELS_DB), seeds=len(seeds), full_every=a.full_every,
                   band_every=a.band_every, pre_s=P.PRE_S, post_s=P.POST_S,
                   band_edges_hz=[float(e) for e in edges],
                   context_counts={c: ctx.count(c) for c in sorted(set(ctx))},
                   runtime_s=round(time.time() - t0, 1)),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"done [{time.time() - t0:.0f}s] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
