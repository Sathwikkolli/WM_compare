"""
align_bench/run_bench.py -- run the alignment bake-off.

One Slurm array task per clip. Each task walks the full 20-attack / 77-config
grid, runs every method on every (reference, distorted) pair, and writes ONE
CSV. score.py concatenates them afterwards.

Distorted audio is generated in memory and never written to disk -- the full
grid would be ~15-20 GB of wavs and we only need the numbers. Pass --save-audio
if you want a few examples kept for the ribbon plots.

Usage:
    python run_bench.py                       # all clips, all methods (local)
    python run_bench.py --clip 3              # one clip
    python run_bench.py --phase1              # 5 clips, all 6 methods (screen)
    python run_bench.py --methods gcc_phat,audalign_fp
    python run_bench.py --attacks crop_head,splice_cut
    # under Slurm, SLURM_ARRAY_TASK_ID selects the clip automatically
"""
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

import attacks_align as AA          # noqa: E402
import methods as M                 # noqa: E402
from make_clips import load_16k, WORK_SR   # noqa: E402

RUN_SLUG = "2026-08-05_align-method-bakeoff"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
CLIPS_JSON = os.path.join(HERE, "clips.json")

FIELDS = [
    "clip_idx", "clip_dur_s", "attack", "param", "family",
    "true_offset", "true_n_seg", "method", "pred_offset",
    "error_samples", "error_ms", "confidence", "ok", "runtime_s", "note",
]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default):
    if flag in argv:
        return [x for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_params(signs, clips, methods_used, attacks_used):
    """params.json -- everything needed to reproduce this run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import platform
    versions = {}
    for mod in ("numpy", "scipy", "librosa", "audalign", "soundfile"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    p = {
        "run": RUN_SLUG,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "emilia_csv": clips.get("manifest"),
        "clip_seed": clips.get("seed"),
        "n_clips": len(clips.get("clips", [])),
        "duration_stats": clips.get("duration_stats"),
        "work_sr": WORK_SR,
        "methods": methods_used,
        "attacks": attacks_used,
        "n_configs": AA.n_configs(),
        "sign_calibration": signs,
        "python": platform.python_version(),
        "packages": versions,
    }
    with open(os.path.join(RESULTS_DIR, "params.json"), "w") as f:
        json.dump(p, f, indent=2)
    return p


def run_clip(clip_idx, clip_path, foreign, methods_used, attacks_used, signs,
             writer, verbose=True):
    ref = load_16k(clip_path)
    dur = len(ref) / WORK_SR
    n_rows = n_skip = 0

    for atk in attacks_used:
        for label, param in AA.ALIGN_GRID[atk]:
            try:
                dist, gt = AA.apply(atk, param, ref, WORK_SR, foreign=foreign)
            except Exception as e:
                if verbose:
                    print(f"    {atk}/{label} GENERATE FAILED: {e}")
                dist, gt = None, None
            if dist is None or gt is None:
                n_skip += 1
                continue

            for mname in methods_used:
                sign = signs.get(mname, 1.0)
                if not np.isfinite(sign):
                    sign = 1.0
                out = M.run(mname, ref, dist, WORK_SR, sign=sign)

                pred = out["offset"]
                if out["ok"] and np.isfinite(pred):
                    err = float(pred - gt["offset"])
                    err_ms = err / WORK_SR * 1000.0
                else:
                    err = err_ms = float("nan")

                writer.writerow({
                    "clip_idx": clip_idx,
                    "clip_dur_s": round(dur, 3),
                    "attack": atk,
                    "param": label,
                    "family": gt["family"],
                    "true_offset": gt["offset"],
                    "true_n_seg": gt["n_segments"],
                    "method": mname,
                    "pred_offset": "" if not np.isfinite(pred) else round(float(pred), 2),
                    "error_samples": "" if not np.isfinite(err) else round(err, 2),
                    "error_ms": "" if not np.isfinite(err_ms) else round(err_ms, 4),
                    "confidence": "" if not np.isfinite(out["confidence"])
                                  else round(float(out["confidence"]), 4),
                    "ok": int(bool(out["ok"])),
                    "runtime_s": round(out["runtime_s"], 4),
                    "note": out["note"][:160],
                })
                n_rows += 1
        if verbose:
            print(f"    {atk:20s} done")
    return n_rows, n_skip


def main(argv):
    if not os.path.exists(CLIPS_JSON):
        raise SystemExit(f"{CLIPS_JSON} missing -- run make_clips.py first")
    with open(CLIPS_JSON) as f:
        clips = json.load(f)

    methods_used = get_list(argv, "--methods", list(M.METHODS.keys()))
    attacks_used = get_list(argv, "--attacks", list(AA.ALIGN_GRID.keys()))

    all_clips = clips["clips"]
    if "--phase1" in argv:
        all_clips = all_clips[:5]
        print("PHASE 1 SCREEN: 5 clips, all methods")

    # Slurm array -> one clip per task; --clip overrides; otherwise all
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    clip_arg = get_arg(argv, "--clip", None, int)
    if clip_arg is not None:
        indices = [clip_arg]
    elif task is not None:
        indices = [int(task)]
    else:
        indices = list(range(len(all_clips)))

    foreign = None
    if clips.get("foreign"):
        try:
            foreign = load_16k(clips["foreign"][0])
        except Exception as e:
            print(f"foreign clip load failed ({e}); insert_foreign will SKIP")

    print("calibrating method sign conventions...")
    signs = M.calibrate(sr=WORK_SR)
    for k, v in signs.items():
        print(f"  {k:16s} " + ("UNUSABLE" if not np.isfinite(v) else f"sign={v:+.0f}"))
    bad = [k for k, v in signs.items() if k in methods_used and not np.isfinite(v)]
    if bad:
        print(f"WARNING: these methods failed calibration and are probably not "
              f"installed / broken: {bad}")

    os.makedirs(DATA_DIR, exist_ok=True)
    write_params(signs, clips, methods_used, attacks_used)

    for ci in indices:
        if ci >= len(all_clips):
            print(f"clip index {ci} out of range ({len(all_clips)} clips) -- nothing to do")
            continue
        path = all_clips[ci]
        out_csv = os.path.join(DATA_DIR, f"raw_clip{ci:03d}.csv")
        print(f"\nclip {ci}: {os.path.basename(path)}")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n_rows, n_skip = run_clip(ci, path, foreign, methods_used,
                                      attacks_used, signs, w)
        print(f"  {n_rows} rows, {n_skip} configs skipped, "
              f"{time.time() - t0:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
