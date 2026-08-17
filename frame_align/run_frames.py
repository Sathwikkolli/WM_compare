"""
frame_align/run_frames.py -- run both frame experiments.

One Slurm array task per clip. Each task walks its own frame draws from
frames.json and writes ONE CSV; score_frames.py concatenates them.

  Experiment A (matched)     reference contains the frame. Truth is known
                             exactly, because make_frames.py chose the cut point.
  Experiment B (null)        reference is a DIFFERENT clip. No truth exists, so
                             every accept is a false alarm by construction.

Both use the same draws, so A and B differ only in which reference the frame is
searched against. That is the whole design: it makes the confidence distributions
directly comparable, which is what produces a threshold.

WHY raw_score IS A COLUMN

`methods.py` squashes each method's native statistic into a [0,1] confidence
(log10 for PSR, /20 for standard_score). Squashing is monotone, so it does not
change a ROC -- but it does destroy the ability to check whether the statistic
itself is comparable across reference lengths, which the run README flags as the
main threat to validity. The un-squashed value is parsed back out of the method's
note and kept alongside. methods.py is imported unmodified, so parsing is the
price of not forking it.

Usage:
    python run_frames.py --ref-kinds native          # PHASE 1
    python run_frames.py --clip 3 --ref-kinds native
    python run_frames.py                             # all ref kinds (phase 2)
    python run_frames.py --methods gcc_phat
    # under Slurm, SLURM_ARRAY_TASK_ID selects the clip
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "align_bench"))

import methods as M                              # noqa: E402  (align_bench)
import refs as R                                 # noqa: E402
from make_clips import load_16k, WORK_SR         # noqa: E402  (align_bench)

RUN_SLUG = "2026-08-18_frame-align-null"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FRAMES_JSON = os.path.join(HERE, "frames.json")
CLIPS_JSON = os.path.join(ROOT, "align_bench", "clips.json")

# The two survivors of the bake-off that have a usable confidence score.
# audalign_corr is excluded on purpose -- see the run README.
DEFAULT_METHODS = ["gcc_phat", "aof"]

FIELDS = [
    "clip_idx", "frame_len_ms", "trial", "start_sample",
    "ref_kind", "ref_target_s", "ref_actual_s", "ref_home_start",
    "experiment", "null_ref_clip_idx",
    "frame_rms", "frame_dbfs", "touches_xfade",
    "method", "true_offset", "pred_offset", "error_samples", "error_ms",
    "confidence", "raw_score", "ok", "runtime_s", "note",
]

_PSR_RE = re.compile(r"psr=([0-9.eE+-]+)")
_SS_RE = re.compile(r"standard_score=([0-9.eE+-]+)")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default):
    if flag in argv:
        return [x for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return list(default)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def raw_score(method, note):
    """The method's native, un-squashed statistic. nan if not reported."""
    rx = _PSR_RE if method == "gcc_phat" else _SS_RE if method == "aof" else None
    if rx is None:
        return float("nan")
    m = rx.search(note or "")
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def dbfs(x):
    r = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return r, (20.0 * np.log10(r) if r > 1e-12 else -120.0)


def ref_seed(clip_idx, kind):
    """Vary where the home clip sits per (clip, ref kind), reproducibly."""
    k = 0 if kind == "native" else int(float(kind))
    return (clip_idx * 7919 + k * 31 + 17) % (2 ** 31 - 1)


def build_refs(clip_idx, home, pool, ref_kinds):
    return {k: R.build(k, home, pool, sr=WORK_SR, seed=ref_seed(clip_idx, k))
            for k in ref_kinds}


def write_params(cal, frames, clips, methods_used, ref_kinds, phase_note):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import platform
    versions = {}
    for mod in ("numpy", "scipy", "librosa", "soundfile", "audio_offset_finder"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            versions[mod] = "not installed"
    p = {
        "run": RUN_SLUG,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "phase": phase_note,
        "emilia_csv": clips.get("manifest"),
        "clip_seed": clips.get("seed"),
        "frame_seed": frames.get("seed"),
        "n_clips": frames.get("n_clips"),
        "frame_lens_ms": frames.get("frame_lens_ms"),
        "n_trials": frames.get("n_trials"),
        "ref_kinds": ref_kinds,
        "xfade_ms": R.XFADE_MS,
        "work_sr": WORK_SR,
        "methods": methods_used,
        "excluded_methods": {
            "audalign_corr": "confidence anti-calibrated at the top end "
                             "(bake-off S7); a threshold from an inverted score "
                             "is worse than no threshold",
            "dtw_subseq": "disqualified in the bake-off, 22.8% false-shift",
        },
        "sign_calibration": cal,
        "python": platform.python_version(),
        "packages": versions,
    }
    with open(os.path.join(RESULTS_DIR, "params.json"), "w") as f:
        json.dump(p, f, indent=2)
    return p


def run_clip(clip_idx, draws, paths, methods_used, ref_kinds, signs, writer,
             verbose=True):
    home = load_16k(paths[clip_idx])

    need_pool = any(k != "native" for k in ref_kinds)
    pool = []
    if need_pool:
        # Enough material to fill the longest reference; loading all 29 others
        # would be wasted IO on most tasks.
        longest = max(float(k) for k in ref_kinds if k != "native")
        for j, p in enumerate(paths):
            if j == clip_idx:
                continue
            try:
                pool.append(load_16k(p))
            except Exception as e:
                print(f"    pool load failed {os.path.basename(p)}: {e}")
            if sum(len(y) for y in pool) > longest * WORK_SR * 1.5:
                break
        if verbose:
            print(f"    pool: {len(pool)} clips, "
                  f"{sum(len(y) for y in pool)/WORK_SR:.0f}s")

    my_refs = build_refs(clip_idx, home, pool, ref_kinds)

    # Null references are built around the PARTNER clip, cached because the
    # rotation reuses each partner many times.
    null_cache = {}

    def null_ref(partner_idx, kind):
        key = (partner_idx, kind)
        if key not in null_cache:
            y = load_16k(paths[partner_idx])
            null_cache[key] = R.build(kind, y, pool, sr=WORK_SR,
                                      seed=ref_seed(partner_idx, kind))
        return null_cache[key]

    n_rows = 0
    for d in draws:
        start = d["start_sample"]
        n = d["frame_len_samples"]
        frame = home[start:start + n]
        if len(frame) < n:
            continue
        rms, db = dbfs(frame)

        for kind in ref_kinds:
            meta_a = my_refs[kind]
            xf = meta_a["home_xfade_n"]
            touches = int(xf > 0 and (start < xf or start + n > len(home) - xf))

            cases = [("A", meta_a, R.true_offset(meta_a, start), "")]
            try:
                meta_b = null_ref(d["null_ref_clip_idx"], kind)
                cases.append(("B", meta_b, float("nan"), d["null_ref_clip_idx"]))
            except Exception as e:
                if verbose:
                    print(f"    null ref {d['null_ref_clip_idx']}/{kind} failed: {e}")

            for exp, meta, truth, partner in cases:
                for mname in methods_used:
                    sign = signs.get(mname, 1.0)
                    if not np.isfinite(sign):
                        sign = 1.0
                    out = M.run(mname, meta["audio"], frame, WORK_SR, sign=sign)

                    pred = out["offset"]
                    good = bool(out["ok"]) and np.isfinite(pred)
                    if good and np.isfinite(truth):
                        err = float(pred - truth)
                        err_ms = err / WORK_SR * 1000.0
                    else:
                        err = err_ms = float("nan")

                    writer.writerow({
                        "clip_idx": clip_idx,
                        "frame_len_ms": d["frame_len_ms"],
                        "trial": d["trial"],
                        "start_sample": start,
                        "ref_kind": kind,
                        "ref_target_s": round(meta["target_s"], 2),
                        "ref_actual_s": round(meta["actual_s"], 2),
                        "ref_home_start": meta["home_start"],
                        "experiment": exp,
                        "null_ref_clip_idx": partner,
                        "frame_rms": round(rms, 8),
                        "frame_dbfs": round(db, 2),
                        "touches_xfade": touches,
                        "method": mname,
                        "true_offset": "" if not np.isfinite(truth) else round(truth, 2),
                        "pred_offset": "" if not good else round(float(pred), 2),
                        "error_samples": "" if not np.isfinite(err) else round(err, 2),
                        "error_ms": "" if not np.isfinite(err_ms) else round(err_ms, 4),
                        "confidence": "" if not np.isfinite(out["confidence"])
                                      else round(float(out["confidence"]), 6),
                        "raw_score": round(raw_score(mname, out.get("note", "")), 4),
                        "ok": int(bool(out["ok"])),
                        "runtime_s": round(out["runtime_s"], 4),
                        "note": (out.get("note") or "")[:160],
                    })
                    n_rows += 1
        if verbose and d["trial"] == 0:
            print(f"    {d['frame_len_ms']:>5d}ms ...")
    return n_rows


def main(argv):
    for path, what in ((FRAMES_JSON, "make_frames.py"),
                       (CLIPS_JSON, "align_bench/make_clips.py")):
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing -- run {what} first")

    with open(FRAMES_JSON) as f:
        frames = json.load(f)
    with open(CLIPS_JSON) as f:
        clips = json.load(f)

    methods_used = get_list(argv, "--methods", DEFAULT_METHODS)
    ref_kinds_raw = get_list(argv, "--ref-kinds", frames.get("ref_kinds", ["native"]))
    ref_kinds = [k if k == "native" else int(k) for k in ref_kinds_raw]
    phase_note = ("phase 1 (native references only)"
                  if ref_kinds == ["native"] else f"ref_kinds={ref_kinds}")

    paths = clips["clips"]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    clip_arg = get_arg(argv, "--clip", None, int)
    if clip_arg is not None:
        indices = [clip_arg]
    elif task is not None:
        indices = [int(task)]
    else:
        indices = sorted({d["clip_idx"] for d in frames["draws"]})

    print("calibrating method sign conventions...")
    cal = M.calibrate(sr=WORK_SR)
    for k in methods_used:
        v = cal.get(k, {})
        state = f"sign={v.get('sign'):+.0f}" if v.get("ok") else "UNUSABLE"
        print(f"  {k:16s} {state:10s} {v.get('note', 'not in registry')}")
    signs = M.signs_only(cal)

    bad = [k for k in methods_used if not cal.get(k, {}).get("ok")]
    if bad and "--force" not in argv:
        raise SystemExit(
            f"\n{bad} failed calibration. Install them, drop them with "
            f"--methods, or pass --force to record them as misses.")

    os.makedirs(DATA_DIR, exist_ok=True)
    write_params(cal, frames, clips, methods_used, ref_kinds_raw, phase_note)
    print(f"\n{phase_note}   methods={methods_used}")

    by_clip = {}
    for d in frames["draws"]:
        by_clip.setdefault(d["clip_idx"], []).append(d)

    for ci in indices:
        draws = by_clip.get(ci, [])
        if not draws:
            print(f"clip {ci}: no draws in frames.json -- skipping")
            continue
        out_csv = os.path.join(DATA_DIR, f"raw_clip{ci:03d}.csv")
        print(f"\nclip {ci}: {os.path.basename(paths[ci])}  ({len(draws)} draws)")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n_rows = run_clip(ci, draws, paths, methods_used, ref_kinds, signs, w)
        print(f"  {n_rows} rows, {time.time() - t0:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
