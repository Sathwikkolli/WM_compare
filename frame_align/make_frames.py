"""
frame_align/make_frames.py -- the frame draw plan.

Writes frames.json: for every (clip, frame_length, trial) it fixes the random
start position and the null partner clip. Nothing about audio content is decided
here -- this file is pure bookkeeping, so it runs in a second on a login node and
can be inspected before any cluster time is spent.

WHY THE PLAN IS A FILE AND NOT AN RNG CALL INSIDE THE RUNNER

Each Slurm array task handles one clip. If the runner drew its own frames, a
requeued task would draw DIFFERENT frames than the one it replaced, and the run
would silently stop being one experiment. Fixing the draws up front means a
failed task can be requeued individually and still produce the same rows.

WHAT IS DELIBERATELY NOT HERE

  frame energy / silence stratification
      Needs the audio. Computed by run_frames.py and written as a raw column, so
      the voiced/low/silent cut points stay a SCORING decision that can be
      revised without re-running the cluster job.

CONVENTIONS

  start_sample   frame = clip[start : start + n], drawn uniformly over every
                 legal start. NOT snapped to AWARE's 42 ms hop grid -- snapping
                 would let a method score by landing on a grid point by luck,
                 and real desync is not grid-aligned.

  true offset    methods report `ref_index - dist_index`. The frame IS dist, and
                 its content sits at reference index p, so the truth is simply p.
                 For native references p == start_sample; for padded references
                 refs.py shifts it by the home clip's position.

  null partner   deterministic rotation, never self. Rotation rather than a
                 random pick so every clip meets a spread of partners with no
                 chance of one pairing dominating the null distribution.

Usage:
    python make_frames.py                    # writes frames.json
    python make_frames.py --trials 20 --seed 0
    python make_frames.py --lens 50,100,250,500,1000,2000
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

# Reuse the bake-off's clip selection verbatim: same 30 Emilia clips, so any
# difference between the two runs is the experiment and not the corpus.
CLIPS_JSON = os.path.join(ROOT, "align_bench", "clips.json")
OUT_JSON = os.path.join(HERE, "frames.json")

WORK_SR = 16000
SEED = 0
N_TRIALS = 20

# 50 ms is just above AWARE's 42 ms detector hop -- the smallest frame that could
# possibly matter. 2000 ms is where whole-clip behaviour should be recovered.
FRAME_LENS_MS = (50, 100, 250, 500, 1000, 2000)

# "native" = the clip as-is (~9-27.8 s). 60/180 are padded, see refs.py.
REF_KINDS = ("native", 60, 180)


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast=str):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return list(default)


def draw_seed(seed, clip_idx, len_ms, trial):
    """Per-draw seed so one clip reproduces without replaying the others.

    Multipliers are distinct primes: no (clip, len, trial) triple collides with
    another within any run we would plausibly do.
    """
    return (seed * 1000003 + clip_idx * 10007 + len_ms * 101 + trial) % (2 ** 31 - 1)


def plan_for_clip(clip_idx, dur_s, n_clips, lens_ms, n_trials, seed):
    """Every draw for one clip. Returns [] for lengths the clip cannot supply."""
    rows = []
    for len_ms in lens_ms:
        n = int(round(len_ms / 1000.0 * WORK_SR))
        max_start = int(dur_s * WORK_SR) - n
        if max_start <= 0:
            # Clip shorter than the frame. Recorded as skipped rather than
            # silently absent, so score_frames.py can tell "no data" from "failed".
            continue
        for trial in range(n_trials):
            rng = np.random.RandomState(draw_seed(seed, clip_idx, len_ms, trial))
            start = int(rng.randint(0, max_start + 1))
            rows.append({
                "clip_idx": clip_idx,
                "frame_len_ms": len_ms,
                "frame_len_samples": n,
                "trial": trial,
                "start_sample": start,
                # rotation: +1 offset guarantees != self even at trial 0
                "null_ref_clip_idx": (clip_idx + 1 + (trial % (n_clips - 1))) % n_clips,
            })
    return rows


def main(argv):
    if not os.path.exists(CLIPS_JSON):
        raise SystemExit(
            f"{CLIPS_JSON} missing.\n"
            f"This run reuses the bake-off's clip selection. Build it first:\n"
            f"    cd {os.path.join(ROOT, 'align_bench')} && python make_clips.py")

    with open(CLIPS_JSON) as f:
        clips = json.load(f)

    lens_ms = get_list(argv, "--lens", FRAME_LENS_MS, int)
    n_trials = get_arg(argv, "--trials", N_TRIALS, int)
    seed = get_arg(argv, "--seed", SEED, int)

    paths = clips["clips"]
    n_clips = len(paths)
    if n_clips < 2:
        raise SystemExit(f"need >=2 clips for the mismatched null, have {n_clips}")

    # Durations come from clips.json's stats where possible; per-clip durations
    # are not stored there, so probe the files. Cheap: soundfile reads the header
    # only, no decode.
    import soundfile as sf
    durs = []
    for p in paths:
        try:
            info = sf.info(p)
            durs.append(info.frames / float(info.samplerate))
        except Exception as e:
            print(f"  ! header read failed, {os.path.basename(p)}: {e}")
            durs.append(0.0)

    usable = [d for d in durs if d > 0]
    if not usable:
        raise SystemExit("could not read any clip -- is EMILIA_CSV pointing at "
                         "the live path? This must run where the audio lives.")

    rows = []
    for ci, d in enumerate(durs):
        rows.extend(plan_for_clip(ci, d, n_clips, lens_ms, n_trials, seed))

    out = {
        "source_clips_json": CLIPS_JSON,
        "clip_seed": clips.get("seed"),
        "seed": seed,
        "work_sr": WORK_SR,
        "frame_lens_ms": lens_ms,
        "n_trials": n_trials,
        "ref_kinds": list(REF_KINDS),
        "n_clips": n_clips,
        "clip_durations_s": [round(d, 3) for d in durs],
        "draws": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # ---- report ------------------------------------------------------------
    print(f"clips: {n_clips}  (durations {min(usable):.1f}-{max(usable):.1f} s)")
    print(f"draws: {len(rows)}  = per clip x {len(lens_ms)} lengths x {n_trials} trials")
    print()
    print(f"  {'frame_len':>10s} {'draws':>7s}  {'clips supplying it':s}")
    for len_ms in lens_ms:
        got = [r for r in rows if r["frame_len_ms"] == len_ms]
        n_c = len({r["clip_idx"] for r in got})
        flag = "" if n_c == n_clips else f"   <-- {n_clips - n_c} clip(s) too short"
        print(f"  {len_ms:>8d}ms {len(got):>7d}  {n_c}/{n_clips}{flag}")

    # Row count is what decides whether the array fits in its time limit, so
    # state it here rather than discovering it from a timeout.
    per_method_rows = len(rows) * len(REF_KINDS) * 2      # x2 = experiments A and B
    print(f"\nrows per method (A+B, all ref kinds): {per_method_rows}")
    print(f"rows total for 2 methods:             {per_method_rows * 2}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
