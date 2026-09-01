"""
informed/bisect_sweep.py -- Stage 2: where does each detector fail?

For every clip and every attack, bisect the strength axis to find the point at
which blind detection stops working, and the point at which informed detection
stops working. The distance between them is the whole result.

    gain = blind crossing - informed crossing,  in that attack's own units

WHY BISECTION AND NOT A GRID

Phase A used 2-8 fixed strengths per attack. Reading a crossing off 5 points
gives an answer +/- one grid step, and we are trying to measure a ~5 dB shift.
Ten bisection steps narrow the bracket to ~0.1% of the range, for a fifth of the
evaluations a grid of that resolution would need.

WHAT IS BEING BISECTED

Not the raw score -- the MARGIN:

    g(t) = score(t) - threshold(t)

Both terms move with strength. The score falls as the attack bites; the
threshold moves because the null distribution shifts as the residual gets
noisier (Stage 1 measured that curve). Bisecting g finds where the detector
stops clearing its own FPR-matched bar, which is the only fair definition.

THE BRACKET MUST BE VERIFIED

Bisection is only valid if the crossing is inside the interval. At t=0 the
detector must succeed; at t=1 it must fail. Both ends are checked and the
result is flagged when they do not hold:

    NO_CROSSING_SURVIVED   still detected at maximum strength -- the gain is a
                           LOWER BOUND, reported as ">= x", never as a number
    NO_CROSSING_FAILED     already failed at minimum strength -- the axis does
                           not start weak enough for this clip

Silently returning t=0 or t=1 in those cases would manufacture crossings that
never happened, and they would look like ordinary data points.

Usage:
    python bisect_sweep.py --clip 0
    python bisect_sweep.py --attacks music_bed,mp3
    python bisect_sweep.py --clip 0 --attacks music_bed --curve   # for Figure 1
    # under Slurm, SLURM_ARRAY_TASK_ID selects the clip
"""
import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                     # noqa: E402
import informed_detector as ID                 # noqa: E402
import strength_axis as SA                     # noqa: E402

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
NULL_DIR = os.path.join(DATA_DIR, "null")
SWEEP_DIR = os.path.join(DATA_DIR, "sweep")
CLIPS_JSON = os.path.join(HERE, "clips.json")

CLIP_SECONDS = 10.0
N_BISECT = 10                 # ~0.1% of the range
FPR = 0.01                    # primary; 0.05 also written by null_calibrate
MESSAGE_BITS = 20

CURVE_POINTS = 15          # --curve mode only, for the gain-curve figure

CURVE_FIELDS = ["clip_id", "attack", "unit", "arm", "t", "value",
                "score", "threshold", "margin"]

FIELDS = [
    "clip_id", "speaker", "attack", "unit", "arm", "fpr",
    "status", "t_cross", "value_cross",
    "score_at_t0", "thr_at_t0", "score_at_t1", "thr_at_t1",
    "n_evals", "message", "runtime_s", "note",
]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default):
    if flag in argv:
        return [x.strip() for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return list(default)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
#  threshold curves from Stage 1
# --------------------------------------------------------------------------- #
def load_thresholds(fpr=FPR):
    """{(attack, arm): (t_grid, threshold)} for interpolation during bisection."""
    files = sorted(glob.glob(os.path.join(NULL_DIR, "null_*.csv")))
    if not files:
        raise SystemExit(
            f"no null_*.csv in {NULL_DIR}\n"
            f"Run Stage 1 first: python null_calibrate.py --prep, then the "
            f"nullcal array.")
    curves = {}
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    if abs(float(r["fpr"]) - fpr) > 1e-9:
                        continue
                    thr = float(r["threshold"])
                    t = float(r["strength"])
                except (ValueError, TypeError, KeyError):
                    continue
                curves.setdefault((r["attack"], r["arm"]), []).append((t, thr))
    out = {}
    for k, pts in curves.items():
        pts.sort()
        out[k] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def threshold_at(curves, attack, arm, t):
    """Interpolate the FPR-matched threshold at an arbitrary strength."""
    key = (attack, arm)
    if key not in curves:
        return float("nan")
    ts, thr = curves[key]
    good = np.isfinite(thr)
    if not good.any():
        return float("nan")
    return float(np.interp(t, ts[good], thr[good]))


# --------------------------------------------------------------------------- #
#  the two detectors
# --------------------------------------------------------------------------- #
def blind_score(adapter, attacked):
    try:
        conf, _bits, _acc = adapter.detect(attacked)
        return float(conf)
    except Exception:
        return float("nan")


def informed_score(org, wm, attacked, sr):
    r = ID.score(org, wm, attacked, sr=sr, method="scalar")
    return r["corr_windowed"] if r["ok"] else float("nan")


def evaluate(attack, t, org, wm, adapter, sr, arm):
    """Score at strength t. nan means the attack could not run."""
    speech = wm if arm != "null" else org
    z = SA.apply_at(attack, t, speech, sr)
    if z is None:
        return float("nan")
    z = np.asarray(z, dtype="float32")
    if arm == "blind":
        return blind_score(adapter, z)
    return informed_score(org, wm, z, sr)


# --------------------------------------------------------------------------- #
#  bisection
# --------------------------------------------------------------------------- #
def find_crossing(attack, arm, org, wm, adapter, sr, curves, n_iter=N_BISECT):
    """Where does `arm` stop clearing its FPR-matched threshold?

    Returns a dict. `status` is one of CROSSED / NO_CROSSING_SURVIVED /
    NO_CROSSING_FAILED / UNAVAILABLE, and only CROSSED carries a usable number.
    """
    def margin(t):
        s = evaluate(attack, t, org, wm, adapter, sr, arm)
        thr = threshold_at(curves, attack, arm, t)
        if not (np.isfinite(s) and np.isfinite(thr)):
            return float("nan"), s, thr
        return s - thr, s, thr

    n_evals = 0
    g0, s0, thr0 = margin(0.0)
    g1, s1, thr1 = margin(1.0)
    n_evals += 2

    base = {"score_at_t0": s0, "thr_at_t0": thr0,
            "score_at_t1": s1, "thr_at_t1": thr1, "n_evals": n_evals}

    if not np.isfinite(g0) or not np.isfinite(g1):
        return dict(base, status="UNAVAILABLE", t_cross=float("nan"),
                    note="attack unavailable or threshold missing")
    if g0 <= 0:
        # Already failing at the weakest setting. Not a crossing at t=0 -- the
        # axis does not start weak enough, and reporting 0 would be a fiction.
        return dict(base, status="NO_CROSSING_FAILED", t_cross=float("nan"),
                    note="detector already below threshold at t=0")
    if g1 > 0:
        # Survived the whole axis. The gain is a lower bound, not a value.
        return dict(base, status="NO_CROSSING_SURVIVED", t_cross=float("nan"),
                    note="detector still above threshold at t=1")

    lo, hi = 0.0, 1.0                    # g(lo) > 0, g(hi) <= 0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        g, _s, _t = margin(mid)
        n_evals += 1
        if not np.isfinite(g):
            # A single unavailable probe should not abort the search; treat it
            # as "failed" so the bracket keeps shrinking from the strong side.
            hi = mid
            continue
        if g > 0:
            lo = mid
        else:
            hi = mid

    return dict(base, status="CROSSED", t_cross=0.5 * (lo + hi),
                n_evals=n_evals, note="")


def run_curve(ci, clip, attacks, adapter, cl, curves, writer, verbose=True):
    """Evaluate a fixed grid instead of bisecting. Figure 1 only.

    Bisection concentrates its probes near the crossing, which is what makes it
    efficient and what makes it useless for drawing a curve. This walks an even
    grid so the shape either side of the crossing is visible.
    """
    org = cl.read_wav(clip["path"])
    n_keep = int(CLIP_SECONDS * cl.SR_MASTER)
    if len(org) < n_keep:
        return 0
    org = org[:n_keep].astype("float32")
    sr = cl.SR_MASTER

    rng = np.random.RandomState(1000 + ci)
    bits = "".join(rng.choice(["0", "1"]) for _ in range(MESSAGE_BITS))
    old = adapter.truth
    try:
        adapter.truth = bits
        wm = np.asarray(adapter.embed(org), dtype="float32")[:n_keep]
    finally:
        adapter.truth = old

    n_rows = 0
    for attack in attacks:
        unit = SA.AXIS[attack].get("unit", "")
        for tt in np.linspace(0.0, 1.0, CURVE_POINTS):
            for arm in ("blind", "informed"):
                old = adapter.truth
                try:
                    adapter.truth = bits
                    s = evaluate(attack, float(tt), org, wm, adapter, sr, arm)
                finally:
                    adapter.truth = old
                thr = threshold_at(curves, attack, arm, float(tt))
                writer.writerow({
                    "clip_id": clip["clip_id"], "attack": attack, "unit": unit,
                    "arm": arm, "t": round(float(tt), 6),
                    "value": round(float(SA.value(attack, float(tt))), 6),
                    "score": round(s, 6) if np.isfinite(s) else "",
                    "threshold": round(thr, 6) if np.isfinite(thr) else "",
                    "margin": round(s - thr, 6)
                              if np.isfinite(s) and np.isfinite(thr) else "",
                })
                n_rows += 1
        if verbose:
            print(f"    {attack:22s} curve done")
    return n_rows


# --------------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------------- #
def write_params(clips, attacks, fpr):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = {
        "run": RUN_SLUG, "phase": "B bisection sweep",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "n_clips": len(clips["clips"]), "clip_seconds": CLIP_SECONDS,
        "attacks": attacks, "n_bisect": N_BISECT, "fpr": fpr,
        "message_bits": MESSAGE_BITS,
        "message": "random per clip, seeded by clip index",
        "informed_statistic": "windowed normalised correlation, 42 ms / 50% "
                              "overlap, mean (registered primary)",
        "host_removal": "scalar gain (registered primary); FIR is secondary",
        "aligner": "gcc_phat, sub-sample, from align_bench/methods.py",
        "excluded": SA.NO_AXIS,
        "python": platform.python_version(),
    }
    with open(os.path.join(RESULTS_DIR, "params_phase_b.json"), "w") as f:
        json.dump(p, f, indent=2)


def run_clip(ci, clip, attacks, adapter, cl, curves, writer, fpr, verbose=True):
    org = cl.read_wav(clip["path"])
    n_keep = int(CLIP_SECONDS * cl.SR_MASTER)
    if len(org) < n_keep:
        print(f"  clip shorter than {CLIP_SECONDS}s -- skipping")
        return 0
    org = org[:n_keep].astype("float32")
    sr = cl.SR_MASTER

    # Random message per clip. A fixed AWARE_BITS for every clip would be a
    # confound: the result could be specific to one bit pattern.
    rng = np.random.RandomState(1000 + ci)
    bits = "".join(rng.choice(["0", "1"]) for _ in range(MESSAGE_BITS))

    t0 = time.time()
    old = adapter.truth
    try:
        adapter.truth = bits
        wm = np.asarray(adapter.embed(org), dtype="float32")[:n_keep]
        c0, _b, a0 = adapter.detect(wm)
    finally:
        adapter.truth = old
    if verbose:
        print(f"  msg={bits}  embedded in {time.time()-t0:.1f}s  "
              f"baseline conf={c0:.4f} bit_acc={a0:.4f}")

    n_rows = 0
    for attack in attacks:
        unit = SA.AXIS[attack].get("unit", "")
        line = f"    {attack:22s}"
        for arm in ("blind", "informed"):
            t1 = time.time()
            # The adapter must carry this clip's message for blind detection to
            # score bit accuracy against the right truth.
            old = adapter.truth
            try:
                adapter.truth = bits
                r = find_crossing(attack, arm, org, wm, adapter, sr, curves)
            finally:
                adapter.truth = old

            v = (SA.value(attack, r["t_cross"])
                 if np.isfinite(r["t_cross"]) else float("nan"))
            writer.writerow({
                "clip_id": clip["clip_id"], "speaker": clip["speaker"],
                "attack": attack, "unit": unit, "arm": arm, "fpr": fpr,
                "status": r["status"],
                "t_cross": "" if not np.isfinite(r["t_cross"]) else round(r["t_cross"], 6),
                "value_cross": "" if not np.isfinite(v) else round(float(v), 6),
                "score_at_t0": round(r["score_at_t0"], 6) if np.isfinite(r["score_at_t0"]) else "",
                "thr_at_t0": round(r["thr_at_t0"], 6) if np.isfinite(r["thr_at_t0"]) else "",
                "score_at_t1": round(r["score_at_t1"], 6) if np.isfinite(r["score_at_t1"]) else "",
                "thr_at_t1": round(r["thr_at_t1"], 6) if np.isfinite(r["thr_at_t1"]) else "",
                "n_evals": r["n_evals"], "message": bits,
                "runtime_s": round(time.time() - t1, 2), "note": r["note"],
            })
            n_rows += 1
            line += (f"  {arm[0]}:{r['status'][:4]}"
                     f"{('=' + format(v, '.3g')) if np.isfinite(v) else ''}")
        if verbose:
            print(line)
    return n_rows


def main(argv):
    if not os.path.exists(CLIPS_JSON):
        raise SystemExit("clips.json missing -- run clips.py first")
    clips = json.load(open(CLIPS_JSON))
    all_clips = clips["clips"]

    fpr = get_arg(argv, "--fpr", FPR, float)
    curves = load_thresholds(fpr)
    have = sorted({k[0] for k in curves})
    print(f"threshold curves for {len(have)} attacks at FPR={fpr}")

    attacks = get_list(argv, "--attacks", [a for a in sorted(SA.AXIS) if a in have])
    missing = [a for a in attacks if a not in have]
    if missing:
        raise SystemExit(f"no null calibration for {missing} -- run Stage 1 for them")

    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if "--clip" in argv:
        indices = [get_arg(argv, "--clip", 0, int)]
    elif task is not None:
        indices = [int(task)]
    else:
        indices = list(range(len(all_clips)))

    import cascade_lib as cl
    print("loading AWARE adapter...")
    adapter = cl.get_adapter("aware")
    print(f"{len(attacks)} attacks x 2 arms x ~{N_BISECT + 2} evals per clip\n")

    os.makedirs(SWEEP_DIR, exist_ok=True)
    write_params(clips, attacks, fpr)

    if "--curve" in argv:
        for ci in indices:
            if ci >= len(all_clips):
                continue
            clip = all_clips[ci]
            out_csv = os.path.join(SWEEP_DIR, f"curve_clip{ci:03d}.csv")
            print(f"curve: clip {ci} [{clip['clip_id']}]")
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                w.writeheader()
                n = run_curve(ci, clip, attacks, adapter, cl, curves, w)
            print(f"  {n} rows -> {out_csv}")
        return

    for ci in indices:
        if ci >= len(all_clips):
            print(f"clip index {ci} out of range ({len(all_clips)})")
            continue
        clip = all_clips[ci]
        out_csv = os.path.join(SWEEP_DIR, f"sweep_clip{ci:03d}.csv")
        print(f"clip {ci} [{clip['clip_id']}] {os.path.basename(clip['path'])}")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n = run_clip(ci, clip, attacks, adapter, cl, curves, w, fpr)
        print(f"  {n} rows, {time.time()-t0:.0f}s -> {out_csv}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
