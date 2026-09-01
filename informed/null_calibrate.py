"""
informed/null_calibrate.py -- Stage 1: what does each detector score on audio
that carries no watermark, at every attack strength?

WHY THIS EXISTS

Blind and informed produce different quantities on different scales -- AWARE's
confidence in [0,1] versus a correlation. Comparing "where each crosses 0.50" is
meaningless, and informed could be made to look arbitrarily good simply by being
more willing to say yes. The comparison is only fair if BOTH arms are equally
willing to say yes on audio with no watermark.

`2026-08-18_frame-align-null` S3 measured the cost of skipping this: a
calibration built on matched data alone shipped a 30-41% false-accept rate while
every matched-data metric looked healthy.

WHY A GRID, NOT BISECTION

Stage 2 bisects the attack strength, so it probes arbitrary values. Nulls cannot
be recomputed at each probe -- that would be 200 clips x 10 bisection steps per
clip per attack. Instead this builds a threshold CURVE over a fixed strength
grid, and Stage 2 interpolates it. The null does not depend on which positive
clip is being tested, so it is computed once per (attack, strength).

WHY THE NULL CLIPS ARE EMBEDDED TOO

The informed statistic correlates a residual against the watermark pattern the
clip WOULD have carried. So each null clip needs `w_clean = wm - org` as a
reference, even though the audio being attacked is the UNwatermarked one. That
is the null: here is audio with no watermark, does the detector claim to see it?

Embedding is the expensive step, so it is cached once (`--prep`) and reused by
all 27 attack tasks.

    python null_calibrate.py --prep          # once: embed + cache null clips
    python null_calibrate.py --attack music_bed
    # under Slurm, SLURM_ARRAY_TASK_ID selects the attack
"""
import csv
import json
import os
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
CACHE = os.path.join(BASE, "real_audio", "null_cache.npz")

CLIP_SECONDS = 10.0
N_NULL = 200                 # see the resolution note below
GRID_POINTS = 12             # strengths per attack for the threshold curve

# Primary and secondary false-positive rates.
#
# RESOLUTION NOTE: with 200 nulls the 1% threshold is the 2nd-highest score, so
# it moves by a whole sample if one clip shifts. It is reported as primary
# because it matches 2026-08-18_frame-align-null, but 5% (10th-highest) is far
# more stable and is reported alongside. Where the two disagree about a crossing,
# say so rather than quoting 1% alone.
FPR_PRIMARY = 0.01
FPR_SECONDARY = 0.05

FIELDS = ["attack", "strength", "arm", "n", "fpr", "threshold",
          "null_p50", "null_p95", "null_max", "n_failed"]

# Every individual null score. Lets thresholds be re-derived at any FPR
# without re-running this stage, and feeds the score histograms.
RAW_FIELDS = ["attack", "strength", "arm", "score"]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


# --------------------------------------------------------------------------- #
#  cache: embed the null clips once
# --------------------------------------------------------------------------- #
def build_cache(argv):
    """Embed N_NULL clips and store org + w_clean. Run once, on a login node."""
    import cascade_lib as cl

    n_null = get_arg(argv, "--n", N_NULL, int)
    seed = get_arg(argv, "--seed", 7, int)

    clips_json = os.path.join(HERE, "clips.json")
    if not os.path.exists(clips_json):
        raise SystemExit("clips.json missing -- run clips.py first")
    positive = {c["path"] for c in json.load(open(clips_json))["clips"]}

    # Null clips are DISJOINT from the 50 positives: reusing them would make the
    # null and the measurement share content.
    import clips as C
    picked = C.select(C.EMILIA_CSV, n_null + len(positive), seed,
                      C.SECONDS, C.DNSMOS_MIN)
    paths = [str(r["path"]) for r in picked if str(r["path"]) not in positive]
    paths = paths[:n_null]
    if len(paths) < n_null:
        raise SystemExit(f"only {len(paths)} disjoint null clips available")

    print(f"embedding {len(paths)} null clips (this is the slow step)...")
    adapter = cl.get_adapter("aware")
    sr = cl.SR_MASTER
    n_keep = int(CLIP_SECONDS * sr)

    orgs, wcs, kept, msgs = [], [], [], []
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            y = cl.read_wav(p)
        except Exception:
            continue
        if len(y) < n_keep:
            continue
        y = y[:n_keep].astype("float32")
        # Random message per clip -- a fixed AWARE_BITS for every clip would be
        # a confound, since results could be specific to one bit pattern.
        bits = "".join(rng.choice(["0", "1"]) for _ in range(20))
        old = adapter.truth
        try:
            adapter.truth = bits
            wm = np.asarray(adapter.embed(y), dtype="float32")[:n_keep]
        except Exception as e:
            print(f"  embed failed for {os.path.basename(p)}: {e}")
            continue
        finally:
            adapter.truth = old
        orgs.append(y)
        wcs.append((wm - y).astype("float32"))
        kept.append(p)
        msgs.append(bits)
        if (i + 1) % 20 == 0:
            print(f"  {len(kept)}/{n_null}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, org=np.stack(orgs), w_clean=np.stack(wcs),
                        paths=np.array(kept), messages=np.array(msgs), sr=sr)
    print(f"\ncached {len(kept)} null clips -> {CACHE}  "
          f"({os.path.getsize(CACHE)/1e6:.0f} MB, {time.time()-t0:.0f}s)")


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit(f"{CACHE} missing -- run `python null_calibrate.py --prep`")
    z = np.load(CACHE, allow_pickle=False)
    return z["org"], z["w_clean"], int(z["sr"])


# --------------------------------------------------------------------------- #
#  calibration
# --------------------------------------------------------------------------- #
def threshold_at(scores, fpr):
    """Lowest score admitting at most `fpr` of the null. nan if none does."""
    s = np.sort(np.asarray([x for x in scores if np.isfinite(x)], dtype=float))
    if not len(s):
        return float("nan")
    k = int(np.ceil((1.0 - fpr) * len(s)))
    if k >= len(s):
        return float(s[-1]) + 1e-9
    return float(s[k])


def calibrate_attack(attack, orgs, wcs, sr, adapter, writer, raw_writer=None,
                     verbose=True):
    grid = SA.grid(attack, GRID_POINTS)
    if grid is None:
        print(f"{attack}: no strength axis -- skipped (control)")
        return 0

    n_rows = 0
    for strength in grid:
        param = SA.to_param(attack, strength)
        blind, informed = [], []
        n_failed = 0
        t0 = time.time()

        for i in range(len(orgs)):
            org = orgs[i]
            # THE NULL: attack the UNWATERMARKED clip.
            z = A.apply(attack, param, org, sr)
            if z is None:
                n_failed += 1
                continue
            z = np.asarray(z, dtype="float32")

            try:
                conf, _b, _a = adapter.detect(z)
                blind.append(float(conf))
            except Exception:
                n_failed += 1

            # Informed: correlate this clip's residual against the watermark it
            # WOULD have carried. There is no watermark, so any high score is a
            # false alarm.
            r = ID.score(org, org + wcs[i], z, sr=sr, method="scalar")
            if r["ok"]:
                informed.append(r["corr_windowed"])

        # Dump the raw null scores, not just the summary. Two reasons:
        # thresholds at any other FPR can then be re-derived WITHOUT re-running
        # this stage, and the score histograms in Figure 3 need the actual
        # distribution rather than percentiles.
        if raw_writer is not None:
            for arm, vals in (("blind", blind), ("informed", informed)):
                for v in vals:
                    if np.isfinite(v):
                        raw_writer.writerow({"attack": attack, "strength": strength,
                                             "arm": arm, "score": round(float(v), 6)})

        for arm, vals in (("blind", blind), ("informed", informed)):
            for fpr in (FPR_PRIMARY, FPR_SECONDARY):
                v = [x for x in vals if np.isfinite(x)]
                writer.writerow({
                    "attack": attack, "strength": strength, "arm": arm,
                    "n": len(v), "fpr": fpr,
                    "threshold": round(threshold_at(v, fpr), 6) if v else "",
                    "null_p50": round(float(np.percentile(v, 50)), 6) if v else "",
                    "null_p95": round(float(np.percentile(v, 95)), 6) if v else "",
                    "null_max": round(float(np.max(v)), 6) if v else "",
                    "n_failed": n_failed,
                })
                n_rows += 1

        if verbose:
            bt = threshold_at(blind, FPR_PRIMARY)
            it = threshold_at(informed, FPR_PRIMARY)
            print(f"  {SA.label(attack, strength):>14s}  "
                  f"blind thr={bt:7.4f} (n={len(blind):3d})  "
                  f"informed thr={it:7.4f} (n={len(informed):3d})  "
                  f"{time.time()-t0:5.0f}s")
    return n_rows


def main(argv):
    if "--prep" in argv:
        return build_cache(argv)

    orgs, wcs, sr = load_cache()
    print(f"null cache: {len(orgs)} clips at {sr} Hz")

    attacks = sorted(a for a in A.SCREEN_GRID if SA.grid(a, 2) is not None)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if "--attack" in argv:
        chosen = [get_arg(argv, "--attack", "")]
    elif task is not None:
        idx = int(task)
        if idx >= len(attacks):
            print(f"task {idx} >= {len(attacks)} attacks -- nothing to do")
            return
        chosen = [attacks[idx]]
    else:
        chosen = attacks

    import cascade_lib as cl
    print("loading AWARE adapter...")
    adapter = cl.get_adapter("aware")

    os.makedirs(NULL_DIR, exist_ok=True)
    for attack in chosen:
        out_csv = os.path.join(NULL_DIR, f"null_{attack}.csv")
        print(f"\n{attack}  ({GRID_POINTS} strengths x {len(orgs)} null clips)")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n = calibrate_attack(attack, orgs, wcs, sr, adapter, w)
        print(f"  {n} rows, {time.time()-t0:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
