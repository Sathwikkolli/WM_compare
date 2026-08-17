"""
damage_ab/make_pairs.py -- pick 5 Emilia clips and build MATCHED arms.

    work/src/<cid>.wav    the clip, unwatermarked
    work/wm/<cid>.wav     the same clip, AWARE-embedded
    work/payloads.json    cid -> the 20 bits embedded

MATCHED PAIRS, NOT DISJOINT ARMS -- and this is the one design difference from
ab_aware that matters. The A/B used disjoint arms because it needed an honest
false-positive rate, and reusing a clip in both arms would have made the
negatives dependent on the positives. This experiment asks a different question:
"does the attack damage unwatermarked audio as badly as watermarked audio?"
That is a WITHIN-CLIP comparison. Using different clips per arm would put
clip-level variance directly on top of the effect being measured, and with n=5
that variance would swamp it.

The consequence, recorded up front so nobody misreads the output: this run
CANNOT produce a false-positive rate. Its `src` arm is not an independent
negative sample -- it is the paired control for a quality measurement. FPR lives
in results/2026-08-14_detector-null-test (n=300). Do not quote the src-arm
confidences here as an FPR.

n=5 is small ON PURPOSE. The claim this run is built to support is "the attack
destroys audio that has no watermark in it", which is a large, mechanical effect
visible per clip -- not a rate that needs tight intervals. If the 5 clips
disagree with each other, that is itself the finding and analyze.py says so.

Reuses ab_aware/make_clips.py's manifest reader rather than re-parsing the
manifest, so both runs resolve Emilia the same way.

Usage:
    python make_pairs.py                  # -> clips.json (login node)
    python make_pairs.py --n 5 --seed 0
    EMILIA_CSV=/path/to/manifest.csv python make_pairs.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ab_aware"))

from make_clips import EMILIA_CSV, read_manifest  # noqa: E402

WORK_SR = 16000
N_CLIPS = 5

# Matches ab_aware and align_bench, so all three cover the same duration regime.
# Nothing here crops, but PESQ wants a decent stretch of audio and a ~9 s floor
# keeps these clips comparable with the A/B's.
MIN_DUR = 9.0

OUT_JSON = os.path.join(HERE, "clips.json")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def load_16k(path):
    import librosa
    y, _ = librosa.load(path, sr=WORK_SR, mono=True)
    return y.astype("float32")


def main(argv):
    n = get_arg(argv, "--n", N_CLIPS, int)
    seed = get_arg(argv, "--seed", 0, int)
    csv_path = get_arg(argv, "--csv", EMILIA_CSV)

    rows = read_manifest(csv_path)
    print(f"manifest rows: {len(rows)}")

    known = [(p, d) for p, d in rows if d is not None]
    if known:
        rows = [(p, d) for p, d in known if d >= MIN_DUR]
        print(f"after MIN_DUR={MIN_DUR}s filter: {len(rows)}")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))

    chosen, durs = [], []
    for i in idx:
        p, _d = rows[i]
        if not os.path.exists(p):
            continue
        try:
            y = load_16k(p)
        except Exception as e:
            print(f"  skip (load failed) {p}: {e}")
            continue
        dur = len(y) / WORK_SR
        if dur < MIN_DUR:
            continue
        chosen.append(p)
        durs.append(dur)
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        raise SystemExit(f"only found {len(chosen)} loadable clips, need {n}")

    durs = np.array(durs)
    out = {
        "manifest": csv_path,
        "seed": seed,
        "work_sr": WORK_SR,
        "min_dur_s": MIN_DUR,
        "design": "MATCHED pairs -- every clip appears in both the src and wm arm. "
                  "Not a negative sample; no FPR can be computed from this run.",
        "clips": chosen,
        "duration_stats": {
            "n": int(len(durs)),
            "min_s": float(durs.min()),
            "median_s": float(np.median(durs)),
            "max_s": float(durs.max()),
            "total_s": float(durs.sum()),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nselected {len(chosen)} clips")
    for i, p in enumerate(chosen):
        print(f"  cl{i:02d}  {durs[i]:5.1f}s  {os.path.basename(p)}")
    print(f"  duration  min={durs.min():.1f}s  median={np.median(durs):.1f}s  "
          f"max={durs.max():.1f}s")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
