"""
align_bench/make_clips.py -- pick the Emilia clips for the alignment bake-off.

Emilia only, natural durations, no concatenation and no truncation: we take
whatever is in the manifest. Emilia clips run ~10 s (fsss/exp_a_repeatability.py
uses MIN_DUR = 9.0), so the benchmark covers the ~10 s regime -- NOT the 3-minute
end. That limitation is recorded in the run's README rather than papered over.

Also picks 5 HELD-OUT clips (never in the main 30) to serve as the foreign audio
for the insert_foreign attack, so nothing external is needed.

Usage:
    python make_clips.py                       # writes clips.json
    python make_clips.py --n 30 --seed 0
    EMILIA_CSV=/path/to/manifest.csv python make_clips.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

WORK_SR = 16000
N_CLIPS = 30
N_FOREIGN = 5
MIN_DUR = 9.0

# Same default fsss/exp_a_repeatability.py uses; override with the env var.
EMILIA_CSV = os.environ.get(
    "EMILIA_CSV",
    "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv",
)

OUT_JSON = os.path.join(HERE, "clips.json")

_PATH_KEYS = ("path", "audio_path", "wav", "filepath", "file", "audio", "filename")
_DUR_KEYS = ("duration", "dur", "length", "seconds", "duration_s")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def read_manifest(csv_path):
    """Return [(path, duration_or_None)] -- tolerant of column naming."""
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Emilia manifest not found: {csv_path}\n"
            f"Set EMILIA_CSV to the live path on the cluster."
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = [c.strip() for c in (rdr.fieldnames or [])]
        pcol = next((c for c in cols if c.lower() in _PATH_KEYS), None)
        dcol = next((c for c in cols if c.lower() in _DUR_KEYS), None)
        if pcol is None:                       # fall back: first column holding a path
            pcol = cols[0] if cols else None
        if pcol is None:
            raise SystemExit(f"could not find a path column in {csv_path}; cols={cols}")
        print(f"manifest columns: {cols}")
        print(f"  using path column '{pcol}'" +
              (f", duration column '{dcol}'" if dcol else ", no duration column"))
        for r in rdr:
            p = (r.get(pcol) or "").strip()
            if not p:
                continue
            d = None
            if dcol:
                try:
                    d = float(r.get(dcol) or "nan")
                except ValueError:
                    d = None
            rows.append((p, d))
    return rows


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

    # keep only clips long enough to survive a 75% head crop and still leave >1 s
    known = [(p, d) for p, d in rows if d is not None]
    if known:
        rows = [(p, d) for p, d in known if d >= MIN_DUR]
        print(f"after MIN_DUR={MIN_DUR}s filter: {len(rows)}")
    else:
        print("no duration column -- will filter after loading")

    if len(rows) < n + N_FOREIGN:
        raise SystemExit(f"only {len(rows)} usable clips, need {n + N_FOREIGN}")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))

    chosen, foreign, durations = [], [], []
    for i in idx:
        p, d = rows[i]
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
        if len(chosen) < n:
            chosen.append(p)
            durations.append(dur)
        elif len(foreign) < N_FOREIGN:
            foreign.append(p)
        else:
            break

    if len(chosen) < n:
        raise SystemExit(f"only found {len(chosen)} loadable clips, need {n}")

    durations = np.array(durations)
    out = {
        "manifest": csv_path,
        "seed": seed,
        "work_sr": WORK_SR,
        "min_dur_s": MIN_DUR,
        "clips": chosen,
        "foreign": foreign,
        "duration_stats": {
            "n": int(len(durations)),
            "min_s": float(durations.min()),
            "median_s": float(np.median(durations)),
            "max_s": float(durations.max()),
            "total_s": float(durations.sum()),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nselected {len(chosen)} clips + {len(foreign)} foreign")
    print(f"  duration  min={durations.min():.1f}s  median={np.median(durations):.1f}s"
          f"  max={durations.max():.1f}s")
    if durations.max() < 60:
        print("  NOTE: no clip exceeds 60 s -- the 3-minute regime is NOT covered.")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
