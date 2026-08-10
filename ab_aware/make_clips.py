"""
ab_aware/make_clips.py -- pick the 40 Emilia clips for the AWARE detection A/B.

TWO DISJOINT ARMS, drawn from one shuffled pool:

    wm    -- 20 clips that WILL be AWARE-watermarked (the positives)
    clean -- 20 clips that stay untouched            (the negatives)

Disjoint, not matched pairs -- that was the call made when the run was designed.
The consequence is recorded here so nobody re-derives it later: clip identity is
confounded with arm, so a difference between arms carries clip-level variance as
well as the watermark. Paired tests between arms are therefore NOT valid, and
analyze.py does not run any. Drawing both arms from the same shuffled pool with
the same duration filter is what keeps them distributionally comparable.

WHY THE NEGATIVES MATTER: without them a detector that returns "watermarked"
unconditionally scores 100% and looks perfect. The negatives are the only thing
that separates real detection from a stuck output.

20 negatives is few, on purpose (the run was specified that way). It can catch a
gross false-positive problem but cannot certify a low FPR -- 0/20 leaves a 95%
Wilson interval of roughly [0, 16%]. analyze.py prints that interval next to
every rate so the number is never read as stronger than it is.

Usage:
    python make_clips.py                       # writes clips.json
    python make_clips.py --n-wm 20 --n-clean 20 --seed 0
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
N_WM = 20
N_CLEAN = 20

# crop_50 keeps half the clip; 9 s in leaves ~4.5 s out, still well above the
# ~1 s floor where detect_watermark's sliding window gives up. Same threshold
# align_bench used, so the two runs cover the same duration regime.
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
    n_wm = get_arg(argv, "--n-wm", N_WM, int)
    n_clean = get_arg(argv, "--n-clean", N_CLEAN, int)
    seed = get_arg(argv, "--seed", 0, int)
    csv_path = get_arg(argv, "--csv", EMILIA_CSV)
    need = n_wm + n_clean

    rows = read_manifest(csv_path)
    print(f"manifest rows: {len(rows)}")

    known = [(p, d) for p, d in rows if d is not None]
    if known:
        rows = [(p, d) for p, d in known if d >= MIN_DUR]
        print(f"after MIN_DUR={MIN_DUR}s filter: {len(rows)}")
    else:
        print("no duration column -- will filter after loading")

    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} usable clips, need {need}")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))

    # One pass over one shuffled pool. Arm assignment is by position in that
    # pool, so the two arms are drawn from the same distribution by construction
    # -- no second sampling step that could skew one arm.
    chosen, durations = [], []
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
        chosen.append(p)
        durations.append(dur)
        if len(chosen) >= need:
            break

    if len(chosen) < need:
        raise SystemExit(f"only found {len(chosen)} loadable clips, need {need}")

    wm_clips = chosen[:n_wm]
    clean_clips = chosen[n_wm:need]
    durations = np.array(durations)

    out = {
        "manifest": csv_path,
        "seed": seed,
        "work_sr": WORK_SR,
        "min_dur_s": MIN_DUR,
        "design": "disjoint arms (NOT matched pairs) -- no paired tests between arms",
        "wm_clips": wm_clips,
        "clean_clips": clean_clips,
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

    wm_d = durations[:n_wm]
    cl_d = durations[n_wm:need]
    print(f"\nselected {len(wm_clips)} wm + {len(clean_clips)} clean")
    print(f"  wm     duration  min={wm_d.min():.1f}s  median={np.median(wm_d):.1f}s  max={wm_d.max():.1f}s")
    print(f"  clean  duration  min={cl_d.min():.1f}s  median={np.median(cl_d):.1f}s  max={cl_d.max():.1f}s")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
