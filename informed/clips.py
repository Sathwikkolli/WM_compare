"""
informed/clips.py -- the frozen 50-clip set for the music sweep.

Selection is copied from `cascade/emilia_bench.py:stage_select()` -- same
filters, same seed, same speaker-stratified round robin -- so this run uses the
SAME 50 clips as the cascade benchmark and results stay comparable.

WHY COPIED RATHER THAN IMPORTED

`emilia_bench` loads watermark adapters at import time to read payload lengths.
This file only needs a clip list, and must run on a login node in two seconds
without touching a model. The constants below are asserted against
`emilia_bench`'s when it can be imported cheaply, so drift is caught rather than
assumed away.

WHY THE +4.9 dB NUMBER NEEDS THIS

`real_attacks_experiment.py` uses ONE LibriVox clip and ONE music track, with no
trial loop -- despite `real_attacks_summary.csv` labelling its rows "mean over
trials". Masking depends heavily on what the speech and the music are, so a
single pair cannot establish a threshold. 50 clips is the fix.

Usage:
    python clips.py                    # writes clips.json
    python clips.py --n 50 --seed 1234
    EMILIA_CSV=/path/to/manifest.csv python clips.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

# Verbatim from cascade/emilia_bench.py -- do not "improve" these independently.
N_CLIPS = 50
SECONDS = 10.0
DNSMOS_MIN = 3.0
SEED = 1234

EMILIA_CSV = os.environ.get(
    "EMILIA_CSV",
    "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv",
)

OUT_JSON = os.path.join(HERE, "clips.json")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def _check_constants():
    """Fail loudly if emilia_bench's constants have moved away from ours."""
    try:
        src = open(os.path.join(ROOT, "cascade", "emilia_bench.py")).read()
    except Exception:
        print("  (could not read emilia_bench.py to verify constants)")
        return
    want = {"N_CLIPS": N_CLIPS, "SECONDS": SECONDS,
            "DNSMOS_MIN": DNSMOS_MIN, "SEED": SEED}
    for name, ours in want.items():
        for line in src.splitlines():
            s = line.strip()
            if s.startswith(name) and "=" in s:
                theirs = s.split("=", 1)[1].split("#")[0].strip()
                try:
                    same = float(theirs) == float(ours)
                except ValueError:
                    same = False
                if not same:
                    raise SystemExit(
                        f"CONSTANT DRIFT: emilia_bench has {name}={theirs}, "
                        f"clips.py has {ours}. The two runs would no longer use "
                        f"the same clips. Reconcile before running.")
                break
    print("  constants match cascade/emilia_bench.py")


def select(csv_path, n, seed, seconds, dnsmos_min):
    import pandas as pd

    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Emilia manifest not found: {csv_path}\n"
            f"Set EMILIA_CSV to the live path on the cluster.")

    df = pd.read_csv(csv_path)
    print(f"manifest rows: {len(df)}   columns: {list(df.columns)}")

    if "dataset" in df.columns:
        df = df[df["dataset"] == "emilia"].copy()
        print(f"  after dataset=='emilia': {len(df)}")

    for col in ("dnsmos", "duration_s", "speaker"):
        if col not in df.columns:
            raise SystemExit(f"manifest has no '{col}' column; cannot reproduce "
                             f"emilia_bench's selection")

    df = df[(df["dnsmos"] >= dnsmos_min) & (df["duration_s"] >= seconds)].copy()
    print(f"  after dnsmos>={dnsmos_min} and duration>={seconds}s: {len(df)}")
    if len(df) < n:
        raise SystemExit(f"only {len(df)} clips pass filters; need {n}")

    # speaker-stratified round robin -- maximise voice diversity
    rng = random.Random(seed)
    by_spk = {}
    for _, r in df.sample(frac=1.0, random_state=seed).iterrows():
        by_spk.setdefault(r["speaker"], []).append(r)
    order = list(by_spk)
    rng.shuffle(order)

    picked, i = [], 0
    while len(picked) < n and any(by_spk.values()):
        spk = order[i % len(order)]
        i += 1
        if by_spk[spk]:
            picked.append(by_spk[spk].pop())

    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} clips after stratification; need {n}")
    return picked[:n]


def main(argv):
    n = get_arg(argv, "--n", N_CLIPS, int)
    seed = get_arg(argv, "--seed", SEED, int)
    csv_path = get_arg(argv, "--csv", EMILIA_CSV)

    print("verifying selection constants...")
    _check_constants()

    picked = select(csv_path, n, seed, SECONDS, DNSMOS_MIN)

    clips = []
    for i, r in enumerate(picked):
        clips.append({
            "clip_id": f"c{i:03d}",
            "path": str(r["path"]) if "path" in r else str(r.iloc[0]),
            "speaker": str(r["speaker"]),
            "duration_s": float(r["duration_s"]),
            # the manifest's own DNSMOS for the SOURCE clip -- the ceiling any
            # mix can approach, and the baseline every degradation is read against
            "dnsmos_source": float(r["dnsmos"]),
        })

    speakers = {c["speaker"] for c in clips}
    durs = [c["duration_s"] for c in clips]
    dns = [c["dnsmos_source"] for c in clips]

    out = {
        "manifest": csv_path,
        "seed": seed,
        "n_clips": len(clips),
        "filters": {"dnsmos_min": DNSMOS_MIN, "min_duration_s": SECONDS},
        "selection_source": "cascade/emilia_bench.py:stage_select (copied)",
        "clips": clips,
        "stats": {
            "n_speakers": len(speakers),
            "duration_min_s": min(durs), "duration_max_s": max(durs),
            "dnsmos_source_min": min(dns), "dnsmos_source_max": max(dns),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nselected {len(clips)} clips from {len(speakers)} speakers")
    print(f"  duration      {min(durs):.1f}-{max(durs):.1f} s")
    print(f"  source DNSMOS {min(dns):.2f}-{max(dns):.2f}  "
          f"(all >= {DNSMOS_MIN} by construction)")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
