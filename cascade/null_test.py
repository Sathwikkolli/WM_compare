"""
null_test.py -- what do the DETECTORS score on audio carrying no watermark?

The question this answers
------------------------
Production decides "watermarked" when a detector clears 0.5. That number came
from upstream defaults (AudioSeal's documented threshold; AWARE's sigmoid is
centred at x0=0.041, so 0.5 is its midpoint). Nobody has checked that it
transfers to *this* cascade, on *this* audio. One observation says clean speech
scores ~0.4-0.55, another says 0.049-0.175. This measures it.

Scope
-----
Only AudioSeal and AWARE are loaded. Timbre has no detector -- its adapter
returns bit-accuracy in the confidence slot, measured against whatever message
the process embedded last -- so it cannot answer "is a watermark present" and is
excluded. That also means this script needs no Timbre repo and no checkpoint.

Nothing here embeds anything. Every input must come back below threshold; any
file that clears it is a false positive.

Usage
-----
    python null_test.py --dir ../audio/clean_set
    python null_test.py --dir ../audio/clean_set --emilia 300
    python null_test.py --emilia 300 --out results_null.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cascade_lib as cl  # noqa: E402

# Only models with a real presence detector.
MODELS = ("audioseal", "aware")

# Upstream defaults, and the more conservative value proposed after clean speech
# was observed in the 0.4-0.55 range. Both reported so the data picks.
THRESHOLDS = {"official": {"audioseal": 0.50, "aware": 0.50},
              "conservative": {"audioseal": 0.65, "aware": 0.55}}

_PATH_KEYS = ("path", "audio_path", "wav", "filepath", "file", "audio", "filename")
_DUR_KEYS = ("duration", "dur", "length", "seconds", "duration_s")

EMILIA_CSV = os.environ.get(
    "EMILIA_CSV",
    "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv",
)


def _pick(row, keys):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return None


def emilia_paths(n, min_dur=5.0, seed=0):
    """N clean Emilia clips. Never watermarked -- it is a source corpus."""
    import random

    if not os.path.exists(EMILIA_CSV):
        print(f"  ! Emilia manifest not found: {EMILIA_CSV}", file=sys.stderr)
        return []
    rows = []
    with open(EMILIA_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            p = _pick(row, _PATH_KEYS)
            if not p:
                continue
            d = _pick(row, _DUR_KEYS)
            try:
                if d is not None and float(d) < min_dur:
                    continue
            except ValueError:
                pass
            rows.append(p)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def group_of(path):
    """Provenance matters: results are reported per group, not pooled blindly."""
    name = os.path.basename(path).lower()
    if "/emilia" in path.replace("\\", "/").lower() or name.startswith("emilia"):
        return "emilia"
    if name.startswith(("clean_10", "clean_11", "clean_12")):
        return "synthetic"          # clean by construction
    if name.startswith(("clean_01", "clean_02", "clean_03",
                        "clean_04", "clean_05", "clean_06")):
        return "osr_speech"         # Open Speech Repository, public domain
    if name.startswith("clean_09"):
        return "non_speech"
    return "uncertain"              # probable originals, provenance not proven


def score_file(path, adapters):
    """(scores per model, duration). Raises on unreadable input."""
    y = cl.read_wav(path, sr=cl.SR_MASTER)
    out = {}
    for m in MODELS:
        conf, bits, _acc = adapters[m].detect(y)
        out[m] = float(conf)
    return out, len(y) / float(cl.SR_MASTER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of clean audio files")
    ap.add_argument("--emilia", type=int, default=0, help="how many Emilia clips to add")
    ap.add_argument("--repeat", type=int, default=1, help="passes per file (determinism check)")
    ap.add_argument("--out", default="null_test_results.csv")
    args = ap.parse_args()

    paths = []
    if args.dir:
        for f in sorted(os.listdir(args.dir)):
            if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")):
                paths.append(os.path.join(args.dir, f))
    if args.emilia:
        paths += emilia_paths(args.emilia)
    if not paths:
        sys.exit("no input files -- pass --dir and/or --emilia")

    print(f"loading detectors: {', '.join(MODELS)}  (Timbre excluded -- no detector)")
    adapters = {}
    for m in MODELS:
        t0 = time.time()
        adapters[m] = cl.get_adapter(m)
        print(f"  {m:10s} loaded in {time.time()-t0:.1f}s")

    print(f"\nscoring {len(paths)} files x {args.repeat} pass(es)\n")
    rows, failures = [], []
    for i, p in enumerate(paths, 1):
        per_pass = []
        try:
            for _ in range(args.repeat):
                scores, dur = score_file(p, adapters)
                per_pass.append(scores)
        except Exception as exc:                       # noqa: BLE001
            failures.append((p, repr(exc)))
            print(f"  [{i}/{len(paths)}] ERROR {os.path.basename(p)}: {exc}")
            continue

        scores = per_pass[0]
        stable = all(pp == scores for pp in per_pass)
        row = {"file": os.path.basename(p), "group": group_of(p),
               "duration_s": round(dur, 1), "stable": stable}
        row.update({f"{m}_conf": round(scores[m], 6) for m in MODELS})
        for label, thr in THRESHOLDS.items():
            row[f"detected_{label}"] = any(scores[m] >= thr[m] for m in MODELS)
        rows.append(row)

        flag = ""
        if row["detected_official"]:
            flag = "  <-- FALSE POSITIVE @0.5"
        if not stable:
            flag += "  <-- UNSTABLE across passes"
        if i <= 40 or flag:
            print(f"  [{i}/{len(paths)}] {row['file'][:38]:38s} "
                  + "  ".join(f"{m}={scores[m]:.4f}" for m in MODELS) + flag)

    if not rows:
        sys.exit("every file failed -- see errors above")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{len(rows)} clean files scored" + (f", {len(failures)} failed" if failures else ""))
    print("=" * 78)

    groups = sorted({r["group"] for r in rows})
    for m in MODELS:
        print(f"\n{m} confidence on clean audio:")
        print(f"  {'group':14s} {'n':>4s} {'min':>8s} {'median':>8s} {'p95':>8s} {'max':>8s}")
        for g in groups:
            vals = sorted(r[f"{m}_conf"] for r in rows if r["group"] == g)
            if not vals:
                continue
            p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
            print(f"  {g:14s} {len(vals):>4d} {vals[0]:>8.4f} "
                  f"{statistics.median(vals):>8.4f} {p95:>8.4f} {vals[-1]:>8.4f}")
        allv = sorted(r[f"{m}_conf"] for r in rows)
        p95 = allv[min(len(allv) - 1, int(0.95 * len(allv)))]
        print(f"  {'ALL':14s} {len(allv):>4d} {allv[0]:>8.4f} "
              f"{statistics.median(allv):>8.4f} {p95:>8.4f} {allv[-1]:>8.4f}")

    print("\nfalse positives (a clean file called watermarked):")
    for label, thr in THRESHOLDS.items():
        fp = [r for r in rows if r[f"detected_{label}"]]
        pct = 100 * len(fp) / len(rows)
        print(f"  {label:14s} {thr}  ->  {len(fp):>4d}/{len(rows)}  ({pct:.1f}%)")
        for r in fp[:8]:
            print(f"       {r['file'][:40]:40s} "
                  + "  ".join(f"{m}={r[f'{m}_conf']:.4f}" for m in MODELS))

    unstable = [r for r in rows if not r["stable"]]
    if unstable:
        print(f"\n! {len(unstable)} file(s) scored differently across passes -- "
              f"detector is not deterministic, calibration is unsafe until fixed")

    # Headroom is the decision input: passing with the top score at 0.49 is not
    # the same as passing with it at 0.05.
    print("\nheadroom below the 0.5 line:")
    for m in MODELS:
        worst = max(r[f"{m}_conf"] for r in rows)
        print(f"  {m:10s} worst clean score {worst:.4f}  ->  margin {0.50 - worst:+.4f}")

    if failures:
        print(f"\nfailed files ({len(failures)}):")
        for p, e in failures[:10]:
            print(f"  {os.path.basename(p)}: {e}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
