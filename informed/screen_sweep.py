"""
informed/screen_sweep.py -- Phase A: the full 27-attack boundary screen.

For every attack, at every strength: how much quality is left, and does
detection survive? The output is a detection-vs-quality curve per attack, and
the attacks whose curves fall below the detection threshold while still above
the usability floor are the vulnerabilities Phase B should target.

RELATIONSHIP TO music_sweep.py

`music_sweep.py` measures ONE attack (music bed) on a fine 14-point SNR grid, to
locate its crossing precisely. This file measures ALL 27 on coarser grids, to
place them relative to each other. They share `quality.py` and `clips.py`; the
row loops are separate because the axes differ (SNR vs attack x strength).

Run the music sweep first -- it is decisive on its own and cheap. Run this when
you want the whole picture.

TWO ARMS

  wm     watermarked speech, attacked   -- the measurement
  unwm   clean speech, attacked         -- the control

The control answers "did the attack damage the audio, or did the watermark?"
`2026-08-17_attack-damage-control` measured the watermark's own cost at 0.06
PESQ over 15 conditions; the same question is asked here for all 27.

SKIPS ARE NOT FAILURES

An attack that cannot run (no ffmpeg, no transformers, no noise asset) is
recorded with ok=0 and a reason. It must never be read as the watermark
surviving, or as the watermark failing.

Usage:
    python screen_sweep.py                          # all clips, all attacks
    python screen_sweep.py --clip 0                 # one clip
    python screen_sweep.py --attacks music_bed,reverb,denoise
    python screen_sweep.py --categories additive,acoustic
    # under Slurm, SLURM_ARRAY_TASK_ID selects the clip
"""
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                   # noqa: E402
import quality as Q                          # noqa: E402

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
CLIPS_JSON = os.path.join(HERE, "clips.json")

CLIP_SECONDS = 10.0                          # matches cascade/emilia_bench.py
DETECT_THRESHOLD = 0.50                      # results/THRESHOLD_DECISION.md

FIELDS = [
    "clip_id", "speaker", "attack", "category", "param", "arm",
    "conf", "bit_acc", "detected",
    "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "nr_backend",
    "pesq", "stoi", "snr_db_measured", "si_snr_db",
    "alignment_breaking", "ok", "runtime_s", "note",
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


def write_params(clips, attacks):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    backend, backend_note = Q.backend()
    versions = {}
    for mod in ("numpy", "scipy", "soundfile", "speechmos", "pesq", "pystoi",
                "librosa", "torch", "transformers", "pydub"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            versions[mod] = "not installed"
    p = {
        "run": RUN_SLUG,
        "phase": "A screen",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "n_clips": len(clips["clips"]),
        "clip_selection": clips.get("selection_source"),
        "clip_seed": clips.get("seed"),
        "clip_seconds": CLIP_SECONDS,
        "n_attacks": len(attacks),
        "n_configs": sum(len(A.SCREEN_GRID[a]) for a in attacks),
        "attacks": {a: [lbl for lbl, _ in A.SCREEN_GRID[a]] for a in attacks},
        "categories": {a: A.CATEGORY.get(a, "uncategorised") for a in attacks},
        "alignment_breaking": sorted(A.ALIGNMENT_BREAKING),
        "arms": ["wm", "unwm"],
        "detect_threshold": DETECT_THRESHOLD,
        "quality_backend": backend,
        "quality_backend_note": backend_note,
        "dnsmos_floor": Q.DNSMOS_FLOOR,
        "python": platform.python_version(),
        "packages": versions,
    }
    with open(os.path.join(RESULTS_DIR, "params_screen.json"), "w") as f:
        json.dump(p, f, indent=2)
    return p


def run_clip(clip, attacks, adapter, cl, writer, verbose=True):
    org = cl.read_wav(clip["path"])
    n_keep = int(CLIP_SECONDS * cl.SR_MASTER)
    if len(org) < n_keep:
        print(f"  clip shorter than {CLIP_SECONDS}s -- skipping")
        return 0, 0
    org = org[:n_keep].astype("float32")

    t0 = time.time()
    wm = np.asarray(adapter.embed(org), dtype="float32")[:n_keep]
    if verbose:
        c0, _, b0 = adapter.detect(wm)
        print(f"  embedded in {time.time()-t0:.1f}s   baseline "
              f"conf={c0:.4f} bit_acc={b0:.4f}")

    tmpdir = tempfile.mkdtemp(prefix="screen_")
    org_path = os.path.join(tmpdir, "org.wav")
    cl.write_wav(org_path, org)

    n_rows = n_skip = 0
    for name in attacks:
        cat = A.CATEGORY.get(name, "uncategorised")
        breaks_align = int(name in A.ALIGNMENT_BREAKING)
        n_ok_here = 0

        for label, param in A.SCREEN_GRID[name]:
            for arm, speech in (("wm", wm), ("unwm", org)):
                t1 = time.time()
                note = ""
                z = A.apply(name, param, speech, cl.SR_MASTER)

                if z is None or not len(np.asarray(z)):
                    n_skip += 1
                    writer.writerow({
                        "clip_id": clip["clip_id"], "speaker": clip["speaker"],
                        "attack": name, "category": cat, "param": label,
                        "arm": arm, "conf": "", "bit_acc": "", "detected": "",
                        "dnsmos_ovrl": "", "dnsmos_sig": "", "dnsmos_bak": "",
                        "nr_backend": "", "pesq": "", "stoi": "",
                        "snr_db_measured": "", "si_snr_db": "",
                        "alignment_breaking": breaks_align, "ok": 0,
                        "runtime_s": round(time.time() - t1, 3),
                        "note": "attack unavailable (dependency or asset) -- "
                                "NOT a watermark result",
                    })
                    continue

                z = np.asarray(z, dtype="float32")
                deg_path = os.path.join(tmpdir, "deg.wav")
                cl.write_wav(deg_path, z)

                try:
                    conf, _bits, bacc = adapter.detect(z)
                except Exception as e:
                    conf, bacc = float("nan"), float("nan")
                    note = f"detect failed: {type(e).__name__}"

                q = Q.score(org_path, deg_path, deg_audio=z, sr=cl.SR_MASTER)

                writer.writerow({
                    "clip_id": clip["clip_id"], "speaker": clip["speaker"],
                    "attack": name, "category": cat, "param": label, "arm": arm,
                    "conf": "" if not np.isfinite(conf) else round(float(conf), 4),
                    "bit_acc": "" if not np.isfinite(bacc) else round(float(bacc), 4),
                    "detected": int(np.isfinite(conf) and conf >= DETECT_THRESHOLD),
                    "dnsmos_ovrl": q["dnsmos_ovrl"], "dnsmos_sig": q["dnsmos_sig"],
                    "dnsmos_bak": q["dnsmos_bak"], "nr_backend": q["nr_backend"],
                    "pesq": q["pesq"], "stoi": q["stoi"],
                    "snr_db_measured": q["snr_db"], "si_snr_db": q["si_snr_db"],
                    "alignment_breaking": breaks_align, "ok": 1,
                    "runtime_s": round(time.time() - t1, 3), "note": note,
                })
                n_rows += 1
                n_ok_here += 1
                try:
                    os.remove(deg_path)
                except OSError:
                    pass

        if verbose:
            state = "ok" if n_ok_here else "ALL SKIPPED"
            print(f"    {name:22s} {cat:12s} {n_ok_here:>3d} rows  {state}")

    try:
        os.remove(org_path)
        os.rmdir(tmpdir)
    except OSError:
        pass
    return n_rows, n_skip


def main(argv):
    if not os.path.exists(CLIPS_JSON):
        raise SystemExit(f"{CLIPS_JSON} missing -- run clips.py first")
    with open(CLIPS_JSON) as f:
        clips = json.load(f)
    all_clips = clips["clips"]

    backend, note = Q.backend()
    print(f"quality backend: {backend}\n  {note}")
    if backend == "none":
        raise SystemExit(
            "\nNo no-reference scorer. The quality axis IS the screen, so there "
            "is nothing to measure.\n    pip install -r requirements_extra.txt")
    if backend.startswith("squim") and "--force" not in argv:
        raise SystemExit(
            "\nRefusing to run on the unvalidated SQUIM fallback -- it scored "
            "noisy audio higher than clean on both signals it could be tested "
            "against, and its scale is not comparable to emilia_bench's 3.0 "
            "filter. Install speechmos, or validate with `python quality.py` "
            "and pass --force.")

    attacks = get_list(argv, "--attacks", sorted(A.SCREEN_GRID))
    if "--categories" in argv:
        want = set(get_list(argv, "--categories", []))
        attacks = [a for a in attacks if A.CATEGORY.get(a) in want]
    unknown = [a for a in attacks if a not in A.SCREEN_GRID]
    if unknown:
        raise SystemExit(f"unknown attacks: {unknown}\n"
                         f"available: {sorted(A.SCREEN_GRID)}")

    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if "--clip" in argv:
        indices = [get_arg(argv, "--clip", 0, int)]
    elif task is not None:
        indices = [int(task)]
    else:
        indices = list(range(len(all_clips)))

    import cascade_lib as cl
    print("\nloading AWARE adapter...")
    adapter = cl.get_adapter("aware")

    n_cfg = sum(len(A.SCREEN_GRID[a]) for a in attacks)
    print(f"{len(attacks)} attacks, {n_cfg} configs, 2 arms "
          f"-> {n_cfg * 2} rows per clip")

    os.makedirs(DATA_DIR, exist_ok=True)
    write_params(clips, attacks)

    for ci in indices:
        if ci >= len(all_clips):
            print(f"clip index {ci} out of range ({len(all_clips)})")
            continue
        clip = all_clips[ci]
        out_csv = os.path.join(DATA_DIR, f"screen_clip{ci:03d}.csv")
        print(f"\nclip {ci} [{clip['clip_id']}] {os.path.basename(clip['path'])}")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n, ns = run_clip(clip, attacks, adapter, cl, w)
        print(f"  {n} rows, {ns} skipped, {time.time()-t0:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
