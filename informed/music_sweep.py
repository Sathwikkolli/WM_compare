"""
informed/music_sweep.py -- Phase A-1: locate the quality/detection crossing.

We know AWARE's confidence crosses 0.50 near +4.9 dB speech-to-music SNR
(real_attacks_summary.csv). We do NOT know where the audio stops being usable.
Without both numbers the crossing cannot be located, and the informed-detection
plan's lead candidate is unsupported.

One Slurm array task per clip. Each task embeds once, then sweeps SNR x music
x arm, writing ONE CSV. score_music_sweep.py concatenates them.

WHAT THIS FIXES ABOUT THE EXISTING EVIDENCE

  n = 1        real_attacks_experiment.py uses one LibriVox clip and one music
               track, with no trial loop, despite its CSV saying "mean over
               trials". Here: 50 speaker-stratified Emilia clips.
  no quality   that script measures confidence and bit accuracy only. Here:
               DNSMOS (no-reference, primary) and PESQ (reference, secondary).

THE TWO ARMS

  wm     watermarked speech + music   -- the measurement
  unwm   clean speech + music         -- the control

The SAME music gain is applied to both arms, computed from the CLEAN speech
power, so the arms differ only by the watermark. 2026-08-17_attack-damage-control
found the watermark itself costs only 0.06 PESQ; if that holds here the music is
doing all the damage and the control can be dropped from later phases.

Usage:
    python music_sweep.py                     # all clips (local)
    python music_sweep.py --clip 3            # one clip
    python music_sweep.py --clips 0-9 --music all    # the diversity probe
    python music_sweep.py --save-audio        # keep the mixes for listening
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

import quality as Q                       # noqa: E402

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
CLIPS_JSON = os.path.join(HERE, "clips.json")
AUDIO_DIR = os.path.join(BASE, "real_audio")

# Fixed clip length, matching cascade/emilia_bench.py's SECONDS. Comparable
# quality scores need comparable durations.
CLIP_SECONDS = 10.0

# Fine spacing where the crossing is expected. The 3.8 dB point from
# real_attacks_experiment.py is deliberately DROPPED -- it was chosen to bracket
# an n=1 result and would bias a 50-clip re-measurement toward confirming it.
SNRS = [20, 15, 10, 8, 6, 5, 4, 3, 2, 1, 0, -3, -6, -10]

DETECT_THRESHOLD = 0.50                   # results/THRESHOLD_DECISION.md

# The canonical track is the one real_attacks_experiment.py used, so this run
# stays comparable to real_attacks_summary.csv. The other three exist only for
# the diversity probe: one track's spectrum may mask AWARE's bands unusually
# well or badly, and if the crossing moves with the track then the main number
# is track-specific and must be reported that way.
MUSIC_SOURCES = {
    "march": "https://archive.org/download/MarchForHonor/March_For_Honor.mp3",
    # Add three contrasting tracks (orchestral / electronic / sparse acoustic)
    # before running --music all. Left empty rather than guessed at: a broken
    # URL discovered inside a 50-task array is far more expensive than one
    # discovered here.
}

FIELDS = [
    "clip_id", "speaker", "clip_dur_s", "music", "snr_db", "arm",
    "conf", "bit_acc", "detected",
    "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_clean", "nr_backend",
    "pesq", "stoi", "snr_db_measured", "si_snr_db",
    "dnsmos_source", "runtime_s", "note",
]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def parse_range(spec, hi):
    """'3' -> [3];  '0-9' -> [0..9];  '1,4,7' -> [1,4,7]."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < hi]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def fetch(url, dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  downloading {os.path.basename(dst)} ...")
    subprocess.run(["curl", "-L", "-s", "-o", dst, url], check=True)
    return dst


def load_music(name, cl):
    """Mono music at SR_MASTER. Raises with a clear message if unavailable."""
    url = MUSIC_SOURCES.get(name)
    if not url:
        raise SystemExit(
            f"music source '{name}' has no URL in MUSIC_SOURCES. "
            f"Add it before running the diversity probe.")
    path = fetch(url, os.path.join(AUDIO_DIR, f"music_{name}.mp3"))
    return cl.read_wav(path)


def write_params(clips, musics, methods_note):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    backend, backend_note = Q.backend()
    versions = {}
    for mod in ("numpy", "scipy", "soundfile", "speechmos", "pesq", "pystoi",
                "torch", "torchaudio"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            versions[mod] = "not installed"
    p = {
        "run": RUN_SLUG,
        "phase": "A-1 music sweep",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "n_clips": len(clips["clips"]),
        "clip_selection": clips.get("selection_source"),
        "clip_seed": clips.get("seed"),
        "clip_seconds": CLIP_SECONDS,
        "snrs_db": SNRS,
        "music": musics,
        "arms": ["wm", "unwm"],
        "detect_threshold": DETECT_THRESHOLD,
        "quality_backend": backend,
        "quality_backend_note": backend_note,
        "dnsmos_floor": Q.DNSMOS_FLOOR,
        "dnsmos_floor_strict": Q.DNSMOS_FLOOR_STRICT,
        "notes": methods_note,
        "python": platform.python_version(),
        "packages": versions,
    }
    with open(os.path.join(RESULTS_DIR, "params.json"), "w") as f:
        json.dump(p, f, indent=2)
    return p


def mix_at_snr(speech, music, snr_db, ref_power):
    """speech + scaled music at the requested SNR.

    `ref_power` is the CLEAN speech power and is passed in rather than measured
    per arm, so both arms get the identical music level and differ only by the
    watermark.
    """
    pm = float(np.mean(music ** 2)) + 1e-20
    a = float(np.sqrt(ref_power / (pm * (10.0 ** (snr_db / 10.0)))))
    return (speech + a * music).astype("float32")


def run_clip(clip, musics, adapter, cl, writer, save_audio=False, verbose=True):
    org = cl.read_wav(clip["path"])
    n_keep = int(CLIP_SECONDS * cl.SR_MASTER)
    if len(org) < n_keep:
        print(f"  clip shorter than {CLIP_SECONDS}s -- skipping")
        return 0
    org = org[:n_keep].astype("float32")
    ref_power = float(np.mean(org ** 2))

    # This clip's own clean score. The usability floor is a DROP from this, not
    # an absolute number -- clean Emilia clips measure 2.86-3.36, so an absolute
    # floor would fail undamaged audio. See quality.py's constants block.
    dnsmos_clean = Q.no_reference(org, cl.SR_MASTER)["ovrl"]
    if verbose:
        print(f"  clean DNSMOS = {dnsmos_clean}")

    t0 = time.time()
    wm = np.asarray(adapter.embed(org), dtype="float32")[:n_keep]
    if verbose:
        c0, _, b0 = adapter.detect(wm)
        print(f"  embedded in {time.time()-t0:.1f}s   baseline "
              f"conf={c0:.4f} bit_acc={b0:.4f}")

    tmpdir = tempfile.mkdtemp(prefix="musicsweep_")
    keep_dir = os.path.join(AUDIO_DIR, "mixes", clip["clip_id"])
    if save_audio:
        os.makedirs(keep_dir, exist_ok=True)

    # PESQ reference is the CLEAN speech: "how far has this moved from clean
    # speech", which is the question PESQ actually answers.
    org_path = os.path.join(tmpdir, "org.wav")
    cl.write_wav(org_path, org)

    n_rows = 0
    for mname, music in musics.items():
        m = music
        if len(m) < len(org):
            m = np.tile(m, int(np.ceil(len(org) / len(m))))
        m = m[:len(org)].astype("float32")

        for snr in SNRS:
            for arm, speech in (("wm", wm), ("unwm", org)):
                t1 = time.time()
                note = ""
                mix = mix_at_snr(speech, m, snr, ref_power)

                mix_path = os.path.join(tmpdir, f"{arm}_{mname}_{snr:+05.1f}.wav")
                cl.write_wav(mix_path, mix)

                try:
                    conf, _bits, bacc = adapter.detect(mix)
                except Exception as e:
                    conf, bacc = float("nan"), float("nan")
                    note = f"detect failed: {type(e).__name__}"

                q = Q.score(org_path, mix_path, deg_audio=mix, sr=cl.SR_MASTER)

                writer.writerow({
                    "clip_id": clip["clip_id"],
                    "speaker": clip["speaker"],
                    "clip_dur_s": round(len(org) / cl.SR_MASTER, 2),
                    "music": mname,
                    "snr_db": snr,
                    "arm": arm,
                    "conf": "" if not np.isfinite(conf) else round(float(conf), 4),
                    "bit_acc": "" if not np.isfinite(bacc) else round(float(bacc), 4),
                    "detected": int(np.isfinite(conf) and conf >= DETECT_THRESHOLD),
                    "dnsmos_ovrl": q["dnsmos_ovrl"],
                    "dnsmos_sig": q["dnsmos_sig"],
                    "dnsmos_bak": q["dnsmos_bak"],
                    "dnsmos_clean": dnsmos_clean,
                    "nr_backend": q["nr_backend"],
                    "pesq": q["pesq"],
                    "stoi": q["stoi"],
                    "snr_db_measured": q["snr_db"],
                    "si_snr_db": q["si_snr_db"],
                    "dnsmos_source": round(clip.get("dnsmos_source", float("nan")), 3),
                    "runtime_s": round(time.time() - t1, 3),
                    "note": note,
                })
                n_rows += 1

                if save_audio:
                    os.replace(mix_path, os.path.join(
                        keep_dir, f"{arm}_{mname}_{snr:+05.1f}dB.wav"))
                else:
                    try:
                        os.remove(mix_path)
                    except OSError:
                        pass

        if verbose:
            print(f"    music={mname:10s} {len(SNRS)*2} rows")

    for p in (org_path,):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    return n_rows


def main(argv):
    if not os.path.exists(CLIPS_JSON):
        raise SystemExit(f"{CLIPS_JSON} missing -- run clips.py first")
    with open(CLIPS_JSON) as f:
        clips = json.load(f)
    all_clips = clips["clips"]

    # ---- the quality backend gate ----------------------------------------
    backend, note = Q.backend()
    print(f"quality backend: {backend}")
    print(f"  {note}")
    if backend == "none":
        raise SystemExit(
            "\nNo no-reference scorer. The whole point of this sweep is the "
            "quality axis, so there is nothing to measure.\n"
            "    pip install --dry-run -r requirements_extra.txt\n"
            "    pip install -r requirements_extra.txt")
    if backend.startswith("squim") and "--force" not in argv:
        raise SystemExit(
            "\nRefusing to run on the SQUIM fallback.\n"
            "It is unvalidated: on the signals it could be checked against it "
            "scored noisy audio HIGHER than clean, and its scale is not "
            "comparable to the Emilia manifest or to emilia_bench's 3.0 filter, "
            "which is what makes our usability floor defensible.\n"
            "Install speechmos, or run `python quality.py` to validate SQUIM on "
            "this machine and pass --force if it passes.")

    which = get_arg(argv, "--music", "march")
    music_names = list(MUSIC_SOURCES) if which == "all" else [w for w in which.split(",")]

    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if "--clip" in argv:
        indices = [get_arg(argv, "--clip", 0, int)]
    elif "--clips" in argv:
        indices = parse_range(get_arg(argv, "--clips", "0", str), len(all_clips))
    elif task is not None:
        indices = [int(task)]
    else:
        indices = list(range(len(all_clips)))

    import cascade_lib as cl
    print("\nloading AWARE adapter...")
    adapter = cl.get_adapter("aware")

    print(f"loading music: {music_names}")
    musics = {n: load_music(n, cl) for n in music_names}

    os.makedirs(DATA_DIR, exist_ok=True)
    write_params(clips, music_names,
                 "PESQ reference is the clean unwatermarked speech. Both arms "
                 "share one music gain, computed from clean speech power.")

    save_audio = "--save-audio" in argv
    for ci in indices:
        if ci >= len(all_clips):
            print(f"clip index {ci} out of range ({len(all_clips)})")
            continue
        clip = all_clips[ci]
        out_csv = os.path.join(DATA_DIR, f"music_clip{ci:03d}.csv")
        print(f"\nclip {ci} [{clip['clip_id']}] {os.path.basename(clip['path'])}")
        t0 = time.time()
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n = run_clip(clip, musics, adapter, cl, w, save_audio=save_audio)
        print(f"  {n} rows, {time.time()-t0:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
