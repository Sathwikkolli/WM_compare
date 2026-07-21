"""
fsss/exp_a_repeatability.py -- Experiment A: FSSS salient-point repeatability.

Measures whether FSSS energy-ratio salient points land in the SAME places before
and after attacks. This is the experiment the FSSS paper DEFINED ("a good salient
point extraction method extracts the same points before and after common signal
manipulations") but never actually ran. It gates the whole content-anchored
hopping design: if anchors don't survive our attacks, hopping tied to them can't.

Two tracks:
  * Track 1 (statistical): the locked VoxWatermark attack subset applied
    programmatically to 30 Emilia clips -> the headline hit-rate/jitter table.
  * Track 2 (case study): the real METAPXYL Pro-Tools mastering stages (declick,
    C-Vox, reverb, limiter, music bed, resample, MP3) on the one client episode.
    Real DAW processing, n=1, ecologically valid.

Runs on Great Lakes (needs the Emilia data, the vox_attacks deps, and the client
Dropbox). Working sample rate = 16 kHz (the rate AWARE -- and thus the anchors --
will actually operate at). Matching tolerance w = +/- 20 ms.

Usage (repo root, wmcompare env):
    python -m fsss.exp_a_repeatability                 # both tracks
    python -m fsss.exp_a_repeatability --track1        # Emilia/Vox only
    python -m fsss.exp_a_repeatability --track2        # METAPXYL only
    python -m fsss.exp_a_repeatability --n 10          # fewer clips (dev)
    python -m fsss.exp_a_repeatability --csv /path/to/emilia.csv
"""

import os
import sys
import csv
import glob
import subprocess
import numpy as np

# --- paths / imports -------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)                     # so `fsss` package imports resolve
sys.path.insert(0, os.path.join(BASE, "cascade"))   # vox_attacks lives here

try:
    from fsss.salient import find_salient_points
    from fsss.match import align_lag, match_points
except ImportError:                          # running as a plain script inside fsss/
    from salient import find_salient_points
    from match import align_lag, match_points

import vox_attacks   # cascade/vox_attacks.py (VOX_GRID, apply, strength_x)

# --- config ----------------------------------------------------------------- #
WORK_SR = 16000
W_MS = 20.0
N_CLIPS = 30
MIN_DUR = 9.0
EMILIA_CSV = os.environ.get(
    "EMILIA_CSV",
    "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv",
)
CLIENT_DIR = os.path.join(BASE, "client_processed")
DROPBOX_ZIP = ("https://www.dropbox.com/scl/fo/o1adko6sziyqo5i0n2hfs/"
               "AJDnEsqZyOWV4Qu07LbZQhM?rlkey=h6juuyn1crmvntc04jutv1uql&dl=1")
OUT_DIR = os.path.join(BASE, "fsss_out")

# Locked Exp-A Vox subset (the attacks that most threaten salient-point position).
LOCKED_ATTACKS = [
    "time_stretch", "time_jitter", "dynamic_compression", "echo",
    "gaussian_noise", "background_noise", "lowpass", "mp3", "encodec",
]

ROW_FIELDS = [
    "track", "clip", "attack", "param", "strength_x",
    "hit_rate", "false_alarm", "median_jitter_ms",
    "n_clean", "n_attacked", "n_matched", "lag_ms", "status",
]


# --- helpers ---------------------------------------------------------------- #
def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def load_16k(path):
    """Load any audio (wav/mp3), downmix to mono, resample to 16 kHz float32."""
    import librosa
    y, _ = librosa.load(path, sr=WORK_SR, mono=True)
    return y.astype("float32")


def pick_clips(csv_path, n):
    """First n Emilia clips at least MIN_DUR long (reuses reverb_test_emilia's rule)."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if float(row["duration_s"]) >= MIN_DUR and os.path.exists(row["path"]):
                    out.append(row["path"])
            except (KeyError, ValueError):
                continue
            if len(out) >= n:
                break
    return out


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


# --- Track 1: Vox attacks x Emilia ------------------------------------------ #
def run_track1(clips, writer, pooled):
    for ci, path in enumerate(clips):
        try:
            y = load_16k(path)
        except Exception as e:
            print(f"[track1] skip {path}: {e}")
            continue
        clean_pts = find_salient_points(y, WORK_SR)
        clip_name = os.path.basename(path)
        print(f"[track1] {ci+1}/{len(clips)} {clip_name}  ({len(clean_pts)} pts)")

        for attack in LOCKED_ATTACKS:
            for label, param in vox_attacks.VOX_GRID.get(attack, []):
                try:
                    y2 = vox_attacks.apply(attack, param, y, WORK_SR)
                except Exception as e:
                    y2 = None
                    print(f"    {attack}/{label}: ERROR {e}")
                if y2 is None:                       # missing dep / unavailable
                    writer.writerow(_row("track1", clip_name, attack, label, param,
                                         status="SKIP"))
                    continue

                att_pts = find_salient_points(y2, WORK_SR)
                if attack == "time_stretch":
                    scale, lag = float(param), 0     # known warp
                else:
                    scale, lag = 1.0, align_lag(y, y2, WORK_SR)
                m = match_points(clean_pts, att_pts, WORK_SR, w_ms=W_MS, scale=scale, lag=lag)
                writer.writerow(_row("track1", clip_name, attack, label, param, m, lag))
                # pool successes/trials per (attack,param) for the summary
                key = (attack, label)
                agg = pooled.setdefault(key, [0, 0, 0])   # matched, clean, n_clips
                agg[0] += m["n_matched"]
                agg[1] += m["n_clean"]
                agg[2] += 1


# --- Track 2: METAPXYL real stages ------------------------------------------ #
def ensure_client_files():
    os.makedirs(CLIENT_DIR, exist_ok=True)
    have = (glob.glob(os.path.join(CLIENT_DIR, "*.wav"))
            + glob.glob(os.path.join(CLIENT_DIR, "*.mp3")))
    if have:
        return sorted(have)
    print("client_processed/ empty -> downloading Dropbox folder...")
    zip_path = os.path.join(CLIENT_DIR, "client_files.zip")
    subprocess.run(["curl", "-L", "-s", "-o", zip_path, DROPBOX_ZIP], check=True)
    subprocess.run(["unzip", "-o", "-j", "-q", zip_path, "-d", CLIENT_DIR])
    return sorted(glob.glob(os.path.join(CLIENT_DIR, "*.wav"))
                  + glob.glob(os.path.join(CLIENT_DIR, "*.mp3")))


def _stage_key(path):
    """Sort key from a filename like '..._Stage 07A.wav'. Lower = earlier/original."""
    b = os.path.basename(path).lower()
    if "original" in b or "stage 0 " in b or "stage 00" in b:
        return -1.0
    i = b.find("stage")
    if i < 0:
        return 999.0
    num = ""
    for ch in b[i + 5:]:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return float(num) if num else 999.0


def run_track2(writer, ref_override=None):
    files = ensure_client_files()
    if not files:
        print("[track2] no client files found; skipping")
        return
    files = sorted(files, key=_stage_key)
    ref_path = ref_override or files[0]
    print(f"[track2] reference = {os.path.basename(ref_path)}")
    print("[track2] NOTE: a single global lag cannot fully align stages with "
          "interior strip-silence cuts; jitter/hit for those is approximate.")
    ref = load_16k(ref_path)
    ref_pts = find_salient_points(ref, WORK_SR)

    for path in files:
        if path == ref_path:
            continue
        stage = os.path.basename(path)
        try:
            att = load_16k(path)
        except Exception as e:
            print(f"[track2] skip {stage}: {e}")
            continue
        att_pts = find_salient_points(att, WORK_SR)
        lag = align_lag(ref, att, WORK_SR)
        m = match_points(ref_pts, att_pts, WORK_SR, w_ms=W_MS, scale=1.0, lag=lag)
        writer.writerow(_row("metapxyl", os.path.basename(ref_path), "stage", stage, stage, m, lag))
        print(f"[track2] {stage:32s} hit={m['hit_rate']:.2f} "
              f"jitter={m['median_jitter_ms']:.1f}ms lag={lag/WORK_SR*1000:.0f}ms")


# --- row builder ------------------------------------------------------------ #
def _row(track, clip, attack, label, param, m=None, lag=0, status="OK"):
    try:
        sx = vox_attacks.strength_x(attack, param)
    except Exception:
        sx = ""
    r = dict(track=track, clip=clip, attack=attack, param=label, strength_x=sx,
             lag_ms=round(lag / WORK_SR * 1000.0, 2), status=status)
    if m is None:
        r.update(hit_rate="", false_alarm="", median_jitter_ms="",
                 n_clean="", n_attacked="", n_matched="")
    else:
        r.update(hit_rate=round(m["hit_rate"], 4) if m["hit_rate"] == m["hit_rate"] else "",
                 false_alarm=round(m["false_alarm"], 4) if m["false_alarm"] == m["false_alarm"] else "",
                 median_jitter_ms=round(m["median_jitter_ms"], 2) if m["median_jitter_ms"] == m["median_jitter_ms"] else "",
                 n_clean=m["n_clean"], n_attacked=m["n_attacked"], n_matched=m["n_matched"])
    return r


# --- main ------------------------------------------------------------------- #
def main(argv):
    do1 = "--track2" not in argv or "--track1" in argv
    do2 = "--track1" not in argv or "--track2" in argv
    if "--track1" in argv and "--track2" not in argv:
        do1, do2 = True, False
    if "--track2" in argv and "--track1" not in argv:
        do1, do2 = False, True

    n_clips = get_arg(argv, "--n", N_CLIPS, int)
    csv_in = get_arg(argv, "--csv", EMILIA_CSV)
    ref = get_arg(argv, "--ref", None)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows_path = os.path.join(OUT_DIR, "exp_a_rows.csv")
    summ_path = os.path.join(OUT_DIR, "exp_a_summary.csv")
    pooled = {}

    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()

        if do1:
            clips = pick_clips(csv_in, n_clips)
            print(f"[track1] {len(clips)} Emilia clips from {csv_in}")
            if not clips:
                print(f"[track1] WARNING: no clips found in {csv_in}")
            run_track1(clips, writer, pooled)

        if do2:
            run_track2(writer, ref_override=ref)

    # summary: pooled hit rate + Wilson CI per (attack, param) over all clips
    if pooled:
        with open(summ_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attack", "param", "n_clips", "matched", "clean",
                        "hit_rate", "wilson_lo", "wilson_hi"])
            for (attack, label), (matched, clean, nclip) in sorted(pooled.items()):
                hr = matched / clean if clean else float("nan")
                lo, hi = wilson(matched, clean)
                w.writerow([attack, label, nclip, matched, clean,
                            round(hr, 4), round(lo, 4), round(hi, 4)])
        print(f"\nwrote {summ_path}")
    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
