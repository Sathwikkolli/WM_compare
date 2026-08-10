"""
ab_aware/run_ab.py -- score the full grid: 40 clips x 11 conditions = 440 rows.

One Slurm array task per clip. Each task writes data/raw_<clip_id>.csv, so a
failed task can be requeued on its own without touching the other 39. Runs
serially over all clips when SLURM_ARRAY_TASK_ID is unset, which is how you test
it on a login node.

WHAT EACH ROW IS: one (clip, condition) decision by the AWARE detector.

    clip_id    wm00..wm19 (positives) | cl00..cl19 (negatives)
    arm        wm | clean            <- the ground-truth label
    condition  clean + the 10 distortions
    conf       detector confidence, the raw score. NOT thresholded here.
    bit_acc    fraction of the 20 embedded bits recovered; empty for negatives,
               which have no payload to compare against
    pesq/snr   perceptual quality vs. the unwatermarked source
    status     ok | skip (attack unavailable) | error

THRESHOLDING IS DEFERRED TO analyze.py, deliberately. Writing a hard
"detected" column here would bake the arbitrary conf>=0.5 cut into the raw data
and make ROC/AUC impossible to compute after the fact. The raw score is the
thing worth keeping.

PESQ READS DIFFERENTLY PER ARM, and conflating them is the easy mistake:
    wm arm, clean condition  -> watermark-only quality. THIS is the number to
                                quote as "what the watermark costs".
    wm arm, attacked         -> watermark AND attack damage, combined.
    clean arm, attacked      -> attack damage alone; the control that says how
                                much of the wm-arm drop is just the attack.
Both desync attacks (time_stretch, crop_50) break sample alignment, so PESQ/SNR
are left empty for them rather than reported wrong -- see attacks_ab.py.

Usage:
    python run_ab.py                      # all 40 clips, serial
    python run_ab.py --clip wm03          # one clip
    SLURM_ARRAY_TASK_ID=7 python run_ab.py
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
sys.path.insert(0, ROOT)

from attacks_ab import ORDER, apply, family, needs_ref_align  # noqa: E402

WORK_SR = 16000
N_BITS = 20
AWARE_MODEL = os.environ.get("AWARE_MODEL", "AWARE")
RUN = os.environ.get("AB_RUN", "2026-08-10_aware-detection-ab")

WORK = os.path.join(HERE, "work")
SRC_DIR = os.path.join(WORK, "src")
WM_DIR = os.path.join(WORK, "wm")
PAYLOADS = os.path.join(WORK, "payloads.json")
DATA_DIR = os.path.join(BASE, "results", RUN, "data")

FIELDS = ["clip_id", "arm", "condition", "family", "conf", "bits",
          "bit_acc", "pesq", "snr_db", "status", "seconds"]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def read_wav(path):
    import soundfile as sf
    y, sr = sf.read(path)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    assert sr == WORK_SR, f"{path} is {sr} Hz, expected {WORK_SR}"
    return np.asarray(y, dtype="float32")


def snr_db(ref, deg):
    n = min(len(ref), len(deg))
    if n == 0:
        return None
    ref, deg = ref[:n], deg[:n]
    noise = deg - ref
    p_n = float((noise ** 2).sum())
    if p_n <= 0:
        return None
    return float(10.0 * np.log10(float((ref ** 2).sum()) / p_n))


def pesq_wb(ref, deg):
    try:
        from pesq import pesq
        n = min(len(ref), len(deg))
        if n < WORK_SR // 2:            # pesq needs a meaningful stretch of audio
            return None
        v = float(pesq(WORK_SR, ref[:n], deg[:n], "wb"))
        return v if v == v else None
    except Exception:
        return None


def score_clip(cid, detector, payloads):
    """Return the 11 rows for one clip."""
    arm = "wm" if cid.startswith("wm") else "clean"
    src = read_wav(os.path.join(SRC_DIR, cid + ".wav"))
    audio = read_wav(os.path.join(WM_DIR, cid + ".wav")) if arm == "wm" else src
    truth = np.asarray(payloads.get(cid, []), dtype=int) if arm == "wm" else None

    from aware.service import detect_watermark

    rows = []
    for cond in ORDER:
        t0 = time.time()
        row = {"clip_id": cid, "arm": arm, "condition": cond,
               "family": family(cond), "conf": "", "bits": "", "bit_acc": "",
               "pesq": "", "snr_db": "", "status": "ok", "seconds": ""}
        try:
            z = apply(cond, audio, WORK_SR)
            if z is None or len(z) == 0:
                row["status"] = "skip"
                rows.append(row)
                continue

            # detect_watermark's short-audio branch returns a 3-tuple instead of
            # the usual 2-tuple; same defensive unpack crop_ratio_sweep_20bps.py
            # uses. crop_50 can trip it on the shortest clips.
            res = detect_watermark(z, WORK_SR, detector)
            pat, conf = res[0], res[1]
            pat = np.asarray(pat).astype(int).ravel()[:N_BITS]

            row["conf"] = round(float(conf), 6)
            row["bits"] = "".join(map(str, pat.tolist()))
            if truth is not None and len(pat):
                n = min(len(pat), len(truth))
                row["bit_acc"] = round(float(np.mean(pat[:n] == truth[:n])), 6)

            if not needs_ref_align(cond):
                p = pesq_wb(src, z)
                s = snr_db(src, z)
                row["pesq"] = round(p, 4) if p is not None else ""
                row["snr_db"] = round(s, 3) if s is not None else ""
        except Exception as e:
            row["status"] = "error"
            row["bits"] = str(e)[:120]
            print(f"  {cid}/{cond} ERROR: {e}", file=sys.stderr)

        row["seconds"] = round(time.time() - t0, 3)
        rows.append(row)
        print(f"  {cid:<6s} {cond:<18s} conf={row['conf']!s:<10s} "
              f"bit_acc={row['bit_acc']!s:<10s} {row['status']}")
    return rows


def main(argv):
    if not os.path.exists(PAYLOADS):
        raise SystemExit("work/payloads.json missing -- run 'python embed.py' first.")
    meta = json.load(open(PAYLOADS))
    payloads = meta["payloads"]

    spec = json.load(open(os.path.join(HERE, "clips.json")))
    all_ids = ([f"wm{i:02d}" for i in range(len(spec["wm_clips"]))] +
               [f"cl{i:02d}" for i in range(len(spec["clean_clips"]))])

    one = get_arg(argv, "--clip", None)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if one:
        ids = [one]
    elif task is not None:
        ids = [all_ids[int(task)]]
    else:
        ids = all_ids

    os.makedirs(DATA_DIR, exist_ok=True)

    from aware.utils.models import load
    _, detector = load(name=AWARE_MODEL)
    print(f"loaded AWARE model {AWARE_MODEL!r}; scoring {len(ids)} clip(s) "
          f"x {len(ORDER)} conditions\n")

    for cid in ids:
        rows = score_clip(cid, detector, payloads)
        out = os.path.join(DATA_DIR, f"raw_{cid}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
