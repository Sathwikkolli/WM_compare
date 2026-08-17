"""
damage_ab/run_damage.py -- score the sweep: 5 clips x 15 conditions = 75 rows.

WHAT ONE ROW IS: one (clip, condition) pair, holding BOTH arms side by side.

That is deliberately different from ab_aware/run_ab.py, which writes one row per
(clip, arm, condition). The question here is a within-clip, within-condition
comparison -- "did the attack hurt the unwatermarked file as much as the
watermarked one?" -- so the two arms belong on one row. A long format would make
that comparison a join, and a join is something that can be got wrong.

THE FOUR PESQ MEASUREMENTS, and why one number is not enough. Every column below
has a different reference signal, and conflating them is the easy mistake:

    wmcost_pesq   src -> wm             what AWARE costs. No attack involved.
                                        Constant per clip; repeated on every row.
    src_pesq      src -> attack(src)    THE CONTROL. Attack damage to audio with
                                        no watermark in it at all. If this is
                                        already unusable, the attack is
                                        destructive and AWARE's failure on it is
                                        not evidence about AWARE.
    wm_pesq       wm  -> attack(wm)     attack damage measured against the
                                        watermarked signal. Compare directly
                                        with src_pesq: same clip, same attack,
                                        the only difference is the watermark.
    comb_pesq     src -> attack(wm)     watermark AND attack together. This is
                                        the column the A/B reported, and on its
                                        own it cannot separate the two causes.

BAND RETENTION IS THE MECHANISM COLUMN. PESQ says "worse"; band_keep says what
happened to 1000-4000 Hz, where AWARE actually writes. A high-pass empties that
band (<<1) and a coarse quantiser floods it with switching noise (>1). Both kill
detection; they are not the same failure and no perceptual score distinguishes
them.

NO THRESHOLDING HERE. Raw confidence is written; 0.5 is applied in analyze.py.
Same rule as run_ab.py -- baking the cut into the raw data destroys the ability
to ask a different threshold later.

Attacked audio is written to work/att/ (and work/listen/ for one clip) so the
damage can be verified by ear rather than only by PESQ. Written as PCM_16; every
number in the CSV is measured in memory BEFORE writing, so the file format
cannot affect a result.

Usage:
    python run_damage.py                  # all clips
    python run_damage.py --clip cl02
    python run_damage.py --no-audio       # numbers only, no WAVs
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
sys.path.insert(0, HERE)

from sweep import (ORDER, AWARE_BAND, apply_cond, band_keep,  # noqa: E402
                   group_of, label_of, STRENGTH_X)

WORK_SR = 16000
N_BITS = 20
AWARE_MODEL = os.environ.get("AWARE_MODEL", "AWARE")
RUN = os.environ.get("DAMAGE_RUN", "2026-08-17_attack-damage-control")

WORK = os.path.join(HERE, "work")
SRC_DIR = os.path.join(WORK, "src")
WM_DIR = os.path.join(WORK, "wm")
ATT_DIR = os.path.join(WORK, "att")
LISTEN_DIR = os.path.join(WORK, "listen")
PAYLOADS = os.path.join(WORK, "payloads.json")
DATA_DIR = os.path.join(BASE, "results", RUN, "data")

# One clip's full condition set gets copied to work/listen/ with flat names, so
# the audio can be auditioned without walking 150 files. Comparing one clip
# across all 15 conditions is the listening test; 5 clips of the same thing is not.
LISTEN_CLIP = os.environ.get("LISTEN_CLIP", "cl00")

FIELDS = [
    "clip_id", "condition", "group", "strength", "label",
    "wmcost_pesq", "wmcost_snr_db",
    "src_pesq", "src_snr_db", "src_band_keep", "src_conf",
    "wm_pesq", "wm_snr_db", "wm_band_keep", "wm_conf", "wm_bit_acc",
    "comb_pesq",
    "status", "seconds",
]


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def read_wav(path):
    import soundfile as sf
    y, sr = sf.read(path)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    assert sr == WORK_SR, f"{path} is {sr} Hz, expected {WORK_SR}"
    return np.asarray(y, dtype="float32")


def write_wav(path, y):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(y, dtype="float32"), WORK_SR, subtype="PCM_16")


def snr_db(ref, deg):
    """Lifted from ab_aware/run_ab.py so the two runs' SNRs are the same number."""
    n = min(len(ref), len(deg))
    if n == 0:
        return None
    ref, deg = ref[:n], deg[:n]
    p_n = float(((deg - ref) ** 2).sum())
    if p_n <= 0:
        return None
    return float(10.0 * np.log10(float((ref ** 2).sum()) / p_n))


def pesq_wb(ref, deg):
    """Wideband PESQ. None rather than a guess when it cannot be computed.

    The min-length truncation handles codec padding (ffmpeg leaves AAC trailing
    padding; measured lag 0 for mp3/opus/aac -- see ab_aware/attacks_ab.py).
    """
    try:
        from pesq import pesq
        n = min(len(ref), len(deg))
        if n < WORK_SR // 2:
            return None
        v = float(pesq(WORK_SR, ref[:n], deg[:n], "wb"))
        return v if v == v else None
    except Exception:
        return None


def r(v, nd):
    return "" if v is None else round(float(v), nd)


def detect(z, detector, truth):
    """(conf, bit_acc) for one signal. bit_acc is None when there is no payload."""
    from aware.service import detect_watermark
    res = detect_watermark(z, WORK_SR, detector)
    pat, conf = res[0], res[1]
    pat = np.asarray(pat).astype(int).ravel()[:N_BITS]
    acc = None
    if truth is not None and len(pat):
        n = min(len(pat), len(truth))
        acc = float(np.mean(pat[:n] == truth[:n]))
    return float(conf), acc


def score_clip(cid, detector, payloads, save_audio=True):
    src = read_wav(os.path.join(SRC_DIR, cid + ".wav"))
    wm = read_wav(os.path.join(WM_DIR, cid + ".wav"))
    truth = np.asarray(payloads[cid], dtype=int)

    # Watermark cost, once per clip. Repeated on every row of this clip so the
    # CSV can be read row-wise without a lookup back to the clean row.
    wc_pesq = pesq_wb(src, wm)
    wc_snr = snr_db(src, wm)

    rows = []
    for cond in ORDER:
        t0 = time.time()
        row = dict.fromkeys(FIELDS, "")
        row.update({
            "clip_id": cid, "condition": cond, "group": group_of(cond),
            "strength": STRENGTH_X.get(cond, ""), "label": label_of(cond),
            "wmcost_pesq": r(wc_pesq, 4), "wmcost_snr_db": r(wc_snr, 3),
            "status": "ok",
        })
        try:
            a_src = apply_cond(cond, src, WORK_SR)
            a_wm = apply_cond(cond, wm, WORK_SR)
            if a_src is None or a_wm is None or len(a_src) == 0 or len(a_wm) == 0:
                row["status"] = "skip"
                rows.append(row)
                continue

            row["src_pesq"] = r(pesq_wb(src, a_src), 4)
            row["src_snr_db"] = r(snr_db(src, a_src), 3)
            row["src_band_keep"] = r(band_keep(src, a_src, WORK_SR), 5)

            row["wm_pesq"] = r(pesq_wb(wm, a_wm), 4)
            row["wm_snr_db"] = r(snr_db(wm, a_wm), 3)
            row["wm_band_keep"] = r(band_keep(wm, a_wm, WORK_SR), 5)
            row["comb_pesq"] = r(pesq_wb(src, a_wm), 4)

            # The src arm is scored too: it says whether the attack pushes
            # unwatermarked audio TOWARD a false positive. It is a paired control,
            # not an independent negative -- no FPR comes out of 5 clips.
            c_src, _ = detect(a_src, detector, None)
            c_wm, acc = detect(a_wm, detector, truth)
            row["src_conf"] = round(c_src, 6)
            row["wm_conf"] = round(c_wm, 6)
            row["wm_bit_acc"] = r(acc, 6)

            if save_audio:
                write_wav(os.path.join(ATT_DIR, cid, f"{cond}_src.wav"), a_src)
                write_wav(os.path.join(ATT_DIR, cid, f"{cond}_wm.wav"), a_wm)
                if cid == LISTEN_CLIP:
                    write_wav(os.path.join(LISTEN_DIR, f"{cond}_src.wav"), a_src)
                    write_wav(os.path.join(LISTEN_DIR, f"{cond}_wm.wav"), a_wm)
        except Exception as e:
            row["status"] = "error"
            row["label"] = str(e)[:120]
            print(f"  {cid}/{cond} ERROR: {e}", file=sys.stderr)

        row["seconds"] = round(time.time() - t0, 3)
        rows.append(row)
        print(f"  {cid:<5s} {cond:<14s} "
              f"src_pesq={row['src_pesq']!s:<8s} wm_pesq={row['wm_pesq']!s:<8s} "
              f"wm_conf={row['wm_conf']!s:<10s} band={row['wm_band_keep']!s:<9s} "
              f"{row['status']}")
    return rows


def main(argv):
    save_audio = "--no-audio" not in argv

    if not os.path.exists(PAYLOADS):
        raise SystemExit("work/payloads.json missing -- run 'python embed_pairs.py' first.")
    payloads = json.load(open(PAYLOADS))["payloads"]

    clips = json.load(open(os.path.join(HERE, "clips.json")))["clips"]
    all_ids = [f"cl{i:02d}" for i in range(len(clips))]

    one = get_arg(argv, "--clip", None)
    ids = [one] if one else all_ids

    os.makedirs(DATA_DIR, exist_ok=True)
    if save_audio:
        os.makedirs(LISTEN_DIR, exist_ok=True)

    from aware.utils.models import load
    _, detector = load(name=AWARE_MODEL)
    print(f"loaded AWARE model {AWARE_MODEL!r}")
    print(f"{len(ids)} clip(s) x {len(ORDER)} conditions = {len(ids) * len(ORDER)} rows")
    print(f"AWARE band for retention: {AWARE_BAND[0]:.0f}-{AWARE_BAND[1]:.0f} Hz")
    print(f"audio: {'work/att/ + work/listen/' if save_audio else 'DISABLED (--no-audio)'}\n")

    out_rows = []
    for cid in ids:
        out_rows += score_clip(cid, detector, payloads, save_audio)

    out = os.path.join(DATA_DIR, "raw.csv" if not one else f"raw_{one}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}  ({len(out_rows)} rows)")
    if save_audio:
        print(f"wrote {ATT_DIR}/  and  {LISTEN_DIR}/ (clip {LISTEN_CLIP}, all conditions)")
    print("\nnext:  python analyze.py")


if __name__ == "__main__":
    main(sys.argv[1:])
