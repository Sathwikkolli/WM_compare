"""
informed/listen_screen.py -- hear the audio behind every verdict in the screen.

Every SECURE / VULNERABLE / NO_FLOOR call in summary_screen.md turns on one
judgement: was the audio still usable at that setting? That judgement is made by
DNSMOS, which is a model, not a listening test. This regenerates the audio at
the exact settings the verdicts were computed from, so the classification can be
checked by ear.

WHY THIS MATTERS MORE THAN IT SOUNDS

DNSMOS P.835 was built to evaluate NOISE SUPPRESSION. Its `bak` sub-score rates
how intrusive the background is -- so it may well penalise a deliberate music
bed as if it were noise, even when the result sounds like an ordinary podcast.

The screen produced exactly the pattern that would cause: `music_bed`, `echo`,
`highpass` and `time_stretch` came back with a BLANK s*, meaning not one tested
setting was judged usable -- including music at 20 dB SNR, where the music is
barely audible. If that sounds fine to you, the floor is wrong for
background-addition attacks and those verdicts need revisiting.

WHAT IT SAVES, per attack

  s*        the setting the verdict was computed at -- the decision point
  weaker    one step milder
  stronger  one step harsher
  (blank s*) the MILDEST setting tested, so you can hear what was already
             judged too damaged

Every file is named with its MEASURED numbers, so what you hear is what was
scored:

    music_bed__20dB__SECURE__dnsmos2.61_drop0.61_conf1.000.wav
                              ^ measured  ^ from clean  ^ detection

Usage:
    python listen_screen.py                       # clip 0, ~3 files per attack
    python listen_screen.py --clip 7
    python listen_screen.py --attacks music_bed,echo,highpass
    python listen_screen.py --all                 # every swept setting
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                    # noqa: E402
import quality as Q                           # noqa: E402

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
CLIPS_JSON = os.path.join(HERE, "clips.json")

CLIP_SECONDS = 10.0
DETECT_THRESHOLD = 0.50
DROP_FLOOR = 0.5


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default):
    if flag in argv:
        return [x.strip() for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return list(default)


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_metrics():
    """Per-config means from score_screen.py. Keeps labels identical to the summary."""
    p = os.path.join(DATA_DIR, "screen_metrics.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing -- run score_screen.py first")
    by = defaultdict(list)
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("conf", "dnsmos", "dnsmos_clean", "drop", "bit_acc", "pesq"):
                r[k] = fnum(r.get(k))
            r["n_ok"] = int(fnum(r.get("n_ok")) or 0)
            by[r["attack"]].append(r)
    return by


def classify(recs):
    """s* and verdict, mirroring score_screen.py exactly.

    Duplicated rather than imported because score_screen's logic lives inside
    main(). If one changes, change both -- the filenames must agree with the
    summary or this whole exercise misleads.
    """
    runnable = [r for r in recs if r["n_ok"] > 0 and np.isfinite(r["dnsmos"])]
    if not runnable:
        return None, "UNAVAILABLE", runnable
    usable = [r for r in runnable
              if np.isfinite(r["drop"]) and r["drop"] <= DROP_FLOOR]
    vulnerable = [r for r in usable
                  if np.isfinite(r["conf"]) and r["conf"] < DETECT_THRESHOLD]
    if not usable:
        return None, "SECURE", runnable
    s = min(usable, key=lambda r: r["conf"] if np.isfinite(r["conf"]) else 9e9)
    if vulnerable:
        return s, "VULNERABLE", runnable
    if len(usable) == len(runnable):
        return s, "NO_FLOOR", runnable
    return s, "SECURE", runnable


def picks(recs, s_star, want_all):
    """Which settings to render."""
    order = [r["param"] for r in recs]
    if want_all:
        return recs
    if s_star is None:
        # Nothing was judged usable. Render the MILDEST setting -- that is the
        # one whose verdict is most likely to be wrong, and the one worth hearing.
        return recs[:1]
    i = order.index(s_star["param"])
    out = [recs[i]]
    if i > 0:
        out.insert(0, recs[i - 1])
    if i + 1 < len(recs):
        out.append(recs[i + 1])
    return out


def safe(s):
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in str(s))


def main(argv):
    out_dir = os.path.expanduser(get_arg(argv, "--out",
                                         os.path.join(BASE, "listen_screen")))
    os.makedirs(out_dir, exist_ok=True)
    want_all = "--all" in argv

    backend, note = Q.backend()
    print(f"quality backend: {backend}\n  {note}\n")

    metrics = load_metrics()
    attacks = get_list(argv, "--attacks", sorted(metrics))
    unknown = [a for a in attacks if a not in metrics]
    if unknown:
        raise SystemExit(f"unknown attacks: {unknown}")

    clips = json.load(open(CLIPS_JSON))["clips"]
    ci = get_arg(argv, "--clip", 0, int)
    clip = clips[ci]

    import cascade_lib as cl
    sr = cl.SR_MASTER
    org = cl.read_wav(clip["path"])[:int(CLIP_SECONDS * sr)].astype("float32")

    print(f"clip {ci} [{clip['clip_id']}] {os.path.basename(clip['path'])}")
    clean_q = Q.no_reference(org, sr)
    print(f"  clean DNSMOS = {clean_q['ovrl']}")

    print("  loading AWARE and embedding...")
    adapter = cl.get_adapter("aware")
    wm = np.asarray(adapter.embed(org), dtype="float32")[:len(org)]
    c0, _, b0 = adapter.detect(wm)
    print(f"  baseline conf={c0:.4f} bit_acc={b0:.4f}\n")

    # References first, so there is always something to A/B against.
    cl.write_wav(os.path.join(out_dir, "_REFERENCE_clean.wav"), org)
    cl.write_wav(os.path.join(out_dir, "_REFERENCE_watermarked.wav"), wm)

    rows = []
    for atk in attacks:
        recs = metrics[atk]
        s_star, verdict, runnable = classify(recs)
        chosen = picks(recs, s_star, want_all)
        star_param = s_star["param"] if s_star else None
        flag = "" if s_star else "   <- NO usable setting; rendering the mildest"
        print(f"{atk:22s} {verdict:12s} s*={str(star_param):>10s}{flag}")

        for r in chosen:
            param = None
            for lbl, prm in A.SCREEN_GRID.get(atk, []):
                if lbl == r["param"]:
                    param = prm
                    break
            if param is None:
                continue
            z = A.apply(atk, param, wm, sr)
            if z is None:
                print(f"    {r['param']:>12s}  could not render")
                continue
            z = np.asarray(z, dtype="float32")

            role = ("s-star" if star_param and r["param"] == star_param
                    else "mildest" if s_star is None else "neighbour")
            name = (f"{safe(atk)}__{safe(r['param'])}__{verdict}__{role}"
                    f"__dnsmos{r['dnsmos']:.2f}_drop{r['drop']:+.2f}"
                    f"_conf{r['conf']:.3f}.wav")
            cl.write_wav(os.path.join(out_dir, name), z)
            rows.append({
                "attack": atk, "param": r["param"], "category": r.get("category", ""),
                "verdict": verdict, "role": role,
                "dnsmos": round(r["dnsmos"], 3), "drop": round(r["drop"], 3),
                "conf": round(r["conf"], 4), "bit_acc": round(r["bit_acc"], 3),
                "file": name,
            })

    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        wr.writeheader()
        wr.writerows(rows)

    with open(os.path.join(out_dir, "LISTEN.md"), "w") as f:
        f.write("# Hear the audio behind every verdict\n\n")
        f.write(f"Clip `{clip['clip_id']}`, clean DNSMOS "
                f"**{clean_q['ovrl']:.3f}**, baseline detection **{c0:.4f}**.\n\n")
        f.write("Filenames carry the MEASURED numbers: `dnsmos` is the quality "
                "score, `drop` is how far it fell from this clip's clean score "
                "(the floor is 0.5), `conf` is detection confidence "
                "(threshold 0.50).\n\n")
        f.write("## The question to ask\n\n")
        f.write("Not *\"can I hear damage?\"* — you will hear damage everywhere. "
                "Ask the attacker's question: **would I still publish this?**\n\n")
        f.write("## Start here\n\n")
        f.write("The suspicious verdicts are the attacks where NOT ONE setting "
                "was judged usable, so `s*` is blank and the file is marked "
                "`mildest`. DNSMOS was built to score noise suppression and may "
                "be unfair to deliberate background sound. **If the mildest "
                "setting sounds fine to you, that verdict is wrong.**\n\n")
        blanks = sorted({r["attack"] for r in rows if r["role"] == "mildest"})
        for a in blanks:
            f.write(f"- `{a}`\n")
        f.write("\n## Everything\n\n")
        f.write("| attack | setting | verdict | role | DNSMOS | drop | conf | bit acc | file |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (r["verdict"], r["attack"], r["param"])):
            f.write(f"| {r['attack']} | {r['param']} | {r['verdict']} | {r['role']} "
                    f"| {r['dnsmos']} | {r['drop']:+} | {r['conf']} | {r['bit_acc']} "
                    f"| `{r['file']}` |\n")

    print(f"\n{len(rows)} files -> {out_dir}")
    print(f"  read: {os.path.join(out_dir, 'LISTEN.md')}")
    print("\nCopy back with:")
    print(f"  scp -r <uniqname>@greatlakes-xfer.arc-ts.umich.edu:{out_dir} .")


if __name__ == "__main__":
    main(sys.argv[1:])
