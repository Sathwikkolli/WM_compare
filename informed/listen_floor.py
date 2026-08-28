"""
informed/listen_floor.py -- what does DNSMOS 3.0 actually sound like?

The whole screen hangs on one number: DNSMOS 3.0 as the line between "an
attacker would still use this" and "they would not". That number is inherited
from `cascade/emilia_bench.py:98`, which uses it to pick source clips -- so it
is at least not invented for this run. But nobody in this project has ever
LISTENED to audio at 3.0 and agreed it is the right line.

This produces the listening set. For each attack it sweeps strength, scores
every setting with DNSMOS, then saves the clips whose scores land closest to a
ladder of targets:

    4.0   clean-ish
    3.5   the strict floor
    3.0   THE FLOOR -- the one that decides every verdict in the screen
    2.5   below the floor
    2.0   clearly degraded

Files are named by their MEASURED score, so what you hear is what was scored.

WHY TWO DIFFERENT ATTACKS

DNSMOS 3.0 from a music bed and DNSMOS 3.0 from white noise are the same number
and do not sound remotely alike. One is a podcast with the music too loud; the
other is a damaged recording. If they sound equally acceptable to you, the floor
is doing its job as a single threshold. If they do not, the screen may need a
per-attack floor, and that is worth knowing BEFORE 50 clips of cluster time.

Usage:
    python listen_floor.py                       # uses clips.json
    python listen_floor.py --audio path/to.wav   # any file you like
    python listen_floor.py --clips 0,7,21 --out ~/listen
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cascade"))

import attacks_screen as A                    # noqa: E402
import quality as Q                           # noqa: E402

TARGETS = [4.0, 3.5, 3.0, 2.5, 2.0]
CLIP_SECONDS = 10.0

# Fine grids, finer than the screen's: here we are hunting for specific quality
# levels rather than characterising an attack.
LADDERS = {
    "music_bed":      [30, 25, 20, 16, 12, 10, 8, 6, 4, 2, 0, -2, -5, -8, -12],
    "gaussian_noise": [45, 40, 35, 30, 25, 20, 15, 12, 10, 8, 5, 3, 0],
    "noise_babble":   [("babble.wav", s) for s in
                       [35, 30, 25, 20, 15, 12, 10, 8, 5, 2, 0]],
    "reverb":         [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
}


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def label_of(name, param):
    if name == "noise_babble":
        return f"{param[1]}dB"
    if name == "reverb":
        return f"rt{param}"
    return f"{param}dB"


def main(argv):
    out_dir = os.path.expanduser(get_arg(argv, "--out",
                                         os.path.join(BASE, "listen_floor")))
    os.makedirs(out_dir, exist_ok=True)

    backend, note = Q.backend()
    print(f"quality backend: {backend}\n  {note}\n")
    if backend == "none":
        raise SystemExit("No quality backend -- pip install -r requirements_extra.txt")
    if backend.startswith("squim") and "--force" not in argv:
        raise SystemExit(
            "Refusing to build a listening set on the unvalidated SQUIM "
            "fallback: the labels on the files would be numbers we do not "
            "trust. Install speechmos.")

    import cascade_lib as cl
    sr = cl.SR_MASTER

    # ---- pick source audio ------------------------------------------------
    sources = []
    if "--audio" in argv:
        p = get_arg(argv, "--audio", "")
        sources.append(("custom", p))
    else:
        import json
        cj = os.path.join(HERE, "clips.json")
        if not os.path.exists(cj):
            raise SystemExit(f"{cj} missing -- run clips.py, or pass --audio")
        clips = json.load(open(cj))["clips"]
        want = get_arg(argv, "--clips", "0,17,34")
        for i in [int(x) for x in want.split(",") if x.strip()]:
            if i < len(clips):
                sources.append((clips[i]["clip_id"], clips[i]["path"]))

    rows = []
    for clip_id, path in sources:
        y = cl.read_wav(path)
        n_keep = int(CLIP_SECONDS * sr)
        if len(y) < n_keep:
            print(f"{clip_id}: shorter than {CLIP_SECONDS}s -- skipping")
            continue
        y = y[:n_keep].astype("float32")

        # The clean reference, so there is always something to A/B against.
        q0 = Q.no_reference(y, sr)
        ref_name = f"{clip_id}_CLEAN_dnsmos{q0['ovrl'] or float('nan'):.2f}.wav"
        cl.write_wav(os.path.join(out_dir, ref_name), y)
        print(f"\n{clip_id}  clean DNSMOS = {q0['ovrl']}")
        rows.append({"clip_id": clip_id, "attack": "clean", "param": "-",
                     "dnsmos": q0["ovrl"], "target": "", "file": ref_name})

        for atk, ladder in LADDERS.items():
            print(f"  sweeping {atk} ...", flush=True)
            scored = []
            for param in ladder:
                z = A.apply(atk, param, y, sr)
                if z is None:
                    continue
                z = np.asarray(z, dtype="float32")
                q = Q.no_reference(z, sr)
                if q["ovrl"] is None:
                    continue
                scored.append((float(q["ovrl"]), param, z))

            if not scored:
                print(f"    {atk}: unavailable, skipped")
                continue

            # For each target, keep the setting whose MEASURED score is nearest.
            for tgt in TARGETS:
                best = min(scored, key=lambda s: abs(s[0] - tgt))
                score, param, z = best
                if abs(score - tgt) > 0.6:
                    # Do not label a 2.1 file as "3.0". Skip and say so.
                    print(f"    target {tgt:.1f}: nearest was {score:.2f} "
                          f"-- outside range, not saved")
                    continue
                lbl = label_of(atk, param)
                fname = (f"{clip_id}_{atk}_{lbl}_dnsmos{score:.2f}"
                         f"_target{tgt:.1f}.wav")
                cl.write_wav(os.path.join(out_dir, fname), z)
                rows.append({"clip_id": clip_id, "attack": atk, "param": lbl,
                             "dnsmos": round(score, 3), "target": tgt,
                             "file": fname})
                print(f"    target {tgt:.1f} -> {lbl:>10s}  measured {score:.2f}")

    # ---- index ------------------------------------------------------------
    idx_csv = os.path.join(out_dir, "index.csv")
    with open(idx_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip_id", "attack", "param",
                                           "dnsmos", "target", "file"])
        wr.writeheader()
        wr.writerows(rows)

    with open(os.path.join(out_dir, "LISTEN.md"), "w") as f:
        f.write("# Listening set — what does DNSMOS 3.0 sound like?\n\n")
        f.write("Every file is named by its **measured** DNSMOS, so what you "
                "hear is what was scored.\n\n")
        f.write("## How to listen\n\n")
        f.write("1. Start with a `CLEAN` file for each clip — that is the ceiling.\n")
        f.write("2. Then play the `target3.0` files. **Ask yourself the "
                "attacker's question, not the engineer's:** would you still "
                "publish this? Would you still use it in a video?\n")
        f.write("3. Compare `music_bed` at 3.0 against `gaussian_noise` at 3.0. "
                "Same number, very different damage. If one is clearly usable "
                "and the other clearly is not, a single global floor is wrong "
                "and the screen needs a per-attack floor.\n")
        f.write("4. Then 3.5 and 2.5, to see how much the verdict would move "
                "if the line moved.\n\n")
        f.write("## What hangs on this\n\n")
        f.write("Every verdict in `summary_screen.md` is decided by whether an "
                "attack kills detection **above or below this line**. Move the "
                "floor and attacks change category. `score_screen.py --floor "
                "3.5` re-runs the whole classification at the strict bar "
                "without recomputing any audio.\n\n")
        f.write("| clip | attack | setting | measured DNSMOS | target | file |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (r["clip_id"], r["attack"],
                                             -(r["dnsmos"] or 0))):
            f.write(f"| {r['clip_id']} | {r['attack']} | {r['param']} | "
                    f"{r['dnsmos']} | {r['target']} | `{r['file']}` |\n")

    print(f"\n{len(rows)} files -> {out_dir}")
    print(f"  index: {idx_csv}")
    print(f"  read:  {os.path.join(out_dir, 'LISTEN.md')}")
    print("\nCopy them back and listen before trusting the 3.0 floor:")
    print(f"  scp -r <user>@greatlakes-xfer.arc-ts.umich.edu:{out_dir} .")


if __name__ == "__main__":
    main(sys.argv[1:])
