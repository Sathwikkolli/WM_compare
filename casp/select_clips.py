"""
casp/select_clips.py -- pick the clips for the CASP threshold experiment.

RUN ON A LOGIN NODE BEFORE sbatch (reads masking/clips.json, which lives on the
cluster and is gitignored). Writes casp/clips2.json in the same schema.

  N clips spread across the pause-fraction range, from almost-continuous speech
  (little silence to hide in) to many pauses (where the Johnston threshold failed).

Both are cropped to --max-s at run time; selection looks at the same crop.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "masking"))
import common as C                                    # noqa: E402


def stats(y, sr=C.SR, frame=0.032):
    n = int(frame * sr)
    k = len(y) // n
    r = np.sqrt(np.mean(y[: k * n].reshape(k, n) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(r)
    peak = np.percentile(db, 95)
    speech = db > peak - 30                           # within 30 dB of loud speech
    return dict(level_db=float(peak), speech_frac=float(speech.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-s", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    meta = json.load(open(C.CLIPS_JSON))
    rows = []
    for c in meta["clips"]:
        y = C.load_16k(c["path"])[: int(a.max_s * C.SR)]
        rows.append({**c, **stats(y)})
    rows = [r for r in rows if r["duration_s"] >= 3.0] or rows
    # spread the picks over the pause-fraction range: most continuous .. most pauses
    rows.sort(key=lambda r: (-r["speech_frac"], -r["level_db"]))
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, a.n)).astype(int))
    picked = [rows[i] for i in idx]
    for j, r in enumerate(picked):
        r["role"] = f"pause_rank{j}of{len(picked) - 1}"      # 0 = most continuous
    json.dump(dict(source=meta["source"], note="selected by casp/select_clips.py",
                   clips=picked), open(os.path.join(HERE, "clips2.json"), "w"), indent=2)
    for i, r in enumerate(picked):
        print(f"[{i}] {r['role']:>18}: {r['clip_id']} {r['utt']}  dur {r['duration_s']}s  "
              f"level {r['level_db']:.1f} dBFS  speech_frac {r['speech_frac']:.2f}")
    print(f"submit with: sbatch --array=0-{len(picked) - 1} casp_threshold.sbatch")


if __name__ == "__main__":
    main()
