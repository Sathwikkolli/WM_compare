"""
casp/bm_null.py -- unwatermarked clips through every attack, scored by every detector.

Gives (a) the per-(system, attack config) native-score null for the 1%-FPR
threshold, and (b) decoded bits, so the payload test's false-alarm rate can be
checked empirically against random payloads.

    python bm_null.py --chunk 0 --n-per-chunk 10     # sbatch array
    python bm_null.py --mock --quick                 # local plumbing test
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bm_core as B                       # noqa: E402
from bm_run import build_systems, RESULTS  # noqa: E402

HEADER = ["clip_id", "system", "family", "label", "sev", "native_score", "bits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-per-chunk", type=int, default=10)
    ap.add_argument("--systems", default="audioseal,aware,timbre,ss")
    ap.add_argument("--families", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fam = [f for f in a.families.split(",") if f] or None
    cfgs = B.attack_configs(fam)
    if a.quick:
        seen, keep = set(), []
        for c in cfgs:
            if c["family"] not in seen:
                seen.add(c["family"])
                keep.append(c)
        cfgs = keep[:10]
    names = ["ss"] if a.mock else [s for s in a.systems.split(",") if s]
    S = build_systems(names)

    if a.mock:
        import run_casp
        y = B.resample(run_casp.synth(), 16000, B.SR)
        clips = [("null_synth", (y / np.max(np.abs(y)) * B.PEAK).astype(np.float32)[: 2 * B.SR])]
    else:
        # null clips: Emilia, seed 1, disjoint from the test clips (seed 0)
        meta = B.emilia_clips(a.n_per_chunk, seed=1, skip=a.chunk * a.n_per_chunk)
        clips = [(cid, B.load_master(p)) for cid, p in meta]

    out_dir = a.out or os.path.join(RESULTS, "null")
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, f"null_chunk{a.chunk:03d}.csv"), "w", newline="")
    w = csv.writer(fh)
    w.writerow(HEADER)
    t0 = time.time()
    print(f"null chunk {a.chunk}: {len(clips)} clips x {len(cfgs)} configs x systems {list(S)}", flush=True)
    for ci, (cid, y) in enumerate(clips):
        host = B.pcm16(y)
        for nm, s in S.items():
            if nm.startswith("ss"):
                s.prepare(host, 7000 + a.chunk * 100 + ci, cid)
        for cfg in [dict(family="none", label="clean", param=None, sev=-1, src="none")] + cfgs:
            try:
                y2 = host if cfg["family"] == "none" else B.run_attack(cfg, host)
            except Exception as e:
                print(f"  attack {cfg['family']}/{cfg['label']} failed: {str(e)[:60]}")
                continue
            if y2 is None:
                continue
            for nm, s in S.items():
                try:
                    score, bits = s.detect(y2, search=(cfg["family"] == "crop_head"))
                except Exception as e:
                    score, bits = float("nan"), np.zeros(s.nbits, int)
                w.writerow([cid, nm, cfg["family"], cfg["label"], cfg["sev"], f"{score:.5f}",
                            "".join(map(str, np.asarray(bits).astype(int).tolist()))])
        fh.flush()
        print(f"  [{ci + 1}/{len(clips)}] {cid}  [{time.time() - t0:.0f}s]", flush=True)
    fh.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
