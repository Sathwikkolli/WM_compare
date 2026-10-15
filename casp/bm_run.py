"""
casp/bm_run.py -- benchmark one clip: systems x operating points x attacks.

    python bm_run.py --selftest                       # cluster: loads every system, one clip, one config per family
    python bm_run.py --dataset asv --index 0          # one clip (sbatch array)
    python bm_run.py --dataset emilia --index 3
    python bm_run.py --mock --quick                   # local plumbing test (SS only, synthetic clip)

Operating points
  ship   neural systems at s = 1 (as shipped). SS is scaled to AWARE's PESQ on that clip.
  match  every system scaled to each PESQ in --targets (default 4.2) by residual scaling.

Rows: one per (clip, system, op, attack config) with native score, decoded bits,
bit accuracy and whether it clears the binomial 1% payload test.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bm_core as B                       # noqa: E402

RUN = os.environ.get("CASP_BENCH_RUN", "2026-10-15_casp-bench-v2")
RESULTS = os.path.join(B.ROOT, "results", RUN)
HEADER = ["dataset", "clip_id", "system", "op", "target", "scale", "status", "pesq", "stoi", "snr_db",
          "family", "label", "sev", "native_score", "bit_acc", "n_bits", "k_ok", "bits_ok", "bits", "payload"]


def build_systems(names, mock=False):
    S = {}
    for n in names:
        try:
            if n == "audioseal":
                S[n] = B.AudioSealSys()
            elif n == "aware":
                S[n] = B.AwareSys()
            elif n == "timbre":
                S[n] = B.TimbreSys()
            elif n == "ss":
                S[n] = B.SSSys("aware_shape", K=16)
            elif n == "ss_casp":
                S[n] = B.SSSys("casp", K=16)
            elif n == "ss_flat":
                S[n] = B.SSSys("flat_abs", K=16)
            elif n == "ss_shuf_all":
                S[n] = B.SSSys("casp_shuf_all", K=16)
            elif n == "ss_shuf_time":
                S[n] = B.SSSys("casp_shuf_time", K=16)
            else:
                print(f"  unknown system {n}")
        except Exception as e:
            print(f"  ! system {n} failed to load: {e}")
            traceback.print_exc()
    return S


def load_clip(dataset, index, mock, quick):
    if mock:
        import run_casp
        y = run_casp.synth()
        y = B.resample(y, 16000, B.SR)
        y = (y / np.max(np.abs(y)) * B.PEAK).astype(np.float32)
        return "synth", y[: int((2.0 if quick else B.MAX_S) * B.SR)]
    if dataset == "asv":
        cid, path = B.asv_clips()[index]
    else:
        cid, path = B.emilia_clips(1, seed=0, skip=index)[0]
    return cid, B.load_master(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="asv", choices=["asv", "emilia"])
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--systems", default="audioseal,aware,timbre,ss,ss_casp,ss_shuf_all,ss_shuf_time")
    ap.add_argument("--targets", default="4.2", help="comma list of PESQ targets for the matched op")
    ap.add_argument("--families", default="", help="comma list; default = all except dac")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fam = [f for f in a.families.split(",") if f] or None
    cfgs = B.attack_configs(fam)
    if a.quick or a.selftest:
        seen, keep = set(), []
        for c in cfgs:
            if c["family"] not in seen:
                seen.add(c["family"])
                keep.append(c)
        cfgs = keep if a.selftest else keep[:12]
    names = ["ss"] if a.mock else [s for s in a.systems.split(",") if s]
    S = build_systems(names)
    if not S:
        sys.exit("no systems loaded")

    cid, host = load_clip(a.dataset, a.index, a.mock, a.quick)
    host = B.pcm16(host)
    n_total = len(cfgs)
    print(f"[{a.dataset}:{cid}] {len(host) / B.SR:.1f}s  systems {list(S)}  attack configs {n_total}  "
          f"quality axis {'PESQ' if B.HAVE_PESQ else 'SNR (fallback)'}", flush=True)
    t0 = time.time()

    rng = np.random.RandomState(1000 * a.seed + (a.index if a.dataset == "asv" else 500 + a.index))
    key = 3000 + a.index + (0 if a.dataset == "asv" else 1000)
    marked, payloads = {}, {}
    for nm, s in S.items():
        if nm.startswith("ss") and not s.prepare(host, key, cid):
            print(f"  {nm}: no CASP probe for {cid}; skipped")
            continue
        bits = rng.randint(0, 2, s.nbits)
        payloads[nm] = bits
        try:
            m = s.embed(host, bits)
            marked[nm] = B.fit_len(m, len(host))
        except Exception as e:
            print(f"  ! embed failed for {nm}: {e}")
            traceback.print_exc()
    print(f"  embedded {list(marked)} [{time.time() - t0:.0f}s]", flush=True)

    # operating points -> scaled signals
    sigs = {}                                   # (system, op, target) -> (signal, scale, status, quality)
    shipped = {}
    for nm in marked:
        if not nm.startswith("ss"):
            z = B.apply_scale(host, marked[nm], 1.0)
            shipped[nm] = B.quality(host, z, full=True)
            sigs[(nm, "ship", "")] = (z, 1.0, "ok", shipped[nm])
    default_q = 4.2 if B.HAVE_PESQ else 20.0          # SNR dB when pesq is not installed (local test only)
    ref_q = (shipped.get("aware", {}).get("pesq") or float(np.median([v["pesq"] for v in shipped.values()])))         if shipped else default_q
    targets = [float(t) for t in a.targets.split(",") if t]
    if not B.HAVE_PESQ and a.targets == "4.2":
        targets = [20.0]
    for nm in marked:
        if nm.startswith("ss"):
            s, q, st = B.match_scale(host, marked[nm], ref_q)
            z = B.apply_scale(host, marked[nm], s)
            sigs[(nm, "ship", "")] = (z, s, st, B.quality(host, z, full=True))
        for t in targets:
            s, q, st = B.match_scale(host, marked[nm], t)
            z = B.apply_scale(host, marked[nm], s)
            sigs[(nm, "match", f"{t:g}")] = (z, s, st, B.quality(host, z, full=True))
    for k, (z, s, st, q) in sigs.items():
        print(f"   {k[0]:10s} {k[1]:5s} {k[2]:>4s}  s={s:.2f} ({st})  PESQ {q['pesq']:.2f}", flush=True)
    print(f"  operating points ready [{time.time() - t0:.0f}s]", flush=True)

    out_dir = a.out or os.path.join(RESULTS, f"{a.dataset}_{cid}")
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "bm_rows.csv"), "w", newline="")
    w = csv.writer(fh)
    w.writerow(HEADER)
    kth = {nm: B.binom_threshold(S[nm].nbits) for nm in marked}

    def emit(nm, op, tgt, scale, status, qual, cfg, score, bits, payload):
        nb = len(payload)
        acc = float(np.mean(np.asarray(bits)[:nb] == payload)) if bits is not None else float("nan")
        k = int(round(acc * nb)) if bits is not None else -1
        w.writerow([a.dataset, cid, nm, op, tgt, f"{scale:.4f}", status, f"{qual['pesq']:.3f}",
                    f"{qual.get('stoi', float('nan')):.3f}", f"{qual['snr_db']:.2f}",
                    cfg["family"], cfg["label"], cfg["sev"], f"{score:.5f}", f"{acc:.4f}", nb, kth[nm],
                    int(k >= kth[nm]) if bits is not None else "",
                    "".join(map(str, np.asarray(bits).astype(int).tolist())) if bits is not None else "",
                    "".join(map(str, payload.astype(int).tolist()))])

    done = 0
    for cfg in [dict(family="none", label="clean", param=None, sev=-1, src="none")] + cfgs:
        for (nm, op, tgt), (z, s, st, q) in sigs.items():
            try:
                y2 = z if cfg["family"] == "none" else B.run_attack(cfg, z)
            except Exception as e:
                print(f"  attack {cfg['family']}/{cfg['label']} failed: {str(e)[:60]}")
                y2 = None
            if y2 is None:
                continue
            try:
                score, bits = S[nm].detect(y2, search=(cfg["family"] == "crop_head"))
            except Exception as e:
                score, bits = float("nan"), None
                print(f"  detect {nm} failed after {cfg['family']}/{cfg['label']}: {str(e)[:60]}")
            emit(nm, op, tgt, s, st, q, cfg, score, bits, payloads[nm])
        done += 1
        if done % 10 == 0:
            fh.flush()
            print(f"   attacks {done}/{n_total + 1} [{time.time() - t0:.0f}s]", flush=True)
    fh.close()
    json.dump(dict(dataset=a.dataset, clip_id=cid, key=key, shipped_pesq=ref_q,
                   systems={nm: dict(nbits=int(S[nm].nbits), k_threshold=int(kth[nm]),
                                     fpr=B.binom_tail(int(S[nm].nbits), int(kth[nm]))) for nm in marked}),
              open(os.path.join(out_dir, "bm_meta.json"), "w"), indent=1)
    print(f"done [{time.time() - t0:.0f}s] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
