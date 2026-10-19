"""
mpeg_wm/bench_run.py -- one ASVspoof clip: 4 watermarkers x the VoxWatermark attacks.

    python bench_run.py --index 0                 # one clip (sbatch array task)
    python bench_run.py --index 0 --selftest      # one config per attack family, fails fast

Systems (random payload per clip):
    AudioSeal 16 bit, AWARE 20 bit, Timbre 10 bit   -- adapters from casp/bm_core.py
    MPEG      16 bit                                -- this folder (Swanson et al. masking,
                                                       blind multibit spread spectrum)
    CASP-1, CASP-2 (+ "-shuf" controls), 16 bit     -- casp_place.py: CASP decides where,
                                                       MPEG's energy decides how much
    e.g. --systems casp1,casp1_shuf,casp2,casp2_shuf  (each "_shuf" reuses its partner's
    tile gains for the clip, so list the partner first)
Operating points:
    ship   every system at its own default strength (MPEG: watermark at the mask, 0 dB)
    match  every system rescaled to PESQ-wb = target (default 4.2), residual scaling
    null   the UNWATERMARKED clip decoded by every system (false-alarm check)
Attacks: the 17 VoxWatermark families (cascade/vox_attacks.py), all strengths,
run at the 22.05 kHz master rate and saved as 16-bit PCM (bm_core.run_attack).

Rows: one per (clip, system, op, attack config) -> results/<run>/<clip_id>/rows.csv
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "casp"))
import bm_core as B                          # noqa: E402  (also puts cascade/ on sys.path)
sys.path.insert(0, HERE)                     # our embed/detect win over any same-named module
import embed as E                            # noqa: E402
import detect as D                           # noqa: E402

RUN = os.environ.get("MPEG_BENCH_RUN", "2026-10-19_mpeg-vox-bench")
RESULTS = os.path.join(ROOT, "results", RUN)
WORK = os.path.join(HERE, "work", RUN)       # wavs for listening; gitignored
CLIPS_JSON = os.path.join(HERE, "clips120.json")
HEADER = ["clip_id", "utt", "system", "op", "target", "scale", "status", "pesq", "stoi", "snr_db",
          "family", "label", "sev", "native_score", "bit_acc", "n_bits", "k_ok", "bits_ok",
          "bits", "payload"]
UNSET = object()                             # sentinel: "not computed yet" (None means "attack unavailable")


class MpegSys:
    """Swanson-style masking watermark, run at 16 kHz; only the residual is resampled
    back, so the host's content above 8 kHz is left untouched."""
    name, nbits, native_sr, has_score = "MPEG", 16, B.SR16, True

    def __init__(self, strength_db=0.0):
        self.strength_db = float(strength_db)
        self.key = 0

    def prepare(self, key, clip_id=None):
        self.key = int(key)

    def _embed16(self, y16, bits):
        return E.embed_watermark(y16, bits, self.key, strength_db=self.strength_db)

    def embed(self, y22, bits, **kw):
        y16 = B.resample(y22, B.SR, B.SR16)
        m16 = self._embed16(y16, bits)
        res = B.fit_len(B.resample(m16 - y16, B.SR16, B.SR), len(y22))
        return (np.asarray(y22, np.float32) + res).astype(np.float32)

    def detect(self, y22, **kw):
        from scipy.stats import norm
        bits, _, p = D.decode(B.resample(y22, B.SR, B.SR16), self.key, self.nbits)
        return float(norm.isf(max(p, 1e-300) / 2)), bits


class CaspSys(MpegSys):
    """MPEG's watermark energy, CASP's placement (casp_place.py). Same key, PN, payload
    layout and blind detector as MPEG. `shared` carries the tile gains from the real
    system to its shuffled control for the same clip, so CASP runs only once per way."""
    has_score = True

    def __init__(self, way, shuffle, shared):
        super().__init__(0.0)
        self.way, self.shuffle, self.shared = way, shuffle, shared
        self.name = f"CASP-{way}" + ("-shuf" if shuffle else "")
        self.info = {}

    def prepare(self, key, clip_id=None):
        self.key, self.clip_id = int(key), clip_id

    def _embed16(self, y16, bits):
        import casp_place as CP
        cached = self.shared.get((self.clip_id, self.way))
        m16, info = CP.embed_casp(y16, bits, self.key, way=self.way, shuffle=self.shuffle,
                                  gain_db=cached)
        if cached is None:
            self.shared[(self.clip_id, self.way)] = info["gain_db"]
        g = info["gain_db"]
        self.info = dict(way=self.way, shuffle=self.shuffle, reused_gains=cached is not None,
                         gain_db_min=float(g.min()), gain_db_max=float(g.max()),
                         casp_dprime_history=[h["d_total"] for h in info.get("history", [])])
        return m16


def build_systems(names, strength_db):
    S, shared = {}, {}
    makers = {"audioseal": B.AudioSealSys, "aware": B.AwareSys, "timbre": B.TimbreSys,
              "mpeg": lambda: MpegSys(strength_db),
              "casp1": lambda: CaspSys(1, False, shared), "casp1_shuf": lambda: CaspSys(1, True, shared),
              "casp2": lambda: CaspSys(2, False, shared), "casp2_shuf": lambda: CaspSys(2, True, shared)}
    for n in names:
        try:
            S[n] = makers[n]()
        except Exception as e:
            print(f"  ! system {n} failed to load: {e}")
            traceback.print_exc()
    return S


def vox_configs():
    import vox_attacks as V
    return B.attack_configs([f"vox_{f}" for f in V.VOX_GRID])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--clips", default=CLIPS_JSON)
    ap.add_argument("--systems", default="mpeg,audioseal,aware,timbre")
    ap.add_argument("--targets", default="4.2", help="comma list of PESQ targets for 'match'; '' = none")
    ap.add_argument("--strength-db", type=float, default=0.0, help="MPEG level relative to the mask")
    ap.add_argument("--selftest", action="store_true", help="one config per attack family")
    ap.add_argument("--no-wavs", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfgs = vox_configs()
    if a.selftest:
        seen, keep = set(), []
        for c in cfgs:
            if c["family"] not in seen:
                seen.add(c["family"])
                keep.append(c)
        cfgs = keep
    S = build_systems([s for s in a.systems.split(",") if s], a.strength_db)
    if not S:
        sys.exit("no systems loaded")

    spec = json.load(open(a.clips))["clips"][a.index]
    cid, utt = spec["clip_id"], spec["utt"]
    host = B.pcm16(B.load_master(spec["path"]))
    print(f"[{cid} {utt}] {len(host) / B.SR:.1f}s  systems {list(S)}  attack configs {len(cfgs)}  "
          f"quality {'PESQ' if B.HAVE_PESQ else 'SNR (pesq missing!)'}", flush=True)
    t0 = time.time()

    # ---- embed: random payload per (clip, system) ----------------------------------
    rng = np.random.RandomState(a.index)
    for nm, s in S.items():
        if isinstance(s, MpegSys):                    # MPEG and the CASP systems share one key
            s.prepare(5000 + a.index, cid)
    marked, payloads = {}, {}
    for nm, s in S.items():
        payloads[nm] = rng.randint(0, 2, s.nbits)
        try:
            t1 = time.time()
            marked[nm] = B.fit_len(s.embed(host, payloads[nm]), len(host))
            print(f"  embedded {nm} [{time.time() - t1:.1f}s]", flush=True)
        except Exception as e:
            print(f"  ! embed failed for {nm}: {e}")
            traceback.print_exc()

    # ---- operating points ----------------------------------------------------------
    sigs = {}                                   # (system, op, target) -> (signal, scale, status, quality)
    host_q = B.quality(host, host, full=True)
    for nm in S:
        sigs[(nm, "null", "")] = (host, 0.0, "ok", host_q)
    targets = [float(t) for t in a.targets.split(",") if t]
    for nm, m in marked.items():
        z = B.apply_scale(host, m, 1.0)
        sigs[(nm, "ship", "")] = (z, 1.0, "ok", B.quality(host, z, full=True))
        for t in targets:
            s, _, st = B.match_scale(host, m, t)
            z = B.apply_scale(host, m, s)
            sigs[(nm, "match", f"{t:g}")] = (z, s, st, B.quality(host, z, full=True))
    for (nm, op, tg), (_, s, st, q) in sigs.items():
        if op != "null":
            print(f"   {nm:9s} {op:5s} {tg:>4s}  s={s:.2f} ({st})  PESQ {q['pesq']:.2f}  "
                  f"SNR {q['snr_db']:.1f} dB", flush=True)

    if not a.no_wavs:
        import soundfile as sf
        wd = os.path.join(WORK, cid)
        os.makedirs(wd, exist_ok=True)
        sf.write(os.path.join(wd, "host.wav"), host, B.SR, subtype="PCM_16")
        for (nm, op, tg), (z, *_r) in sigs.items():
            if op != "null":
                sf.write(os.path.join(wd, f"{nm}_{op}{tg}.wav"), z, B.SR, subtype="PCM_16")

    # ---- attacks x detection -------------------------------------------------------
    out_dir = a.out or os.path.join(RESULTS, cid)
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "rows.csv"), "w", newline="")
    w = csv.writer(fh)
    w.writerow(HEADER)
    kth = {nm: B.binom_threshold(S[nm].nbits) for nm in S}

    def emit(nm, op, tg, scale, status, q, cfg, score, bits):
        payload = payloads[nm]
        nb = len(payload)
        ok = bits is not None
        acc = float(np.mean(np.asarray(bits)[:nb] == payload)) if ok else float("nan")
        k = int(round(acc * nb)) if ok else -1
        w.writerow([cid, utt, nm, op, tg, f"{scale:.4f}", status, f"{q['pesq']:.3f}",
                    f"{q.get('stoi', float('nan')):.3f}", f"{q['snr_db']:.2f}",
                    cfg["family"], cfg["label"], cfg["sev"], f"{score:.5f}", f"{acc:.4f}", nb,
                    kth[nm], int(k >= kth[nm]) if ok else "",
                    "".join(map(str, np.asarray(bits).astype(int).tolist())) if ok else "",
                    "".join(map(str, payload.astype(int).tolist()))])

    all_cfgs = [dict(family="none", label="clean", param=None, sev=-1, src="none")] + cfgs
    for i, cfg in enumerate(all_cfgs):
        null_attacked = UNSET                            # the host is attacked once, shared by all systems
        for (nm, op, tg), (z, s, st, q) in sigs.items():
            if op == "null" and null_attacked is not UNSET:
                y2 = null_attacked
            else:
                try:
                    y2 = z if cfg["family"] == "none" else B.run_attack(cfg, z)
                except Exception as e:
                    print(f"  attack {cfg['family']}/{cfg['label']} failed: {str(e)[:80]}")
                    y2 = None
                if op == "null":
                    null_attacked = y2
            if y2 is None:
                continue
            try:
                score, bits = S[nm].detect(y2)
            except Exception as e:
                score, bits = float("nan"), None
                print(f"  detect {nm} failed after {cfg['family']}/{cfg['label']}: {str(e)[:80]}")
            emit(nm, op, tg, s, st, q, cfg, score, bits)
        fh.flush()
        if (i + 1) % 10 == 0:
            print(f"   attacks {i + 1}/{len(all_cfgs)} [{time.time() - t0:.0f}s]", flush=True)
    fh.close()

    json.dump(dict(clip_id=cid, utt=utt, speaker=spec.get("speaker"), path=spec["path"],
                   mpeg_key=5000 + a.index, mpeg_strength_db=a.strength_db, targets=targets,
                   systems={nm: dict(nbits=int(S[nm].nbits), k_threshold=int(kth[nm]),
                                     fpr=B.binom_tail(int(S[nm].nbits), int(kth[nm])),
                                     **getattr(S[nm], "info", {}))
                            for nm in S}),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"done [{time.time() - t0:.0f}s] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
