"""
fsss/exp_v12_metapxyl_compare.py -- robustness comparison, stock AWARE vs the
key-driven salient-hopping staircase, on 10 Emilia clips.

CONFIGS
  stock      AWARE original as shipped (load() defaults: native band, tol 6 = quiet).
             The literal "AWARE original watermark", no hopping.
  n1_wide    no-hop control at band 500-4000, tol -6 (matched band + loudness to the
             staircase, but NO hopping). Isolates the hopping effect from the
             loudness/band difference. (drop with --no-control)
  staircase  librosa-anchored subband HOPPING, N=3, band 500-4000, tol -6.

Read: stock vs staircase = "as-deployed" (BUT confounded -- staircase is louder and
wider-band, and robustness tracks loudness). n1_wide vs staircase = the honest
"did hopping help" comparison at matched band + strength. Tols are printed in the
header so the loudness confound stays explicit. Detection = STOCK detector for all,
with the detector-band fix (band set on both embedder and detector per config).

ATTACKS
  METAPXYL-proxy mastering chain: dynamic_compression, echo, mp3, quantization,
  lowpass, gaussian_noise.
  Important VoxWatermark additions: time_stretch, time_jitter (desync -- the family
  content-anchoring targets), highpass (band discriminator, strips 500-1000),
  encodec (neural-codec ceiling), background_noise (realistic additive), opus.
  Each swept across its full VOX_GRID strengths. Attacks with a missing dep SKIP.

No PESQ, no figures -- results only.
    conda activate wmcompare
    python -m fsss.exp_v12_metapxyl_compare
    python -m fsss.exp_v12_metapxyl_compare --nclips 5 --no-control --attacks mp3,lowpass
"""

import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark

import vox_attacks
from fsss.staircase import StaircaseAWAREEmbedder
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR

WM_BITS = 20
METAPXYL_PROXY = ["dynamic_compression", "echo", "mp3", "quantization",
                  "lowpass", "gaussian_noise"]
VOX_IMPORTANT = ["time_stretch", "time_jitter", "highpass", "encodec",
                 "background_noise", "opus"]
DEFAULT_ATTACKS = METAPXYL_PROXY + VOX_IMPORTANT
DET_CONF = 0.5
OUT_DIR = os.path.join(BASE, "fsss_out")


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def get_list(argv, flag, default, cast):
    if flag in argv:
        return [cast(x) for x in argv[argv.index(flag) + 1].split(",") if x.strip()]
    return default


def parse_band(tok):
    lo, hi = tok.split("-")
    return int(lo), int(hi)


def bit_acc(bits, pattern):
    p = np.asarray(pattern).astype(int).ravel()
    b = np.asarray(bits).astype(int).ravel()
    n = min(len(b), len(p))
    return float(np.mean(b[:n] == p[:n])) if n else float("nan")


def build_model(embedder, cfg, key, rate, seglen):
    """Return the embedder object for a config (stock = the plain embedder)."""
    kind = cfg["kind"]
    if kind == "stock":
        return embedder
    band = cfg["band"]
    if kind == "n1":
        m = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=1, band_range=band,
            segment_mode="fixed", segment_len_s=seglen, tolerance_db=cfg["tol"])
    else:  # hop
        m = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=cfg["nbands"], band_range=band,
            segment_mode="librosa", anchor="librosa_flux", anchor_rate=rate,
            segment_len_s=seglen, tolerance_db=cfg["tol"])
    m.embedding_bands = band
    return m


def main(argv):
    seed = get_arg(argv, "--seed", 0, int)
    dur = get_arg(argv, "--dur", 5.0, float)
    nclips = get_arg(argv, "--nclips", 10, int)
    tol = get_arg(argv, "--tol", -6.0, float)          # staircase / n1_wide loudness
    nbands = get_arg(argv, "--nbands", 3, int)
    seglen = get_arg(argv, "--seglen", 0.5, float)
    rate = get_arg(argv, "--rate", 3.5, float)
    key = get_arg(argv, "--key", "thesis")
    band = parse_band(get_arg(argv, "--band", "500-4000"))
    attacks = get_list(argv, "--attacks", DEFAULT_ATTACKS, str)
    control = "--no-control" not in argv
    pesq_only = "--pesq-only" in argv          # skip attacks, PESQ + clean only (fast)
    do_pesq = pesq_only or "--pesq" in argv

    clips = pick_clips(EMILIA_CSV, nclips)
    if not clips:
        print("no Emilia clips found")
        return
    audios = [load_16k(c)[:int(round(dur * WORK_SR))] for c in clips]
    bits = np.random.default_rng(seed).integers(0, 2, size=WM_BITS, dtype=np.int32)

    embedder, detector = load()

    pesq_metric = None
    if do_pesq:
        try:
            from aware.metrics.audio import PESQ
            pesq_metric = PESQ()
        except Exception as e:
            print(f"PESQ unavailable ({e}); continuing without it")
            do_pesq = False

    stock_band = tuple(int(x) for x in embedder.embedding_bands)
    stock_tol = float(getattr(embedder, "tolerance_db", float("nan")))

    configs = [{"name": "stock", "kind": "stock", "band": stock_band, "tol": stock_tol}]
    if control:
        configs.append({"name": "n1_wide", "kind": "n1", "band": band, "tol": tol,
                        "nbands": 1})
    configs.append({"name": "staircase", "kind": "hop", "band": band, "tol": tol,
                    "nbands": nbands})

    print(f"clips    : {len(audios)}   attacks: {attacks}")
    print(f"configs  :")
    for c in configs:
        extra = f"  N={c.get('nbands')}" if c["kind"] == "hop" else \
                ("  no-hop" if c["kind"] == "n1" else "  no-hop (original)")
        print(f"    {c['name']:10s} band={c['band'][0]}-{c['band'][1]:<5d} "
              f"tol={c['tol']:>4}{extra}")
    print("NOTE: robustness tracks loudness (lower tol = louder). stock is quiet "
          "(tol~6), staircase/n1_wide loud (tol-6) -> stock-vs-staircase is loudness+"
          "band confounded; use n1_wide-vs-staircase to isolate hopping.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "exp_v12_metapxyl_compare.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["clip", "config", "band", "tol", "attack", "param",
                     "bit_acc", "conf", "detected", "pesq"])

    # (config, attack) -> list of (acc, det)
    agg = {}
    pesq_agg = {c["name"]: [] for c in configs}   # config -> [clean PESQ per clip]

    def rec(cn, atk, acc, det):
        agg.setdefault((cn, atk), []).append((acc, det))

    for ci, (clip, audio) in enumerate(zip(clips, audios)):
        print(f"clip {ci+1}/{len(audios)}: {os.path.basename(clip)}")
        for cfg in configs:
            cn = cfg["name"]
            detector.embedding_bands = cfg["band"]
            try:
                model = build_model(embedder, cfg, key, rate, seglen)
                wm = embed_watermark(audio, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=model)
            except Exception as e:
                print(f"    {cn:10s} EMBED FAILED: {e}")
                continue
            bandstr = f"{cfg['band'][0]}-{cfg['band'][1]}"
            pat, conf = detect_watermark(wm, WORK_SR, detector)
            acc = bit_acc(bits, pat)
            pv = float("nan")
            if do_pesq:
                try:
                    m = min(len(wm), len(audio))
                    pv = float(pesq_metric(wm[:m], audio[:m], WORK_SR))
                    pesq_agg[cn].append(pv)
                except Exception as e:
                    print(f"    {cn:10s} PESQ failed: {e}")
            rec(cn, "clean", acc, int(conf >= DET_CONF))
            writer.writerow([ci, cn, bandstr, cfg["tol"], "clean", "", round(acc, 4),
                             round(float(conf), 4), int(conf >= DET_CONF),
                             "" if np.isnan(pv) else round(pv, 4)])
            if pesq_only:
                print(f"    {cn:10s} clean  acc={acc:.3f}  conf={float(conf):.3f}  "
                      f"PESQ={pv:.3f}")
                continue
            for atk in attacks:
                for label, param in vox_attacks.VOX_GRID.get(atk, []):
                    try:
                        wa = vox_attacks.apply(atk, param, wm.astype("float32"), WORK_SR)
                    except Exception:
                        continue
                    if wa is None:
                        continue
                    pat, conf = detect_watermark(wa, WORK_SR, detector)
                    acc = bit_acc(bits, pat)
                    rec(cn, atk, acc, int(conf >= DET_CONF))
                    writer.writerow([ci, cn, bandstr, cfg["tol"], atk, label,
                                     round(acc, 4), round(float(conf), 4),
                                     int(conf >= DET_CONF), ""])
    fout.close()

    # ----------------------------------------------------------------------- #
    cfg_names = [c["name"] for c in configs]

    def pesq_mean(cn):
        vals = [v for v in pesq_agg.get(cn, []) if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    if do_pesq:
        print("\n" + "=" * 60)
        print("CLEAN PESQ  (watermarked vs original, mean over clips; higher = more "
              "imperceptible)")
        for c in configs:
            cn = c["name"]
            print(f"  {cn:10s} band={c['band'][0]}-{c['band'][1]:<5d} tol={c['tol']:>4}"
                  f"   PESQ={pesq_mean(cn):.3f}")
        print("  (loudness check: stock tol6 should be highest; n1_wide tol-6 full-band "
              "lowest; staircase tol-6 hop in between — hopping perturbs 1/N bins/frame.)")

    if pesq_only:
        print("\n(--pesq-only: attacks skipped)")
        print(f"wrote {csv_path}")
        return

    def det_total(cn, atk):
        rows = agg.get((cn, atk), [])
        return sum(r[1] for r in rows), len(rows)

    def acc_mean(cn, atk):
        rows = agg.get((cn, atk), [])
        a = [r[0] for r in rows if not np.isnan(r[0])]
        return float(np.mean(a)) if a else float("nan")

    colw = 13
    header = f"  {'attack':20s}" + "".join(f"{n:>{colw}s}" for n in cfg_names)

    print("\n" + "=" * len(header))
    print("DETECTION SURVIVAL  (detected / total, conf>=0.5, over clips x strengths)")
    print(header)
    order = ["clean"] + attacks
    for atk in order:
        line = f"  {atk:20s}"
        for cn in cfg_names:
            d, t = det_total(cn, atk)
            line += f"{f'{d}/{t}':>{colw}s}"
        print(line)
    # overall (attacks only, excl clean)
    line = f"  {'TOTAL (attacks)':20s}"
    for cn in cfg_names:
        d = sum(det_total(cn, a)[0] for a in attacks)
        t = sum(det_total(cn, a)[1] for a in attacks)
        line += f"{f'{d}/{t}':>{colw}s}"
    print("  " + "-" * (len(header) - 2))
    print(line)

    print("\n" + "=" * len(header))
    print("MEAN BIT-ACCURACY")
    print(header)
    for atk in order:
        line = f"  {atk:20s}"
        for cn in cfg_names:
            line += f"{acc_mean(cn, atk):>{colw}.3f}"
        print(line)

    print(f"\nwrote {csv_path}")
    print("read: n1_wide vs staircase = did hopping help at matched band+loudness "
          "(same tol-6, both 500-4000). stock = original reference (quiet, so higher "
          "survival is partly loudness). Watch desync (time_stretch/jitter) + highpass "
          "+ encodec; mastering-proxy (compress/echo/mp3/lowpass/noise) is the core.")


if __name__ == "__main__":
    main(sys.argv[1:])
