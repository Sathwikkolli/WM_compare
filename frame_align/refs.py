"""
frame_align/refs.py -- build the reference audio a frame is searched against.

Three reference kinds:

  native      the clip as-is (~9-27.8 s). Phase 1 uses only this.
  60, 180     seconds. Emilia has no clips this long, so they are BUILT by
              joining other clips around the home clip.

WHY LONG REFERENCES EXIST AT ALL

False-alarm rate should scale with the number of candidate positions in the
search space. A 200 ms frame against a 10 s reference has ~50 places to go wrong;
against 180 s, ~900. Open item #3 of the bake-off records that nothing above
27.8 s has ever been tested, and the client chain is minutes long.

WHY A PADDED REFERENCE IS NOT A REAL LONG RECORDING -- READ THIS

`align_bench/make_clips.py` refuses concatenation on purpose. This file breaks
that rule knowingly, and the cost is real: speaker, room and channel change at
every junction, which is not what a single long recording looks like. A padded
180 s reference may be EASIER than a real one (each segment is spectrally
distinct, so wrong-position matches are less plausible) or HARDER (more
uncorrelated material to throw up spurious peaks). We do not know which.

Every padded row carries ref_kind=padded in the raw data. If the long-reference
numbers disagree with native, concatenation is a live explanation and has to be
ruled out before the result means anything.

Junctions get a 10 ms raised-cosine crossfade. That is not cosmetic: a butt
splice produces a click, a click is a strong broadband landmark, and free
landmarks would make the long-reference numbers look better than they are.
The crossfade is the conservative choice.

  Side effect: the home clip's first and last 10 ms are attenuated by the
  crossfade. `home_xfade_n` is returned so score_frames.py can exclude frames
  that touch it. At 10 ms out of a >=9 s clip this is a fraction of a percent of
  draws, but it is recorded rather than assumed negligible.

OFFSET CONVENTION

Methods report `ref_index - dist_index`. The frame is dist and its sample 0 sits
at reference index `home_start + start_sample`, so that sum IS the truth. No sign
reasoning required anywhere downstream.

Usage:
    python refs.py                  # self-test: verify home_start is recoverable
    python refs.py --probe          # does Emilia have genuinely long clips?
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

WORK_SR = 16000
XFADE_MS = 10.0

EMILIA_CSV = os.environ.get(
    "EMILIA_CSV",
    "/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv",
)


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


# --------------------------------------------------------------------------- #
#  joining
# --------------------------------------------------------------------------- #
def _xfade_concat(pieces, xf_n):
    """Join with an equal-power-ish raised-cosine crossfade.

    Returns (audio, starts) where starts[i] is the index in the output at which
    pieces[i]'s ORIGINAL sample 0 lands. Overlap means this is not a running sum
    of lengths, and getting it wrong would offset every ground truth by 10 ms per
    junction -- small enough to survive the 50 ms bar and poison the 1 ms one.
    """
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, dtype="float32"), []

    out = pieces[0].astype("float32").copy()
    starts = [0]
    for p in pieces[1:]:
        p = p.astype("float32")
        k = int(min(xf_n, len(out), len(p)))
        if k <= 0:
            starts.append(len(out))
            out = np.concatenate([out, p])
            continue
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / max(k - 1, 1))).astype("float32")
        joint = out[-k:] * (1.0 - w) + p[:k] * w
        starts.append(len(out) - k)
        out = np.concatenate([out[:-k], joint, p[k:]])
    return out, starts


def _fill(pool, n_needed, rng, take_from_end):
    """Draw >= n_needed samples from `pool`, then trim to exactly n_needed.

    take_from_end=True keeps the TAIL (material that leads into the home clip),
    False keeps the HEAD. Trimming at the far end means the junction next to home
    is never mid-crossfade of an earlier join.
    """
    if n_needed <= 0 or not pool:
        return np.zeros(0, dtype="float32")
    chunks, total = [], 0
    order = rng.permutation(len(pool))
    i = 0
    while total < n_needed:
        y = pool[order[i % len(order)]]
        chunks.append(y)
        total += len(y)
        i += 1
        if i > 4 * len(pool) + 8:        # pool too short to reach the target
            break
    filler, _ = _xfade_concat(chunks, int(XFADE_MS / 1000.0 * WORK_SR))
    if len(filler) < n_needed:           # pad with what we have, do not fabricate
        return filler
    return filler[-n_needed:] if take_from_end else filler[:n_needed]


# --------------------------------------------------------------------------- #
#  the public call
# --------------------------------------------------------------------------- #
def build(kind, home, pool, sr=WORK_SR, seed=0):
    """Reference audio containing `home`.

    kind : "native" or a target length in seconds (int/float)
    home : the clip frames are drawn from
    pool : other clips, used as padding. Never includes home.

    Returns {audio, home_start, ref_kind, target_s, actual_s, n_pieces,
             home_xfade_n, note}.
    """
    home = np.asarray(home, dtype="float32")
    xf_n = int(XFADE_MS / 1000.0 * sr)

    if kind == "native" or kind is None:
        return {"audio": home, "home_start": 0, "ref_kind": "native",
                "target_s": len(home) / sr, "actual_s": len(home) / sr,
                "n_pieces": 1, "home_xfade_n": 0, "note": "clip as-is"}

    target_n = int(float(kind) * sr)
    if target_n <= len(home):
        # Asking for a reference shorter than the clip. Degrade to native and say
        # so rather than truncating the home clip, which would break the truth.
        return {"audio": home, "home_start": 0, "ref_kind": "native",
                "target_s": float(kind), "actual_s": len(home) / sr,
                "n_pieces": 1, "home_xfade_n": 0,
                "note": f"requested {kind}s < clip {len(home)/sr:.1f}s -- used native"}

    rng = np.random.RandomState(seed)
    available = target_n - len(home)
    pre_n = int(rng.randint(0, available + 1))
    post_n = available - pre_n

    pre = _fill(pool, pre_n, rng, take_from_end=True)
    post = _fill(pool, post_n, rng, take_from_end=False)

    pieces = [p for p in (pre, home, post) if len(p)]
    audio, starts = _xfade_concat(pieces, xf_n)
    home_start = starts[1] if len(pre) else starts[0]

    short = len(audio) < 0.95 * target_n
    return {
        "audio": audio,
        "home_start": int(home_start),
        "ref_kind": "padded",
        "target_s": float(kind),
        "actual_s": len(audio) / sr,
        "n_pieces": len(pieces),
        # home is crossfaded on whichever sides actually have a neighbour
        "home_xfade_n": xf_n * ((1 if len(pre) else 0) + (1 if len(post) else 0)),
        "note": (f"padded to {len(audio)/sr:.1f}s"
                 + (f" -- POOL EXHAUSTED, wanted {kind}s" if short else "")),
    }


def true_offset(ref_meta, start_sample):
    """Ground truth for a frame cut at `start_sample` of the home clip."""
    return float(ref_meta["home_start"] + start_sample)


# --------------------------------------------------------------------------- #
#  probe: does Emilia actually need padding?
# --------------------------------------------------------------------------- #
def probe(csv_path=EMILIA_CSV, targets=(60, 180)):
    """Count manifest clips long enough to serve as native long references.

    If this finds enough, the padded path is unnecessary and the run gets
    strictly more trustworthy. Worth the two seconds it takes to check.
    """
    if not os.path.exists(csv_path):
        print(f"manifest not found: {csv_path}")
        print("Set EMILIA_CSV. This must run where the corpus lives.")
        return None
    dur_keys = ("duration", "dur", "length", "seconds", "duration_s")
    durs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = [c.strip() for c in (rdr.fieldnames or [])]
        dcol = next((c for c in cols if c.lower() in dur_keys), None)
        if dcol is None:
            print(f"no duration column in manifest; cols={cols}")
            return None
        for r in rdr:
            try:
                durs.append(float(r.get(dcol) or "nan"))
            except ValueError:
                pass
    durs = np.array([d for d in durs if np.isfinite(d)])
    if not len(durs):
        print("no usable durations in manifest")
        return None
    print(f"manifest clips: {len(durs)}   max {durs.max():.1f}s   "
          f"median {np.median(durs):.1f}s")
    out = {}
    for t in targets:
        n = int((durs >= t).sum())
        out[t] = n
        verdict = "padding NOT needed" if n >= 30 else "padding needed"
        print(f"  >= {t:>3d}s : {n:>6d} clips   <- {verdict}")
    return out


# --------------------------------------------------------------------------- #
#  self-test
# --------------------------------------------------------------------------- #
def _selftest():
    """Verify home_start is where we say it is.

    If the crossfade bookkeeping in _xfade_concat is off by even one junction,
    every ground truth in the padded conditions is wrong by ~10 ms -- which would
    pass the 50 ms bar and quietly destroy the 1 ms results. Synthetic audio, so
    this runs anywhere, no Emilia needed.
    """
    sys.path.insert(0, os.path.join(ROOT, "align_bench"))
    import methods as M

    sr = WORK_SR
    rng = np.random.RandomState(0)

    def blip(dur_s, f0):
        n = int(dur_s * sr)
        t = np.arange(n) / sr
        y = (rng.randn(n) * 0.05).astype("float32")
        y += (0.3 * np.sin(2 * np.pi * f0 * t)).astype("float32")
        return (y * np.linspace(0.4, 1.0, n)).astype("float32")

    home = blip(9.0, 500.0)
    pool = [blip(9.0, f) for f in (700.0, 1100.0, 1900.0, 2600.0)]

    print("self-test: recover home_start from a built reference\n")
    ok_all = True
    for kind in ("native", 60, 180):
        meta = build(kind, home, pool, sr=sr, seed=1)
        # cut a frame from the middle of home and see where the reference says
        # it lives; the answer must be home_start + start
        start = 3 * sr
        frame = home[start:start + sr]
        truth = true_offset(meta, start)
        out = M.run("gcc_phat", meta["audio"], frame, sr, sign=1.0)
        err_ms = abs(out["offset"] - truth) / sr * 1000.0
        ok = err_ms < 1.0
        ok_all &= ok
        print(f"  {str(kind):>6s}  actual={meta['actual_s']:6.1f}s  "
              f"pieces={meta['n_pieces']:>2d}  home_start={meta['home_start']:>8d}  "
              f"err={err_ms:7.3f}ms  {'OK' if ok else 'FAIL'}")
        print(f"          {meta['note']}")

    print("\n" + ("all conditions OK" if ok_all else
                  "FAILED -- do not run the benchmark until this passes"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe(get_arg(sys.argv, "--csv", EMILIA_CSV))
    else:
        sys.exit(_selftest())
