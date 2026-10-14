"""
casp/validate.py -- is the Python CASP port behaving, and what is "1 unit of audible"?

Writes casp/<out>/validation.json and validation.png:

  1. level growth      : IR peak and rms vs input level (20..80 dB SPL), 1 kHz tone.
                         Should be strongly compressive (adaptation loops).
  2. intensity JND ref : D(dL) for dL in {0.25..3} dB on the 1 kHz / 60 dB SPL tone
                         (the paper's fitting condition). D_JND = D(JND_DB).
  3. level independence: D(1 dB) at 30..80 dB SPL. A roughly flat curve is
                         Weber's law, which the model is built to reproduce.
  4. sample-rate check : D for the same stimuli at 16 kHz vs 44.1 kHz
                         (paper: 44.1 kHz; we run at 16 kHz for cost).

JND_DB is an ASSUMED value (the paper fits the model to Viemeister & Bacon 1988
at 60 dB but the number is only in their Fig. 4). Default 1.0 dB; sensitivity is
reported by using D at 0.5 and 1.5 dB as well.
"""
import argparse
import json
import os

import numpy as np

from casp_front import CASP, P_REF

HERE = os.path.dirname(os.path.abspath(__file__))
JND_DB = 1.0


def tone(fs, level_db, dur=0.2, pre=0.15, post=0.15, f=1000.0, ramp=0.01):
    n = int(dur * fs)
    t = np.arange(n) / fs
    w = np.ones(n)
    r = int(ramp * fs)
    w[:r] = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    w[-r:] = w[:r][::-1]
    s = np.sqrt(2) * np.sin(2 * np.pi * f * t) * w * P_REF * 10 ** (level_db / 20)
    return np.concatenate([np.zeros(int(pre * fs)), s, np.zeros(int(post * fs))]).astype(np.float32)


def calibrate(casp, jnd_db=JND_DB, level_db=60.0):
    """D_JND: IR distance for a `jnd_db` increment on the 1 kHz / 60 dB SPL tone
    (the paper's fitting condition). One audibility unit (d' ~ 1)."""
    fs = casp.fs
    a = tone(fs, level_db)
    return float(casp.dist(casp.ir(a), casp.ir(a * 10 ** (jnd_db / 20))))


def run(fs, threads, levels, dls, quick=False):
    c = CASP(fs=fs, threads=threads)
    out = {"fs": fs, "fs_ir": c.fs_ir, "n_channels": int(len(c.fc))}
    # level growth
    ir = []
    for L in levels:
        a = c.ir(tone(fs, L))[0]
        ir.append((L, float(a.abs().max()), float(a.pow(2).mean().sqrt())))
    out["level_growth"] = ir
    # intensity-increment distances at 60 dB
    base = tone(fs, 60)
    ib = c.ir(base)
    out["D_60"] = {str(dl): float(c.dist(ib, c.ir(base * 10 ** (dl / 20)))) for dl in dls}
    # Weber-ish: D(1 dB) across levels
    if not quick:
        wl = []
        for L in levels:
            a = tone(fs, L)
            wl.append((L, float(c.dist(c.ir(a), c.ir(a * 10 ** (1.0 / 20))))))
        out["D_1dB_vs_level"] = wl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fs-compare", action="store_true", help="also run at 44.1 kHz")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    levels = [20, 30, 40, 50, 60, 70, 80]
    dls = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    res = {"jnd_db_assumed": JND_DB, "runs": []}
    res["runs"].append(run(16000, a.threads, levels, dls, a.quick))
    if a.fs_compare:
        res["runs"].append(run(44100, a.threads, levels, dls, a.quick))
    r0 = res["runs"][0]
    res["D_JND"] = r0["D_60"][str(JND_DB)]
    res["D_JND_lo"] = r0["D_60"]["0.5"]
    res["D_JND_hi"] = r0["D_60"]["1.5"]
    with open(os.path.join(a.out, "validation.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in res["runs"]:
        lab = f"fs={r['fs']}"
        L, pk, rms = zip(*r["level_growth"])
        ax[0].plot(L, 20 * np.log10(np.maximum(rms, 1e-9)), "o-", label=lab)
        ax[1].plot([float(k) for k in r["D_60"]], list(r["D_60"].values()), "o-", label=lab)
        if "D_1dB_vs_level" in r:
            l2, d2 = zip(*r["D_1dB_vs_level"])
            ax[2].plot(l2, d2, "o-", label=lab)
    ax[0].set(xlabel="tone level (dB SPL)", ylabel="IR rms (dB re 1 MU)",
              title="1  Level growth (compressive?)")
    ax[1].axvline(JND_DB, color="k", ls=":", label=f"assumed JND {JND_DB} dB")
    ax[1].set(xlabel="intensity increment (dB)", ylabel="||dIR||",
              title="2  D vs increment, 1 kHz @ 60 dB SPL")
    ax[2].set(xlabel="tone level (dB SPL)", ylabel="||dIR|| for +1 dB",
              title="3  Weber check (flat = Weber)")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "validation.png"), dpi=130)
    print(json.dumps({k: res[k] for k in ("D_JND", "D_JND_lo", "D_JND_hi")}, indent=1))


if __name__ == "__main__":
    main()
