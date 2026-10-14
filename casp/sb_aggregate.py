"""
casp/sb_aggregate.py -- tables and figures for the single-bit experiment.

    python sb_aggregate.py            # after sbatch sb.sbatch finishes

Why equal QUALITY: at equal energy this detector rewards putting change into
weak cells (log-ratio statistic), which is exactly what sounds worst. The
fair comparison is therefore detection at equal PESQ (or SNR if PESQ is absent).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("CASP_SB_RUN", "2026-10-14_casp-sb")
RESULTS = os.environ.get("CASP_SB_DIR", os.path.join(ROOT, "results", RUN))
ARMS = ["aware_shape", "casp", "johnston", "flat_abs"]
ATT = ["clean", "awgn_20", "awgn_10", "awgn_5", "lp_3500", "lp_2500", "lp_2000", "mp3_32", "mp3_16", "opus_12"]
COL = {"aware_shape": "tab:red", "casp": "tab:blue", "johnston": "tab:green", "flat_abs": "tab:gray"}
Z = 3.3


def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "sb_rows.csv")))
    if not files:
        raise SystemExit(f"no sb_rows.csv under {RESULTS}")
    R = pd.concat([pd.read_csv(f) for f in files])
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, "*", "sb_meta.json")))]
    clips = sorted(R.clip_id.unique())
    wm, neg = R[R.kind == "wm"].copy(), R[R.kind == "neg"]
    qcol = "pesq" if wm.pesq.notna().any() else "snr_db"
    wm["absz"] = wm.z_std.abs()
    wm["db"] = 10 * np.log10(wm.g ** 2)                   # energy re AWARE-6dB, dB (g is an amplitude factor)

    L = [f"# Single-bit watermark: four budgets, equal quality ({len(clips)} clips)\n",
         f"Clips: {', '.join(clips)}. Detected if |z_std| >= {Z} (wrong-key null per file). "
         f"Quality axis: **{qcol}**. g = 1 is the energy of AWARE's 6 dB rule; g = 0.1 is -20 dB.\n"]

    # false positives
    fp = neg.groupby("attack").agg(n=("detected", "size"), fp=("detected", "sum"),
                                   max_abs_z=("z_std", lambda s: s.abs().max())).reindex(ATT).reset_index()
    L += ["## Unwatermarked audio, same attacks, true key\n", md(fp, "{:.2f}"), ""]

    # CASP natural levels
    nat = pd.DataFrame([dict(clip=m["clip_id"], **m["casp_natural_g"]) for m in metas])
    L += ["## CASP's own just-noticeable level, as strength g (energy re AWARE 6 dB)\n", md(nat, "{:.3f}"), ""]

    # smallest g that stays detected
    rows = []
    for (c, arm, att), d in wm.groupby(["clip_id", "arm", "attack"]):
        d = d.sort_values("g", ascending=False)
        ok = d.detected.values
        k = 0
        while k < len(ok) and ok[k]:
            k += 1
        if k == 0:
            rows.append(dict(clip=c, arm=arm, attack=att, g_min=np.nan, q_at_min=np.nan))
        else:
            r = d.iloc[k - 1]
            rows.append(dict(clip=c, arm=arm, attack=att, g_min=r.g, q_at_min=r[qcol]))
    G = pd.DataFrame(rows)
    gt = G.pivot_table(index="arm", columns="attack", values="g_min", aggfunc="median").reindex(ARMS)[ATT]
    gt_db = 20 * np.log10(gt)
    L += ["## Weakest strength that is still detected (median over clips, dB re AWARE-6dB energy; lower = quieter)\n",
          "Empty = not detected even at g = 1 in most clips.\n", md(gt_db.reset_index(), "{:.1f}"), ""]
    qt = G.pivot_table(index="arm", columns="attack", values="q_at_min", aggfunc="mean").reindex(ARMS)[ATT]
    L += [f"## {qcol} at that weakest detected strength (mean over clips; higher = cleaner for the same detection)\n",
          md(qt.reset_index(), "{:.2f}"), ""]

    # equal-quality detection rate
    levels = [4.2, 4.35, 4.5, 4.6] if qcol == "pesq" else [15.0, 20.0, 25.0, 30.0]
    L += [f"## Detection rate at equal {qcol} (interpolated per clip; share of clips detected)\n"]
    for q0 in levels:
        t = {}
        for arm in ARMS:
            for att in ATT:
                hits = []
                for c in clips:
                    d = wm[(wm.clip_id == c) & (wm.arm == arm) & (wm.attack == att)].dropna(subset=[qcol])
                    if len(d) < 2:
                        continue
                    d = d.sort_values(qcol)
                    x, z = d[qcol].values, d.absz.values
                    if q0 < x.min() or q0 > x.max():
                        continue
                    hits.append(float(np.interp(q0, x, z) >= Z))
                t[(arm, att)] = np.mean(hits) if hits else np.nan
        tab = pd.DataFrame({att: [t[(arm, att)] for arm in ARMS] for att in ATT}, index=ARMS).reset_index()
        L += [f"**{qcol} = {q0}**\n", md(tab.rename(columns={"index": "arm"}), "{:.2f}"), ""]

    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L))
    G.to_csv(os.path.join(RESULTS, "g_min.csv"), index=False)
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(20, 7.5), sharey=True)
    for a_, att in zip(ax.ravel(), ATT):
        for arm in ARMS:
            d = wm[(wm.arm == arm) & (wm.attack == att)].groupby("g").agg(
                q=(qcol, "mean"), det=("detected", "mean")).reset_index()
            xs = d.q if qcol == "pesq" and d.q.notna().any() else d.g.map(lambda v: 10 * np.log10(v ** 2))
            a_.plot(xs, d.det, "o-", ms=3, color=COL[arm], label=arm)
        a_.set(title=att, ylim=(-0.03, 1.03))
        a_.grid(alpha=0.3)
    for a_ in ax[1]:
        a_.set_xlabel(qcol if qcol == "pesq" and wm.pesq.notna().any() else "energy re AWARE 6 dB (dB)")
    for a_ in ax[:, 0]:
        a_.set_ylabel("share of clips detected")
    ax[0, 0].legend()
    fig.suptitle("Single bit: detection vs quality, per attack")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_sb_detection.png"), dpi=125)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for arm in ARMS:
        d = wm[(wm.arm == arm) & (wm.attack == "clean")].groupby("g").absz.median().reset_index()
        ax[0].semilogy(10 * np.log10(d.g ** 2), d.absz, "o-", color=COL[arm], label=arm)
        d2 = wm[(wm.arm == arm) & (wm.attack == "clean")].groupby("g")[qcol].mean().reset_index()
        ax[1].plot(10 * np.log10(d2.g ** 2), d2[qcol], "o-", color=COL[arm], label=arm)
    ax[0].axhline(Z, color="k", ls=":")
    ax[0].set(xlabel="energy re AWARE 6 dB (dB)", ylabel="median |z_std| (clean)", title="Detection strength vs energy")
    ax[1].set(xlabel="energy re AWARE 6 dB (dB)", ylabel=qcol, title="Quality vs energy")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_sb_energy.png"), dpi=125)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
