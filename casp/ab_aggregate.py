"""
casp/ab_aggregate.py -- tables and figures for the stock-AWARE vs AWARE+CASP run.

    python ab_aggregate.py            # after the sbatch array finishes

Reads results/<RUN>/<clip>/ab_rows.csv + ab_meta.json, writes summary.md and
figures next to them. With n = 5 clips this is a smoke-level comparison: per-clip
wins/ties/losses are reported, not significance tests.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("CASP_AB_RUN", "2026-10-14_casp-ab")
RESULTS = os.environ.get("CASP_AB_DIR", os.path.join(ROOT, "results", RUN))
ARMS = ["AWARE", "AWARE+CASP"]
ORDER = ["clean", "mp3_64", "mp3_32", "awgn_20", "awgn_10", "lp_3500", "lp_2500"]


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "ab_rows.csv")))
    if not files:
        raise SystemExit(f"no ab_rows.csv under {RESULTS}")
    rows = pd.concat([pd.read_csv(f) for f in files])
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, "*", "ab_meta.json")))]
    clips = sorted(rows["clip_id"].unique())
    wm = rows[rows.kind == "wm"]
    neg = rows[rows.kind == "neg"]
    L = [f"# Stock AWARE vs AWARE + CASP budget: {len(clips)} clips\n",
         f"Clips: {', '.join(clips)}. Same frozen AWARE detector in both arms; detected = conf >= 0.5.\n"]

    # quality
    q = []
    for m in metas:
        for arm in ARMS:
            q.append(dict(clip=m["clip_id"], arm=arm, **m["quality"][arm]))
    qd = pd.DataFrame(q)
    qs = qd.groupby("arm")[["pesq", "stoi", "snr_db"]].agg(["mean", "std"]).reindex(ARMS)
    qs.columns = [f"{a}_{b}" for a, b in qs.columns]
    L += ["## Quality (watermarked vs original)\n", md(qs.reset_index()), ""]
    pv = qd.pivot(index="clip", columns="arm", values="pesq")
    has_pesq = bool(pv.notna().all().all())
    if has_pesq:
        pv["delta"] = pv["AWARE+CASP"] - pv["AWARE"]
        L += ["Per clip PESQ:\n", md(pv.reset_index()), ""]
    keys = ("kappa", "eta_stock_median", "eta_casp_median", "frac_capped", "E_ratio", "align_corr")
    b = pd.DataFrame([dict(clip=m["clip_id"], **{k: m["budget"].get(k) for k in keys}) for m in metas])
    L += ["## Budget actually used by arm B\n", md(b), ""]

    # robustness
    t = []
    for att in ORDER:
        d = wm[wm.attack == att]
        r = dict(attack=att)
        for arm in ARMS:
            x = d[d.arm == arm]
            r[f"{arm} bit_acc"] = x.bit_acc.mean()
            r[f"{arm} det"] = f"{int(x.detected.sum())}/{len(x)}"
        pa = d[d.arm == "AWARE"].set_index("clip_id").bit_acc
        pc = d[d.arm == "AWARE+CASP"].set_index("clip_id").bit_acc
        diff = (pc - pa).round(6)
        r["CASP vs AWARE (W/T/L)"] = f"{int((diff > 0).sum())}/{int((diff == 0).sum())}/{int((diff < 0).sum())}"
        n = neg[neg.attack == att]
        r["false pos (clean audio)"] = f"{int(n.detected.sum())}/{len(n)}"
        t.append(r)
    T = pd.DataFrame(t)
    L += ["## Robustness (mean over clips)\n", md(T), "",
          "W/T/L = clips where AWARE+CASP bit accuracy is higher / equal / lower than stock AWARE.\n",
          "## Per-clip bit accuracy\n"]
    pcl = wm.pivot_table(index=["attack", "clip_id"], columns="arm", values="bit_acc").reset_index()
    pcl["attack"] = pd.Categorical(pcl["attack"], ORDER, ordered=True)
    L += [md(pcl.sort_values(["attack", "clip_id"])), ""]
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L))
    T.to_csv(os.path.join(RESULTS, "robustness_table.csv"), index=False)
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(ORDER))
    for i, arm in enumerate(ARMS):
        col = ["tab:red", "tab:blue"][i]
        m = [wm[(wm.attack == a) & (wm.arm == arm)].bit_acc.mean() for a in ORDER]
        dr = [wm[(wm.attack == a) & (wm.arm == arm)].detected.mean() for a in ORDER]
        ax[0].bar(x + (i - 0.5) * 0.38, m, 0.38, color=col, alpha=0.65, label=arm)
        ax[1].bar(x + (i - 0.5) * 0.38, dr, 0.38, color=col, alpha=0.65, label=arm)
        for j, a in enumerate(ORDER):
            v = wm[(wm.attack == a) & (wm.arm == arm)].bit_acc.values
            ax[0].scatter(np.full(len(v), j + (i - 0.5) * 0.38), v, s=12, color=col,
                          edgecolor="k", lw=0.4, zorder=3)
    ax[0].axhline(0.5, color="gray", ls=":")
    ax[0].set(ylim=(0.4, 1.02), ylabel="bit accuracy (20 bits)", title="Bit accuracy (dots = clips)")
    ax[1].set(ylim=(0, 1.05), ylabel="fraction detected (conf >= 0.5)", title="Detection rate")
    for a in ax:
        a.set_xticks(x, ORDER, rotation=30)
        a.legend()
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_robustness.png"), dpi=130)

    if has_pesq:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for c, r in pv.iterrows():
            ax[0].plot([0, 1], [r["AWARE"], r["AWARE+CASP"]], "o-", label=c)
        ax[0].set_xticks([0, 1], ARMS)
        ax[0].set(ylabel="PESQ", title="PESQ per clip")
        ax[0].legend(fontsize=7)
        sv = qd.pivot(index="clip", columns="arm", values="stoi")
        for c, r in sv.iterrows():
            ax[1].plot([0, 1], [r["AWARE"], r["AWARE+CASP"]], "o-")
        ax[1].set_xticks([0, 1], ARMS)
        ax[1].set(ylabel="STOI", title="STOI per clip")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_quality.png"), dpi=130)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
