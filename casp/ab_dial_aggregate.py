"""
casp/ab_dial_aggregate.py -- tables and figures for the dial sweep.

    python ab_dial_aggregate.py        # after the ab_dial sbatch array finishes

x-axes are (a) CASP audibility of the REAL embedded watermark (1 = just
noticeable) and (b) PESQ. The question the figures answer: at the same
audibility (or the same PESQ), does the CASP-shaped family keep more bits
under attack than flat AWARE?
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("CASP_DIAL_RUN", "2026-10-14_casp-dial")
RESULTS = os.environ.get("CASP_DIAL_DIR", os.path.join(ROOT, "results", RUN))
HARD = ["awgn_10", "awgn_5", "lp_2500", "lp_2000", "mp3_16", "opus_12"]
ALL_ATT = ["clean"] + HARD
COL = {"AWARE": "tab:red", "CASP": "tab:blue"}


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def interp_at(x, y, x0):
    """y at x0 by linear interpolation over sorted x; NaN outside the range."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if x0 < x[0] or x0 > x[-1]:
        return np.nan
    return float(np.interp(x0, x, y))


def main():
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RESULTS, "*", "dial_rows.csv")))])
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, "*", "dial_meta.json")))]
    clips = sorted(rows.clip_id.unique())
    wm, neg = rows[rows.kind == "wm"], rows[rows.kind == "neg"]

    am = pd.DataFrame([dict(clip_id=m["clip_id"], family=a["family"], dial=a["dial"], A_mean=a["A_mean"],
                            A_max=a["A_max"], pesq=a["quality"]["pesq"], stoi=a["quality"]["stoi"],
                            snr_db=a["quality"]["snr_db"], at_cap=a["frac_at_range_cap"],
                            E_ratio=a["budget"].get("E_ratio"))
                       for m in metas for a in m["arms"]])
    for c in ("pesq", "stoi", "snr_db", "E_ratio"):          # None -> NaN when pesq/pystoi are absent
        am[c] = pd.to_numeric(am[c], errors="coerce")
    piv = wm.pivot_table(index=["clip_id", "family", "dial"], columns="attack", values="bit_acc").reset_index()
    piv["hard_mean"] = piv[HARD].mean(axis=1)
    det = wm.assign(d=wm.detected.astype(float)).pivot_table(index=["clip_id", "family", "dial"],
                                                             columns="attack", values="d").reset_index()
    det["hard_det"] = det[HARD].mean(axis=1)
    D = am.merge(piv, on=["clip_id", "family", "dial"]).merge(
        det[["clip_id", "family", "dial", "hard_det"]], on=["clip_id", "family", "dial"])
    D.to_csv(os.path.join(RESULTS, "dial_all.csv"), index=False)

    g = D.groupby(["family", "dial"]).mean(numeric_only=True).reset_index()
    L = [f"# Dial sweep: AWARE tolerance vs CASP target ({len(clips)} clips)\n",
         f"Clips: {', '.join(clips)}. Same frozen detector. A = CASP audibility of the real watermark "
         "(1 = just noticeable). hard = mean over awgn_10, awgn_5, lp_2500, lp_2000, mp3_16, opus_12.\n",
         "## Per dial point (mean over clips)\n"]
    cols = ["family", "dial", "A_mean", "A_max", "pesq", "stoi", "snr_db", "at_cap", "clean"] + HARD + ["hard_mean", "hard_det"]
    L += [md(g[cols].sort_values(["family", "dial"])), ""]

    fp = neg.groupby("attack").detected.agg(["sum", "count"])
    L += ["## False positives on unwatermarked audio\n",
          md(pd.DataFrame([dict(attack=a, fp=f"{int(r['sum'])}/{int(r['count'])}") for a, r in fp.iterrows()])), ""]

    # equal-audibility / equal-PESQ comparison, per clip
    out = []
    for c in clips:
        d = D[D.clip_id == c]
        ca, aw = d[d.family == "CASP"], d[d.family == "AWARE"]
        for _, r in aw.iterrows():
            row = dict(clip=c, aware_tol=r.dial, A=r.A_mean, pesq=r.pesq, aware_hard=r.hard_mean)
            row["casp_hard_at_A"] = interp_at(ca.A_mean, ca.hard_mean, r.A_mean)
            row["casp_hard_at_PESQ"] = interp_at(ca.pesq, ca.hard_mean, r.pesq) if pd.notna(r.pesq) else np.nan
            out.append(row)
    E = pd.DataFrame(out)
    E["delta_A"] = E.casp_hard_at_A - E.aware_hard
    E["delta_PESQ"] = E.casp_hard_at_PESQ - E.aware_hard
    E.to_csv(os.path.join(RESULTS, "equal_comparison.csv"), index=False)
    L += ["## CASP family minus AWARE at the SAME audibility / SAME PESQ (hard-attack bit accuracy)\n",
          "Only AWARE points that fall inside the CASP family's range are comparable (NaN otherwise).\n",
          md(E.groupby("aware_tol").agg(n_A=("delta_A", "count"), mean_delta_at_A=("delta_A", "mean"),
                                       n_P=("delta_PESQ", "count"), mean_delta_at_PESQ=("delta_PESQ", "mean")).reset_index()),
          "", f"All comparable points: delta at equal A = {E.delta_A.mean():+.3f} "
              f"(n={int(E.delta_A.count())}), at equal PESQ = {E.delta_PESQ.mean():+.3f} (n={int(E.delta_PESQ.count())}).\n",
          "Positive = CASP-shaped budget keeps more bits. 20-bit payload: one bit = 0.05.\n"]
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for a_, att in zip(ax.ravel(), ALL_ATT):
        for fam in ("AWARE", "CASP"):
            s = g[g.family == fam].sort_values("A_mean")
            a_.plot(s.A_mean, s[att], "o-", color=COL[fam], label=fam)
            for _, r in s.iterrows():
                a_.annotate(f"{r.dial:g}", (r.A_mean, r[att]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a_.set(title=att, ylim=(0.45, 1.03))
        a_.grid(alpha=0.3)
    ax.ravel()[-1].axis("off")
    ax.ravel()[0].legend()
    for a_ in ax[1]:
        a_.set_xlabel("CASP audibility of the watermark (units)")
    for a_ in ax[:, 0]:
        a_.set_ylabel("bit accuracy")
    fig.suptitle("Robustness vs audibility (labels = AWARE tolerance dB / CASP target)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_dial_robustness.png"), dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for fam in ("AWARE", "CASP"):
        s = g[g.family == fam].sort_values("A_mean")
        pts = D[D.family == fam]
        ax[0].scatter(pts.A_mean, pts.hard_mean, s=10, color=COL[fam], alpha=0.25)
        ax[0].plot(s.A_mean, s.hard_mean, "o-", color=COL[fam], label=fam)
        ax[1].scatter(pts.pesq, pts.hard_mean, s=10, color=COL[fam], alpha=0.25)
        ax[1].plot(s.pesq, s.hard_mean, "o-", color=COL[fam], label=fam)
        ax[2].scatter(pts.A_mean, pts.pesq, s=10, color=COL[fam], alpha=0.25)
        ax[2].plot(s.A_mean, s.pesq, "o-", color=COL[fam], label=fam)
    ax[0].set(xlabel="CASP audibility (units)", ylabel="hard-attack bit accuracy", title="Robustness vs audibility")
    ax[1].set(xlabel="PESQ", ylabel="hard-attack bit accuracy", title="Robustness vs PESQ")
    ax[2].set(xlabel="CASP audibility (units)", ylabel="PESQ", title="Do the two judges agree?")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_dial_tradeoff.png"), dpi=130)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
