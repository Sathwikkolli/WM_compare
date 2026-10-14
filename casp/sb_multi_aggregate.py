"""
casp/sb_multi_aggregate.py -- AWARE vs CASP multibit: tables and figures.

    python sb_multi_aggregate.py        # after the sb_multi sbatch array finishes

All systems are matched to AWARE 6 dB's quality on each clip. Scores are not in
the same units: AWARE = detector confidence (found at >= 0.5); CASP/AWSHAPE =
z-equivalent of a chi-square presence test (found at p <= 1e-3, z_eq ~ 3.3).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("CASP_SBM_RUN", "2026-10-14_casp-sbm")
RESULTS = os.environ.get("CASP_SBM_DIR", os.path.join(ROOT, "results", RUN))
SYS = ["AWARE6", "CASP_K1", "CASP_K4", "CASP_K8", "CASP_K20", "AWSHAPE_K20"]
LABEL = {"AWARE6": "AWARE 6 dB (20 bits)", "CASP_K1": "CASP 1 bit", "CASP_K4": "CASP 4 bits",
         "CASP_K8": "CASP 8 bits", "CASP_K20": "CASP 20 bits", "AWSHAPE_K20": "AWARE-shape 20 bits (control)"}
ATT = ["clean", "awgn_20", "awgn_10", "awgn_5", "lp_3500", "lp_2500", "lp_2000", "mp3_32", "mp3_16", "opus_12", "requant_6"]
COL = {"AWARE6": "tab:red", "CASP_K1": "#9ecae1", "CASP_K4": "#6baed6", "CASP_K8": "#3182bd",
       "CASP_K20": "#08519c", "AWSHAPE_K20": "tab:gray"}


def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                                     else ("" if isinstance(v, (float, np.floating)) else str(v))
                                     for v in r) + " |")
    return "\n".join(out)


def table(wm, value, agg="mean", fmt="{:.2f}"):
    t = wm.pivot_table(index="system", columns="attack", values=value, aggfunc=agg).reindex(SYS)
    t = t[[a for a in ATT if a in t.columns]]
    t["mean"] = t.mean(axis=1)
    t.insert(0, "system", [LABEL[s] for s in t.index])
    return t.reset_index(drop=True)


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "sbm_rows.csv")))
    if not files:
        raise SystemExit(f"no sbm_rows.csv under {RESULTS}")
    R = pd.concat([pd.read_csv(f) for f in files])
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, "*", "sbm_meta.json")))]
    clips = sorted(R.clip_id.unique())
    wm, neg = R[R.kind == "wm"].copy(), R[R.kind == "neg"]
    wm["detected"] = wm.detected.astype(float)
    wm["score_c"] = wm.score.clip(upper=40)
    qa = metas[0].get("quality_axis", "pesq")

    L = [f"# AWARE vs CASP multibit, equal quality: {len(clips)} clips, {wm.attack.nunique()} attack conditions\n",
         f"Quality axis for matching: **{qa}** (every system matched to AWARE 6 dB's value on each clip). "
         "Detected = AWARE confidence >= 0.5, CASP/control chi-square p <= 1e-3.\n"]

    L += ["## 1. Detection rate (share of clips where the watermark is found)\n", md(table(wm, "detected")), ""]
    L += ["## 2. Bit accuracy (mean over clips; chance = 0.50)\n", md(table(wm, "bit_acc")), ""]
    L += ["## 3. Detection score (mean over clips; AWARE = confidence 0-1, others = z-equivalent, capped at 40)\n",
          md(table(wm, "score_c", fmt="{:.1f}"), "{:.1f}"), ""]

    # head-to-head, CASP 20 bits vs AWARE
    rows = []
    for att in ATT:
        a = wm[(wm.attack == att) & (wm.system == "AWARE6")].set_index("clip_id")
        for other in ("CASP_K20", "AWSHAPE_K20"):
            c = wm[(wm.attack == att) & (wm.system == other)].set_index("clip_id")
            j = a.join(c, lsuffix="_a", rsuffix="_c", how="inner")
            d_bit = (j.bit_acc_c - j.bit_acc_a).round(6)
            d_det = j.detected_c - j.detected_a
            rows.append(dict(attack=att, versus=LABEL[other],
                             bit_acc_diff=float(d_bit.mean()), bit_W_T_L=f"{int((d_bit > 0).sum())}/{int((d_bit == 0).sum())}/{int((d_bit < 0).sum())}",
                             det_diff=float(d_det.mean()), det_W_T_L=f"{int((d_det > 0).sum())}/{int((d_det == 0).sum())}/{int((d_det < 0).sum())}"))
    H = pd.DataFrame(rows)
    L += ["## 4. 20-bit head-to-head, per clip (difference = system minus AWARE; W/T/L = clips where it is better/equal/worse)\n",
          md(H, "{:+.3f}"), ""]

    # false positives
    fp = neg.assign(detected=neg.detected.astype(float)).pivot_table(
        index="system", columns="attack", values="detected", aggfunc="sum").reindex(
        ["AWARE6", "CASP_K1", "CASP_K4", "CASP_K8", "CASP_K20"])[[a for a in ATT if a in neg.attack.unique()]]
    fp.insert(0, "system", [LABEL[s] for s in fp.index])
    L += [f"## 5. False positives on unwatermarked audio (count out of {len(clips)})\n", md(fp.reset_index(drop=True), "{:.0f}"), ""]

    # matching diagnostics
    q = []
    for m in metas:
        for nm, s in m["systems"].items():
            q.append(dict(clip=m["clip_id"], system=nm, g=s["g"], status=s["status"], **s["quality"]))
    Q = pd.DataFrame(q)
    qs = Q.groupby("system").agg(pesq=("pesq", "mean"), stoi=("stoi", "mean"), snr_db=("snr_db", "mean"),
                                 g=("g", "mean"), clamped_high=("status", lambda s: int((s == "clamped_high").sum())),
                                 clamped_low=("status", lambda s: int((s == "clamped_low").sum()))).reindex(SYS).reset_index()
    qs["system"] = qs.system.map(LABEL)
    L += ["## 6. Matching diagnostics (g = strength, 1 = AWARE-6 dB energy; clamped = target quality not reachable inside g in [0.01, 2])\n",
          md(qs, "{:.3f}"), ""]

    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L))
    H.to_csv(os.path.join(RESULTS, "head_to_head.csv"), index=False)
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["AWARE6", "CASP_K1", "CASP_K20", "AWSHAPE_K20"]
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x = np.arange(len(ATT))
    w = 0.2
    for k, s in enumerate(show):
        for j, (val, ttl) in enumerate((("detected", "Detection rate"), ("bit_acc", "Bit accuracy"))):
            m = [wm[(wm.system == s) & (wm.attack == a)][val].mean() for a in ATT]
            ax[j].bar(x + (k - 1.5) * w, m, w, color=COL[s], label=LABEL[s])
            ax[j].set_ylabel(ttl)
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_ylim(0.4, 1.02)
    ax[0].set_ylim(0, 1.05)
    ax[1].set_xticks(x, ATT, rotation=20)
    ax[0].legend(ncol=4, fontsize=8)
    fig.suptitle(f"AWARE 6 dB vs CASP multibit at equal quality ({len(clips)} clips)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_sbm_attacks.png"), dpi=125)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for s in ["CASP_K1", "CASP_K4", "CASP_K8", "CASP_K20"]:
        K = int(s.split("K")[1])
        ax[0].plot([K], [wm[(wm.system == s)].detected.mean()], "o", color=COL[s], ms=9)
        ax[1].plot([K], [wm[(wm.system == s)].bit_acc.mean()], "o", color=COL[s], ms=9)
    for a_, val, ttl in ((ax[0], "detected", "mean detection rate over all attacks"),
                         (ax[1], "bit_acc", "mean bit accuracy over all attacks")):
        a_.axhline(wm[wm.system == "AWARE6"][val].mean(), color="tab:red", ls="--", label="AWARE 6 dB (20 bits)")
        a_.set(xlabel="payload bits (CASP scheme)", ylabel=ttl, xscale="log", xticks=[1, 4, 8, 20])
        a_.set_xticklabels(["1", "4", "8", "20"])
        a_.grid(alpha=0.3)
        a_.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_sbm_payload.png"), dpi=125)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
