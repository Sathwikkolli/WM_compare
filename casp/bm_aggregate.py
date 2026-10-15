"""
casp/bm_aggregate.py -- tables and figures for the SS-vs-neural benchmark.

    python bm_aggregate.py            # after bm.sbatch and bm_null.sbatch finish

Primary metric ("payload test"): the decoded payload matches the embedded one in
at least k bits, k from the binomial null at 1% FPR (10 bits: 9; 16: 13; 20: 16).
This is the only test all four systems share (Timbre has no presence score).
Secondary ("native presence"): the system's own score above the 99th percentile
of its scores on unwatermarked, identically attacked clips.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("CASP_BENCH_RUN", "2026-10-14_casp-bench")
RESULTS = os.environ.get("CASP_BENCH_DIR", os.path.join(ROOT, "results", RUN))
SYS_ORDER = ["audioseal", "aware", "timbre", "ss", "ss_casp"]
LABEL = {"audioseal": "AudioSeal", "aware": "AWARE", "timbre": "Timbre", "ss": "SS (AWARE shape)",
         "ss_casp": "SS (CASP shape)", "ss_flat": "SS (flat)"}
KEY_FAMS = ["lp_hz", "vox_encodec", "vox_gaussian_noise", "vox_mp3", "vox_opus", "requant", "notch_worst",
            "vox_time_stretch", "resample"]


def md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            (fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
             else ("" if isinstance(v, (float, np.floating)) else str(v))) for v in r) + " |")
    return "\n".join(out)


def load():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "bm_rows.csv")))
    if not files:
        raise SystemExit(f"no bm_rows.csv under {RESULTS}")
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    R["bits_ok"] = pd.to_numeric(R.bits_ok, errors="coerce")
    nfiles = sorted(glob.glob(os.path.join(RESULTS, "null", "null_chunk*.csv")))
    N = pd.concat([pd.read_csv(f, dtype={"bits": str}) for f in nfiles], ignore_index=True) if nfiles else None
    return R, N


def add_notch_worst(R):
    n = R[R.family == "notch"]
    if n.empty:
        return R
    g = n.groupby(["dataset", "clip_id", "system", "op", "target"], dropna=False)
    w = g.agg(bits_ok=("bits_ok", "min"), bit_acc=("bit_acc", "min"), pesq=("pesq", "first"),
              stoi=("stoi", "first"), snr_db=("snr_db", "first"), scale=("scale", "first")).reset_index()
    w["family"], w["label"], w["sev"] = "notch_worst", "worst", 0
    return pd.concat([R, w], ignore_index=True)


def native_threshold(N):
    if N is None:
        return None
    d = N.dropna(subset=["native_score"])
    d = d[d.system != "timbre"]
    return d.groupby(["system", "family", "label"]).native_score.quantile(0.99).rename("thr").reset_index()


def null_payload_fpr(N, nbits, kmap, n_rand=50, seed=0):
    """Empirical false-alarm rate of the payload test on clean attacked audio."""
    if N is None:
        return None
    rng = np.random.RandomState(seed)
    rows = []
    for s, g in N.groupby("system"):
        if s not in kmap or not isinstance(kmap[s], int):
            continue
        n = nbits[s]
        hit = tot = 0
        for b in g.bits.dropna():
            v = np.array([int(c) for c in str(b)][:n])
            if len(v) < n:
                continue
            pay = rng.randint(0, 2, (n_rand, n))
            hit += int((((pay == v).sum(axis=1)) >= kmap[s]).sum())
            tot += n_rand
        rows.append(dict(system=s, null_fpr=hit / max(tot, 1), n=tot))
    return pd.DataFrame(rows)


def main():
    R, N = load()
    R = add_notch_worst(R)
    R = R[R.family != "none"].copy() if False else R
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, "*", "bm_meta.json"))]
    nbits, kmap, fprs = {}, {}, {}
    for m in metas:
        for s, v in m["systems"].items():
            nbits[s], kmap[s] = v["nbits"], v["k_threshold"]
            fprs[s] = v.get("fpr", float("nan"))
    n_clips = R.groupby("dataset").clip_id.nunique().to_dict()

    # attach native threshold
    thr = native_threshold(N)
    if thr is not None:
        R = R.merge(thr, on=["system", "family", "label"], how="left")
        R["native_ok"] = (R.native_score > R.thr).astype(float).where(R.thr.notna())
    L = [f"# SS vs AudioSeal / AWARE / Timbre\n",
         f"Clips per dataset: {n_clips}. Payload test thresholds (1% binomial): "
         + ", ".join(f"{LABEL.get(s, s)} {kmap[s]}/{nbits[s]}" for s in SYS_ORDER if s in kmap) + ".\n"]

    ops = R[["op", "target"]].drop_duplicates().astype(str).values.tolist()
    syss = [s for s in SYS_ORDER if s in R.system.unique()]

    # quality at each operating point
    q = (R[R.family == "none"].groupby(["op", "target", "system"], dropna=False)
         [["pesq", "stoi", "snr_db", "scale"]].mean().reset_index())
    q["system"] = q.system.map(LABEL)
    L += ["## Quality at each operating point (mean over clips)\n", md(q, "{:.3f}"), ""]

    # TPR by family
    body = R[~R.family.isin(["none"])]
    for op, tgt in ops:
        sub = body[(body.op == op) & (body.target.astype(str) == tgt)]
        if sub.empty:
            continue
        t = (sub.groupby(["family", "system"]).bits_ok.mean().unstack("system").reindex(columns=syss))
        t.columns = [LABEL[c] for c in t.columns]
        t["_ord"] = [KEY_FAMS.index(f) if f in KEY_FAMS else 99 for f in t.index]
        t = t.sort_values(["_ord"]).drop(columns="_ord").reset_index()
        L += [f"## Payload-test TPR at about 1% FPR, operating point `{op}` {tgt}: mean over attack strengths and clips\n",
              md(t), ""]
        # strength curves for key families
        for fam in KEY_FAMS:
            f = sub[sub.family == fam]
            if f.empty:
                continue
            c = (f.groupby(["sev", "label", "system"]).bits_ok.mean().unstack("system").reindex(columns=syss))
            c.columns = [LABEL[x] for x in c.columns]
            c = c.reset_index().sort_values("sev").drop(columns="sev").rename(columns={"label": "strength (mild to strong)"})
            L += [f"### {fam}, `{op}` {tgt}\n", md(c), ""]

    # breaking points
    bp = []
    for op, tgt in ops:
        sub = body[(body.op == op) & (body.target.astype(str) == tgt)]
        for (fam, s), g in sub.groupby(["family", "system"]):
            c = g.groupby(["sev", "label"]).bits_ok.mean().reset_index().sort_values("sev")
            bad = c[c.bits_ok < 0.5]
            bp.append(dict(op=f"{op} {tgt}", family=fam, system=LABEL[s],
                           breaks_at=bad.label.iloc[0] if len(bad) else "survives all"))
    BP = pd.DataFrame(bp).pivot_table(index=["op", "family"], columns="system", values="breaks_at",
                                      aggfunc="first").reset_index()
    L += ["## Breaking point: first strength where TPR falls below 0.5\n", md(BP), ""]

    # paired differences SS vs each neural (per clip mean over all attacks)
    pr = []
    for op, tgt in ops:
        sub = body[(body.op == op) & (body.target.astype(str) == tgt)]
        pc = sub.groupby(["dataset", "clip_id", "system"]).bits_ok.mean().unstack("system")
        for base in ("audioseal", "aware", "timbre"):
            if "ss" in pc and base in pc:
                d = (pc["ss"] - pc[base]).dropna()
                if len(d) >= 3:
                    rng = np.random.RandomState(0)
                    bs = [rng.choice(d.values, len(d)).mean() for _ in range(1000)]
                    pr.append(dict(op=f"{op} {tgt}", versus=LABEL[base], mean_diff=d.mean(),
                                   ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                   clips_ss_better=int((d > 0).sum()), n=len(d)))
    if pr:
        L += ["## SS minus each neural system, mean TPR over all attacks (bootstrap 95% over clips)\n",
              md(pd.DataFrame(pr), "{:+.3f}"), ""]

    fp = null_payload_fpr(N, nbits, kmap)
    if fp is not None and len(fp):
        fp["system"] = fp.system.map(lambda s: LABEL.get(s, s))
        L += ["## False alarms: payload test on unwatermarked, attacked clips (random payloads)\n",
              md(fp, "{:.4f}"), ""]
    if thr is not None and "native_ok" in R:
        nt = (body[(body.op == "match")].groupby(["family", "system"]).native_ok.mean()
              .unstack("system"))
        nt = nt[[c for c in syss if c in nt.columns and c != "timbre"]]
        nt.columns = [LABEL[c] for c in nt.columns]
        L += ["## Native presence score, TPR at 1% FPR from the null (matched point; Timbre has none)\n",
              md(nt.reset_index()), ""]

    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for op, tgt in ops:
        sub = body[(body.op == op) & (body.target.astype(str) == tgt)]
        if sub.empty:
            continue
        h = sub.groupby(["family", "system"]).bits_ok.mean().unstack("system").reindex(columns=syss)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(h))))
        im = ax.imshow(h.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(h.columns)), [LABEL[c] for c in h.columns], rotation=20)
        ax.set_yticks(range(len(h.index)), h.index)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                v = h.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, label="TPR at 1% FPR (payload test)")
        ax.set_title(f"operating point {op} {tgt}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"fig_heatmap_{op}_{tgt or 'ship'}.png"), dpi=130)
        plt.close(fig)

    sub = body[(body.op == "match")]
    fams = [f for f in KEY_FAMS if f in sub.family.unique()]
    if fams:
        fig, ax = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
        ax = np.atleast_1d(ax)
        for a_, fam in zip(ax, fams):
            f = sub[sub.family == fam]
            for s in syss:
                c = f[f.system == s].groupby(["sev", "label"]).bits_ok.mean().reset_index().sort_values("sev")
                if len(c):
                    a_.plot(c.sev, c.bits_ok, "o-", ms=3, label=LABEL[s])
            a_.set(title=fam, xlabel="strength (mild to strong)", ylim=(-0.03, 1.03))
            a_.grid(alpha=0.3)
        ax[0].set_ylabel("TPR (payload test)")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "fig_curves_match.png"), dpi=130)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
