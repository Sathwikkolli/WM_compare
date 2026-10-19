"""
mpeg_wm/bench_aggregate.py -- collect results/<run>/<clip>/rows.csv into tables + figures.

    python bench_aggregate.py            # after the sbatch array finishes

Writes into results/<run>/:
    all_rows.csv.gz     every row from every clip
    by_config.csv       mean bit accuracy / detection rate per (system, op, attack config)
    by_family.csv       the same, pooled over each attack family's strengths
    summary.md          quality, clean decoding, false alarms, per-family table
    clips.json          the clip list used (copied from clips120.json)
    fig_heatmap_<op>.png, fig_family_bars_<op>.png
"""
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.environ.get("MPEG_BENCH_RUN", "2026-10-19_mpeg-vox-bench")
RESULTS = os.path.join(ROOT, "results", RUN)
ORDER = ["MPEG", "AudioSeal", "AWARE", "Timbre"]
NAMES = {"mpeg": "MPEG", "audioseal": "AudioSeal", "aware": "AWARE", "timbre": "Timbre"}


def load():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "rows.csv")))
    if not files:
        raise SystemExit(f"no rows.csv under {RESULTS}")
    # keep_default_na=False: pandas would otherwise read op "null" as a missing value.
    df = pd.concat([pd.read_csv(f, dtype={"bits": str, "payload": str, "target": str},
                                keep_default_na=False, na_values=["", "nan", "NaN"]) for f in files],
                   ignore_index=True)
    if df["op"].isna().any():
        raise SystemExit(f"{int(df['op'].isna().sum())} rows have no op; check the CSV parsing")
    df["target"] = df["target"].fillna("")
    df["system"] = df["system"].map(NAMES).fillna(df["system"])
    df["op_t"] = np.where(df["target"] != "", df["op"] + "@" + df["target"], df["op"])
    df["config"] = df["family"].str.replace("vox_", "", regex=False) + " " + df["label"].astype(str)
    return df, len(files)


def table(df, keys):
    g = df.groupby(keys, sort=False)
    return g.agg(n=("bit_acc", "size"), bit_acc=("bit_acc", "mean"),
                 bit_acc_sd=("bit_acc", "std"), detect=("bits_ok", "mean")).reset_index()


def md(df, floatfmt=".3f"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def heatmap(bc, op, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = bc[bc["op_t"] == op]
    if d.empty:
        return
    piv = d.pivot_table(index="config", columns="system", values="bit_acc", sort=False)
    piv = piv[[c for c in ORDER if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * piv.shape[1], 0.6 + 0.24 * piv.shape[0]))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"Mean bit accuracy, op = {op}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def family_bars(bf, op, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = bf[bf["op_t"] == op]
    if d.empty:
        return
    piv = d.pivot_table(index="family", columns="system", values="detect", sort=False)
    piv = piv[[c for c in ORDER if c in piv.columns]]
    piv.index = [i.replace("vox_", "") for i in piv.index]
    ax = piv.plot.bar(figsize=(12, 4), width=0.8)
    ax.set_ylabel("payload detected (binomial ~1% FPR test)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Detection rate per attack family, op = {op}")
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    df, n_clips = load()
    df.to_csv(os.path.join(RESULTS, "all_rows.csv.gz"), index=False)
    if os.path.exists(os.path.join(HERE, "clips120.json")):
        shutil.copy(os.path.join(HERE, "clips120.json"), os.path.join(RESULTS, "clips.json"))

    att = df[df["family"] != "none"]
    bc = table(att, ["system", "op_t", "family", "label", "sev", "config"])
    bf = table(att, ["system", "op_t", "family"])
    bc.to_csv(os.path.join(RESULTS, "by_config.csv"), index=False)
    bf.to_csv(os.path.join(RESULTS, "by_family.csv"), index=False)

    clean = df[df["family"] == "none"]
    marked = clean[clean["op"] != "null"]
    q = (marked.groupby(["system", "op_t"], sort=False)
         .agg(clips=("pesq", "size"), pesq=("pesq", "mean"), pesq_sd=("pesq", "std"),
              stoi=("stoi", "mean"), snr_db=("snr_db", "mean"), scale=("scale", "median"),
              clamped=("status", lambda s: int((s != "ok").sum())),
              clean_bit_acc=("bit_acc", "mean"), clean_detect=("bits_ok", "mean"))
         .reset_index())

    null = df[df["op"] == "null"]
    fa = (null.assign(attacked=null["family"] != "none")
          .groupby(["system", "attacked"]).agg(n=("bits_ok", "size"), false_alarm=("bits_ok", "mean"),
                                               bit_acc=("bit_acc", "mean")).reset_index())
    k = (df.drop_duplicates("system")[["system", "n_bits", "k_ok"]]
         .assign(rule=lambda d: d["k_ok"].astype(str) + "/" + d["n_bits"].astype(str)))

    ops = [o for o in ["ship"] + sorted(o for o in df["op_t"].unique() if o.startswith("match"))
           if o in set(df["op_t"])]
    lines = [f"# MPEG-masking watermark vs AudioSeal / AWARE / Timbre on VoxWatermark attacks",
             "", f"Run `{RUN}`: **{n_clips}** bona fide ASVspoof2019 LA eval clips (5 s crops), "
             f"{att['config'].nunique()} attack configs from {att['family'].nunique()} VoxWatermark families.",
             "", "Detection = payload test: at least k of n bits correct, k chosen so a random "
             "payload passes with ~1% probability.", "", md(k[["system", "rule"]]), "",
             "## Quality and clean decoding", "", md(q), "",
             "## False alarms (unwatermarked clips, decoded against the clip's payload)", "", md(fa), ""]
    for op in ops:
        piv_acc = bf[bf["op_t"] == op].pivot_table(index="family", columns="system", values="bit_acc", sort=False)
        piv_det = bf[bf["op_t"] == op].pivot_table(index="family", columns="system", values="detect", sort=False)
        cols = [c for c in ORDER if c in piv_acc.columns]
        both = piv_acc[cols].copy()
        for c in cols:
            both[c] = [f"{a:.3f} / {d:.2f}" for a, d in zip(piv_acc[c], piv_det[c])]
        both.index = [i.replace("vox_", "") for i in both.index]
        lines += [f"## Per attack family, op = {op} (mean bit acc / detection rate, all strengths pooled)",
                  "", md(both.reset_index().rename(columns={"index": "family"})), ""]
        heatmap(bc, op, os.path.join(RESULTS, f"fig_heatmap_{op.replace('@', '_')}.png"))
        family_bars(bf, op, os.path.join(RESULTS, f"fig_family_bars_{op.replace('@', '_')}.png"))
    lines += ["Full per-strength numbers: `by_config.csv`. Figures: `fig_heatmap_*.png`, "
              "`fig_family_bars_*.png`.", ""]
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
