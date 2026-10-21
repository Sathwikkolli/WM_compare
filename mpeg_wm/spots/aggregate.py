"""
spots/aggregate.py -- turn every clip's cells.csv into the answer:
WHERE does CASP see room MPEG misses (headroom > 0), and where does it say MPEG allows
too much (headroom < 0)?

    python aggregate.py                       # after the sbatch array finishes

Statistics: per clip, the median headroom of the cells in a group; then, across clips,
the median of those clip medians with a 95% bootstrap interval (clips resampled), so
the clip is the unit and one odd clip cannot dominate.

Writes results/<run>/:
    summary.md                 tables + how to read them
    contexts_full.csv          full-band headroom per speech context
    contexts_band.csv          per (context, band) and per (band context, band)
    fig_context_band.png       heat map: context x band, median headroom (dB)
    listening_candidates.csv   cells for the listening test (Phase 3)
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.environ.get("SPOTS_RUN", "2026-10-22_casp-spots")
RESULTS = os.path.join(ROOT, "results", RUN)
CONTEXT_ORDER = ["long_pause", "short_gap", "tail", "onset", "voiced", "fricative", "decay", "other"]
HIGH_CONF_MAX_HZ = 6263      # the top band (6.3-8 kHz) sits at CASP's 16 kHz edge: low confidence


def load():
    files = sorted(glob.glob(os.path.join(RESULTS, "*", "cells.csv")))
    if not files:
        raise SystemExit(f"no cells.csv under {RESULTS}")
    df = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=["", "nan"]) for f in files],
                   ignore_index=True)
    return df[np.isfinite(df["headroom_db"])], len(files)


def clip_bootstrap(df, keys, n_boot=4000, seed=0):
    """Median of per-clip medians per group, with a 95% bootstrap interval over clips."""
    per_clip = df.groupby(keys + ["clip_id"])["headroom_db"].median().reset_index()
    n_cells = df.groupby(keys).size()
    rng = np.random.default_rng(seed)
    rows = []
    for g, d in per_clip.groupby(keys, sort=False):
        g = g if isinstance(g, tuple) else (g,)
        v = d["headroom_db"].to_numpy()
        boots = np.median(rng.choice(v, size=(n_boot, len(v))), axis=1) if len(v) > 1 else v
        rows.append(dict(zip(keys, g), clips=len(v), cells=int(n_cells.loc[g if len(g) > 1 else g[0]]),
                         headroom_db=float(np.median(v)),
                         ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                         share_room=float(np.mean(v > 0))))
    return pd.DataFrame(rows)


def verdict(r):
    if r["ci_lo"] > 1.0:
        return "CASP sees room MPEG misses"
    if r["ci_hi"] < -1.0:
        return "MPEG allows too much (per CASP)"
    return "agree (within +-1 dB)"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def heatmap(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = t.pivot_table(index="context", columns="band_hz", values="headroom_db", sort=False)
    piv = piv.reindex([c for c in CONTEXT_ORDER if c in piv.index])
    piv = piv[sorted(piv.columns, key=lambda s: float(s.split("-")[0]))]
    fig, ax = plt.subplots(figsize=(13, 0.6 + 0.55 * len(piv)))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("CASP headroom vs deployed MPEG (dB): green = room MPEG misses, red = MPEG allows too much")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df, n_clips = load()
    full = df[df["kind"] == "full"]
    band = df[df["kind"] == "band"].copy()
    band["band_hz"] = band["band_lo_hz"].astype(int).astype(str) + "-" + band["band_hi_hz"].astype(int).astype(str)

    tf = clip_bootstrap(full, ["context"])
    tf["verdict"] = tf.apply(verdict, axis=1)
    tf = tf.set_index("context").reindex([c for c in CONTEXT_ORDER if c in set(tf["context"])]).reset_index()
    tb = clip_bootstrap(band, ["context", "band_hz"])
    tbc = clip_bootstrap(band, ["band_context", "band_hz"])
    tf.to_csv(os.path.join(RESULTS, "contexts_full.csv"), index=False)
    pd.concat([tb.assign(group="context"), tbc.rename(columns={"band_context": "context"}).assign(group="band_context")]) \
        .to_csv(os.path.join(RESULTS, "contexts_band.csv"), index=False)
    heatmap(tb, os.path.join(RESULTS, "fig_context_band.png"))

    # strongest disagreements, high-confidence bands only, for the listening test
    hb = band[band["band_hi_hz"] <= HIGH_CONF_MAX_HZ + 1]
    cand = pd.concat([full.assign(band_hz="250-8000"), hb])
    cand = cand[cand["clipped"] == 0]
    picks = []
    for (ctx, sign), d in cand.assign(sign=np.sign(cand["headroom_db"])).groupby(["context", "sign"]):
        d = d.reindex(d["headroom_db"].abs().sort_values(ascending=False).index)
        picks.append(d.drop_duplicates("clip_id").head(4))
    pd.concat(picks)[["clip_id", "kind", "block", "t_s", "band_lo_hz", "band_hi_hz", "context", "band_context",
                      "headroom_db"]].to_csv(os.path.join(RESULTS, "listening_candidates.csv"), index=False)

    tb_sig = tb.assign(verdict=tb.apply(verdict, axis=1))
    tb_sig = tb_sig[tb_sig["verdict"] != "agree (within +-1 dB)"].sort_values("headroom_db", ascending=False)
    lines = ["# Where CASP and the deployed MPEG watermark disagree", "",
             f"Run `{RUN}`: **{n_clips}** clips. Each cell: one 32 ms block of the deployed MPEG watermark "
             "(mask + time envelope), probed at -6/0/+6 dB with 3 PN seeds; CASP d' interpolated to the clip's "
             "reference (median d' of full-band probes at MPEG's level).", "",
             "**Headroom (dB):** how much louder (+) or quieter (-) than MPEG's level the probe can be before CASP "
             "rates it as noticeable as a typical MPEG-level probe. Median of per-clip medians, 95% bootstrap "
             "interval over clips. Verdict needs the interval to clear +-1 dB.", "",
             "## Full-band probes, by speech context", "", md(tf.round(2)), "",
             "## Per band: groups where CASP and MPEG clearly disagree", "",
             "6.3-8 kHz is at CASP's 16 kHz edge: treat it as low confidence.", "",
             md(tb_sig.round(2)) if not tb_sig.empty else "(none)", "",
             "Heat map: `fig_context_band.png`. Full tables: `contexts_full.csv`, `contexts_band.csv`. "
             "Stimuli for the listening test: `listening_candidates.csv`.", ""]
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
