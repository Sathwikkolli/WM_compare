"""
masking/aggregate_phase1.py -- after the Phase 1 array (login node): tables,
paired tests, Gate 2 verdict and figures -> results/<run>/summary.md

Gate 2 (pre-registered in the run README):
  A  mask shape is less audible at equal energy:  median PESQ(mask) - PESQ(aware) > 0,
     Wilcoxon p < 0.05                                                   (E3)
  B  mask shape carries more energy at equal PESQ: median gain >= +1 dB  (E4)
  C  codecs do not strip it more: median corr(mask_eq_quality) / corr(aware)
     >= 0.9 at mp3_64 AND opus_32                                         (E6)
  PASS = (A or B) and C
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from run_phase1 import RUN
import common as C

RES = os.path.join(C.ROOT, "results", RUN)
DATA, FIGS = os.path.join(RES, "data"), os.path.join(RES, "figures")


def load(pat):
    fs = sorted(glob.glob(os.path.join(DATA, pat)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def wilcoxon_p(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.all(d == 0):
        return float("nan")
    return float(wilcoxon(d).pvalue)


def main():
    os.makedirs(FIGS, exist_ok=True)
    e2 = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "e2_asv*.json")))])
    e3, e4, e5, e6 = load("e3_*.csv"), load("e4_*.csv"), load("e5_*.csv"), load("e6_*.csv")
    if e3.empty:
        raise SystemExit(f"no data in {DATA}")
    L = [f"# Phase 1 (E2–E6) summary — {RUN}", "",
         f"Clips: **{e3.clip_id.nunique()}**. Shapes: `aware` = |X|·10^(−6/20); "
         "`mask` = Johnston-style masking threshold (`masking/psy.py`); `flat` = constant (control). "
         "Random ±1 signs on |X| in 1–4 kHz, phase kept (AWARE's watermark is ~signs × budget, E1a).", ""]

    # ---- E2 ----
    L += ["## E2 — where the two budgets differ", "", "| quantity | median over clips | min | max |", "|---|---|---|---|"]
    for k in ("ratio_db_median", "ratio_db_p10", "ratio_db_p90", "frac_mask_gt_aware", "energy_ratio_db",
              "ratio_db_quiet_frames", "ratio_db_loud_frames", "frac_bins_tonal", "ratio_db_tonal",
              "ratio_db_noisy", "aware_above_mask", "aware_above_mask_tonal", "aware_above_mask_noisy"):
        v = pd.to_numeric(e2[k], errors="coerce")
        L.append(f"| {k} | {v.median():.3f} | {v.min():.3f} | {v.max():.3f} |")
    L += ["", "ratio = mask budget / AWARE budget per bin (dB); `aware_above_mask` = share of bins where "
              "AWARE's change exceeds the modelled threshold.", ""]

    # ---- E3 ----
    p3 = e3.pivot(index="clip_id", columns="shape", values="pesq")
    s3 = e3.pivot(index="clip_id", columns="shape", values="stoi")
    d3 = p3["mask"] - p3["aware"]
    df3 = p3["flat"] - p3["aware"]
    pA = wilcoxon_p(d3)
    A = bool(d3.median() > 0 and pA < 0.05)
    L += ["## E3 — equal energy: which shape sounds better?", "",
          "| shape | PESQ mean | PESQ median | STOI mean | Δ PESQ vs aware (median) | clips better | Wilcoxon p |",
          "|---|---|---|---|---|---|---|"]
    for sh, d in (("aware", None), ("mask", d3), ("flat", df3)):
        extra = ("| — | — | — |" if d is None else
                 f"| {d.median():+.3f} | {(d > 0).sum()}/{len(d)} | {wilcoxon_p(d):.3g} |")
        L.append(f"| {sh} | {p3[sh].mean():.3f} | {p3[sh].median():.3f} | {s3[sh].mean():.3f} " + extra)
    L += ["", f"**Gate 2A: {'PASS' if A else 'fail'}** (median Δ {d3.median():+.3f}, p = {pA:.3g}).", ""]

    # ---- E4 ----
    g = e4.energy_gain_db
    B = bool(g.median() >= 1.0)
    L += ["## E4 — equal quality: extra energy the mask shape carries at AWARE's PESQ", "",
          f"| median | mean | IQR | min | max | clips > 0 dB |", "|---|---|---|---|---|---|",
          f"| {g.median():+.2f} dB | {g.mean():+.2f} dB | {g.quantile(.25):+.2f} … {g.quantile(.75):+.2f} | "
          f"{g.min():+.2f} | {g.max():+.2f} | {(g > 0).sum()}/{len(g)} |", "",
          f"Mask gain over its own threshold at that point: median {e4.mask_gain_db.median():+.2f} dB "
          "(0 dB = exactly at the modelled threshold).", "",
          f"**Gate 2B: {'PASS' if B else 'fail'}** (median {g.median():+.2f} dB; rule ≥ +1 dB).", ""]

    # ---- E5 ----
    if not e5.empty:
        L += ["## E5 — tonal vs noisy frames (equal energy within each frame type)", "",
              "| frames | aware PESQ | mask PESQ | Δ median | clips better | p |", "|---|---|---|---|---|---|"]
        for kind, d in e5.groupby("frames"):
            pv = d.pivot(index="clip_id", columns="shape", values="pesq")
            dd = pv["mask"] - pv["aware"]
            L.append(f"| {kind} | {pv['aware'].mean():.3f} | {pv['mask'].mean():.3f} | {dd.median():+.3f} | "
                     f"{(dd > 0).sum()}/{len(dd)} | {wilcoxon_p(dd):.3g} |")
        L.append("")

    # ---- E6 ----
    L += ["## E6 — codec survival of the change", "",
          "| codec | shape | corr | kept (dB) | sign agree | sign agree (energy-wtd) |", "|---|---|---|---|---|---|"]
    for (codec, shape), d in e6.groupby(["codec", "shape"], sort=False):
        L.append(f"| {codec} | {shape} | {d['corr'].median():.3f} | {d.kept_db.median():+.2f} | "
                 f"{d.sign_agree.median():.3f} | {d.sign_agree_wtd.median():.3f} |")
    ratios = {}
    for codec in ("mp3_64", "opus_32"):
        pv = e6[e6.codec == codec].pivot(index="clip_id", columns="shape", values="corr")
        ratios[codec] = float((pv["mask_eq_quality"] / pv["aware"]).median())
    Cg = all(v >= 0.9 for v in ratios.values())
    L += ["", "corr = correlation between the change before and after the codec (1 = fully kept). "
              "kept = energy of the change after / before.", "",
          f"**Gate 2C: {'PASS' if Cg else 'fail'}** (median corr ratio mask_eq_quality/aware: " +
          ", ".join(f"{k} {v:.3f}" for k, v in ratios.items()) + "; rule ≥ 0.9 at both).", ""]

    gate = (A or B) and Cg
    L.insert(2, f"> **Gate 2: {'PASS' if gate else 'FAIL'}** — A (E3) {'✓' if A else '✗'}, "
                f"B (E4) {'✓' if B else '✗'}, C (E6) {'✓' if Cg else '✗'}. Rule: (A or B) and C.\n")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        maps = sorted(glob.glob(os.path.join(DATA, "e2_maps_*.npz")))
        if maps:
            z = np.load(maps[0])
            fig, ax = plt.subplots(1, 4, figsize=(20, 4))
            to_db = lambda v: 20 * np.log10(np.maximum(v, 1e-6))
            ext = [0, z["M"].shape[1] * 256 / 16000, z["freqs"][0], z["freqs"][-1]]
            for i, (k, t) in enumerate((("M", "|X| (dB)"), ("A", "AWARE budget (dB)"), ("B", "mask budget (dB)"))):
                im = ax[i].imshow(to_db(z[k]), origin="lower", aspect="auto", extent=ext, vmin=-60, vmax=40)
                ax[i].set(title=t, xlabel="s", ylabel="Hz"); fig.colorbar(im, ax=ax[i])
            im = ax[3].imshow(to_db(z["B"]) - to_db(z["A"]), origin="lower", aspect="auto", extent=ext,
                              cmap="RdBu", vmin=-20, vmax=20)
            ax[3].set(title="mask − AWARE (dB): blue = mask allows more", xlabel="s"); fig.colorbar(im, ax=ax[3])
            fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e2_budget_maps.png"), dpi=120); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].scatter(p3["aware"], p3["mask"], label="mask"); ax[0].scatter(p3["aware"], p3["flat"], label="flat")
        lim = [p3.min().min() - .05, 4.65]; ax[0].plot(lim, lim, "k--", lw=.8)
        ax[0].set(title="E3: PESQ at equal energy", xlabel="aware shape", ylabel="other shape"); ax[0].legend()
        ax[1].hist(g, bins=15); ax[1].axvline(0, color="k", lw=.8)
        ax[1].set(title="E4: extra energy at AWARE's PESQ", xlabel="dB")
        piv = e6.groupby(["codec", "shape"])["corr"].median().unstack()
        piv.plot.bar(ax=ax[2]); ax[2].set(title="E6: change kept after codec (corr)", ylim=(0, 1))
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e3_e4_e6.png"), dpi=120); plt.close(fig)
        L += ["## Figures", "", "- `figures/e2_budget_maps.png`", "- `figures/e3_e4_e6.png`", ""]
    except Exception as e:
        L += [f"(figures failed: {e})", ""]

    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
