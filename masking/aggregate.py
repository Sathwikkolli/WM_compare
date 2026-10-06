"""
masking/aggregate.py -- after the array finishes (login node): combine the
per-clip files in results/<run>/data/ into tables, figures and summary.md.

Gate 1 (pre-registered in the run README):
  * budget is BINDING  if >= 25% of coefficients use > 95% of their budget
  * budget is SLACK    if <  10% do
  * in between: partly binding -- look at where the saturation sits
  * errors CLUSTER     if window bit accuracy differs by >= 0.10 between the
                       quietest and loudest window quintile (clean or attacked)

Usage:
    python aggregate.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import common as C


def load(pattern):
    files = sorted(glob.glob(os.path.join(C.DATA, pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def fmt(m, s):
    return "—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"


def write_params(qual):
    if not os.path.exists(C.CLIPS_JSON):        # off-cluster re-aggregation: keep the cluster's params.json
        return
    import subprocess
    from importlib.metadata import version, PackageNotFoundError

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None
    commit = subprocess.run(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    params = dict(run=C.RUN, git_commit=commit, aware_model=C.AWARE_MODEL, n_bits=C.N_BITS,
                  num_iterations=int(qual.num_iterations.iloc[0]),
                  tolerance_db=float(qual.tolerance_db.iloc[0]),
                  payload_seed_rule="RandomState(seed*1000 + clip_index), seed=0",
                  attacks=list(C.ATTACKS), e1b_window_samples=16128, e1b_hop_samples=4000,
                  clips=json.load(open(C.CLIPS_JSON)),
                  versions={p: v(p) for p in ("aware", "torch", "numpy", "librosa", "pesq", "pystoi", "scipy")})
    json.dump(params, open(os.path.join(C.RESULTS, "params.json"), "w"), indent=2)


def true_initial(z, tol_db):
    """Original magnitudes, rebuilt from the bounds: upper = M * (1 + 10^(-tol/20)).
    The npz `initial` field cannot be trusted for runs captured before the
    in-place fix in common.OptimizeCapture (it holds the last iterate)."""
    return z["upper"] / (1 + 10 ** (-tol_db / 20))


def e1a_from_npz(tol_db):
    """E1a stats recomputed from data/e1a_<clip>.npz (the per-clip e1a_<clip>.json
    files of the 2026-10-06 run are wrong -- see the run README). Writes
    e1a_stats.csv beside summary.md."""
    from run_e0_e1 import e1a_stats
    rows = []
    for f in sorted(glob.glob(os.path.join(C.DATA, "e1a_*.npz"))):
        z = np.load(f)
        cap = dict(initial=true_initial(z, tol_db), final=z["final"], upper=z["upper"],
                   freq_indices=z["freq_indices"])
        st, _ = e1a_stats(cap, tol_db)
        st["clip_id"] = os.path.basename(f)[4:-4]
        rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, "e1a_stats.csv"), index=False)
    return df


def main():
    os.makedirs(C.FIGS, exist_ok=True)
    qual = load("e0_quality_*.csv")
    att = load("e0_attacks_*.csv")
    win = load("e1b_*.csv")
    e1a = e1a_from_npz(float(qual.tolerance_db.iloc[0]))
    if qual.empty:
        raise SystemExit(f"no data in {C.DATA}")
    n = len(qual)
    L = [f"# E0 + E1 summary — {C.RUN}", "",
         f"Clips: **{n}** bona fide ASVspoof2019 LA eval utterances (one per speaker). "
         f"AWARE `{C.AWARE_MODEL}`, {int(qual.num_iterations.iloc[0])} iterations, "
         f"tolerance {qual.tolerance_db.iloc[0]} dB, {C.N_BITS}-bit payload.", ""]

    # ---- E0 quality ----
    L += ["## E0 — quality of the watermarked audio", "",
          "| metric | mean ± std | median | min |", "|---|---|---|---|"]
    for k in ("pesq", "stoi", "snr_db", "si_snr_db", "odg", "embed_seconds"):
        v = pd.to_numeric(qual[k], errors="coerce").dropna()
        if v.empty:
            L.append(f"| {k} | not measured | | |")
        else:
            L.append(f"| {k} | {v.mean():.3f} ± {v.std():.3f} | {v.median():.3f} | {v.min():.3f} |")
    L.append("")

    # ---- E0 robustness ----
    order = list(C.ATTACKS)
    g = att.groupby("attack")
    L += ["## E0 — robustness", "",
          "| attack | bit acc (mean ± std) | BER | clips perfect | clips ≥ 0.8 | conf mean |",
          "|---|---|---|---|---|---|"]
    rob = []
    for a in order:
        if a not in g.groups:
            continue
        d = g.get_group(a)
        r = dict(attack=a, bit_acc=d.bit_acc.mean(), bit_acc_std=d.bit_acc.std(),
                 ber=d.ber.mean(), perfect=(d.bit_acc == 1).mean(),
                 ge08=(d.bit_acc >= 0.8).mean(), conf=d.conf.mean(), n=len(d))
        rob.append(r)
        L.append(f"| {a} | {fmt(r['bit_acc'], r['bit_acc_std'])} | {r['ber']:.3f} | "
                 f"{r['perfect']:.0%} | {r['ge08']:.0%} | {r['conf']:.3f} |")
    pd.DataFrame(rob).to_csv(os.path.join(C.RESULTS, "e0_robustness_table.csv"), index=False)
    L.append("")

    # ---- E1a ----
    sat = e1a.frac_util_gt_0p95.mean()
    verdict = ("BINDING" if sat >= 0.25 else "SLACK" if sat < 0.10 else "PARTLY BINDING")
    L += ["## E1a — how much of AWARE's budget is used", "",
          "| quantity | mean over clips | min | max |", "|---|---|---|---|"]
    for k in ("util_mean", "util_median", "frac_util_gt_0p95", "frac_util_gt_0p5",
              "frac_util_lt_0p05", "quiet_util_mean", "loud_util_mean", "quiet_frac_sat",
              "loud_frac_sat", "sat_pushed_up", "spearman_frame_energy_vs_util",
              "rel_change_energy_db"):
        L.append(f"| {k} | {e1a[k].mean():.3f} | {e1a[k].min():.3f} | {e1a[k].max():.3f} |")
    L += ["", f"**Gate 1a verdict: budget is {verdict}** "
              f"({sat:.1%} of coefficients use > 95% of their budget; rule: ≥ 25% binding, < 10% slack).", ""]

    # ---- E1b ----
    L += ["## E1b — which windows carry the watermark", "",
          "Window bit accuracy (1 s windows, 250 ms hop) by loudness quintile of the original audio "
          "(Q1 = quietest).", ""]
    cluster = {}
    if not win.empty:
        L += ["| condition | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 − Q1 | Spearman(dBFS, acc) | speech<0.5 acc | speech≥0.5 acc |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        from scipy.stats import spearmanr
        for cond, d in win.groupby("condition", sort=False):
            d = d.dropna(subset=["bit_acc"]).copy()
            d["q"] = pd.qcut(d.win_dbfs.rank(method="first"), 5, labels=False)
            qm = d.groupby("q").bit_acc.mean()
            gap = qm.iloc[-1] - qm.iloc[0]
            cluster[cond] = gap
            rho = spearmanr(d.win_dbfs, d.bit_acc).correlation
            lo = d[d.speech_frac < 0.5].bit_acc.mean()
            hi = d[d.speech_frac >= 0.5].bit_acc.mean()
            L.append(f"| {cond} | " + " | ".join(f"{v:.3f}" for v in qm) +
                     f" | {gap:+.3f} | {rho:.3f} | {lo:.3f} | {hi:.3f} |")
        clustered = any(abs(v) >= 0.10 for v in cluster.values())
        L += ["", f"**Gate 1b verdict: errors {'CLUSTER by loudness' if clustered else 'do NOT cluster by loudness'}** "
                  f"(largest |Q5 − Q1| = {max(abs(v) for v in cluster.values()):.3f}; rule: ≥ 0.10).", ""]

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        npz = sorted(glob.glob(os.path.join(C.DATA, "e1a_*.npz")))
        utils = []
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for f in npz:
            z = np.load(f)
            init = true_initial(z, float(qual.tolerance_db.iloc[0]))
            delta = z["upper"] - init
            u = np.divide(np.abs(z["final"] - init), delta,
                          out=np.zeros_like(delta), where=delta > 1e-12)
            utils.append(u)
            nf = len(z["freq_indices"])
            U, Mg = u.reshape(nf, -1), init.reshape(nf, -1)
            freqs = z["freq_indices"] * C.SR / 1024
            ax[1].plot(freqs, U.mean(axis=1), alpha=0.3, lw=0.8)
            ax[2].scatter(10 * np.log10(np.sum(Mg ** 2, axis=0) + 1e-12), U.mean(axis=0), s=2, alpha=0.3)
        ax[0].hist(np.concatenate(utils), bins=50, range=(0, 1.0001))
        ax[0].set(title="E1a: budget utilisation, all coefficients", xlabel="|Δ| / budget", ylabel="count")
        ax[0].set_yscale("log")
        ax[1].set(title="Mean utilisation per frequency bin", xlabel="Hz", ylabel="utilisation")
        ax[2].set(title="Per-frame utilisation vs frame energy", xlabel="frame band energy (dB)", ylabel="utilisation")
        fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "e1a_utilisation.png"), dpi=130); plt.close(fig)

        if not win.empty:
            fig, ax = plt.subplots(1, len(cluster), figsize=(5 * len(cluster), 4), squeeze=False)
            for i, (cond, d) in enumerate(win.groupby("condition", sort=False)):
                ax[0, i].scatter(d.win_dbfs, d.bit_acc, s=6, alpha=0.4, c=d.speech_frac, cmap="viridis")
                ax[0, i].set(title=f"E1b: {cond}", xlabel="window loudness (dBFS)", ylabel="window bit accuracy",
                             ylim=(0, 1.05))
            fig.tight_layout(); fig.savefig(os.path.join(C.FIGS, "e1b_window_acc_vs_loudness.png"), dpi=130)
            plt.close(fig)
        L += ["## Figures", "", "- `figures/e1a_utilisation.png`", "- `figures/e1b_window_acc_vs_loudness.png`", ""]
    except Exception as e:
        L += [f"(figures failed: {e})", ""]

    write_params(qual)
    open(os.path.join(C.RESULTS, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
