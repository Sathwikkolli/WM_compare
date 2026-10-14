"""casp/plots.py -- figures for one clip. Called by run_casp.make_all()."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import threshold as TH

SR, HOP = TH.SR, TH.HOP
AWARE_TOL = 6.0


def _t(U):
    return np.arange(U) * HOP / SR


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def make_all(out, cid, x_n, X, bands, blocks, g, maps, A_by, conds, d_jnd, level):
    F, U = X.shape
    f = TH.psy.freqs()
    t = _t(U)
    M_db = 20 * np.log10(np.maximum(np.abs(X), 1e-6))
    f0, f1 = bands[0][0], bands[-1][1]
    Tmain = 1.0 if 1.0 in maps else max(maps)
    tol_fu = TH.tol_db(np.where(maps[Tmain]["eta_fu"] > 0, maps[Tmain]["eta_fu"], np.nan))

    # 1 ---- overview ---------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    tt = np.arange(len(x_n)) / SR
    ax[0].plot(tt, x_n, lw=0.5)
    ax[0].set(ylabel="waveform", title=f"{cid} - host, CASP level {level:g} dB SPL")
    im = ax[1].pcolormesh(t, f / 1000, M_db, shading="auto", cmap="magma",
                          vmin=np.percentile(M_db, 5), vmax=np.percentile(M_db, 99.5))
    ax[1].set(ylabel="kHz", ylim=(0, 5), title="host |STFT| (dB)")
    for y in (f[f0] / 1000, f[f1 - 1] / 1000):
        ax[1].axhline(y, color="w", ls=":", lw=0.8)
    fig.colorbar(im, ax=ax[1], pad=0.01)
    im = ax[2].pcolormesh(t, f[f0:f1] / 1000, tol_fu[f0:f1], shading="auto", cmap="viridis_r",
                          vmin=0, vmax=30)
    ax[2].set(ylabel="kHz", xlabel="time (s)",
              title=f"CASP tolerance map (target {Tmain:g}): dB below |X|; AWARE = {AWARE_TOL:g} dB everywhere.  "
                    f"Dark = more room")
    fig.colorbar(im, ax=ax[2], pad=0.01, label="tolerance (dB)")
    _save(fig, out, "fig1_overview.png")

    # 2 ---- vs AWARE --------------------------------------------------------
    allow = AWARE_TOL - tol_fu                      # +: CASP allows a larger change than AWARE
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    lim = 15
    im = ax[0].pcolormesh(t, f[f0:f1] / 1000, allow[f0:f1], shading="auto", cmap="RdBu_r",
                          vmin=-lim, vmax=lim)
    ax[0].set(xlabel="time (s)", ylabel="kHz",
              title="CASP allowance minus AWARE (dB).  red = CASP would allow MORE, blue = AWARE exceeds CASP")
    fig.colorbar(im, ax=ax[0], pad=0.01)
    vals = allow[f0:f1].ravel()
    vals = vals[np.isfinite(vals)]
    ax[1].hist(vals, bins=60, color="gray")
    ax[1].axvline(0, color="k")
    ax[1].set(xlabel="allowance difference (dB)", ylabel="cells",
              title=f"AWARE exceeds CASP in {100 * np.mean(vals < 0):.0f}% of cells\n"
                    f"median {np.median(vals):+.1f} dB")
    _save(fig, out, "fig2_vs_aware.png")

    # 3 ---- tolerance over time ----------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cm = plt.get_cmap("viridis")
    ctr = np.array([(u0 + u1 - 1) / 2 for u0, u1 in blocks]) * HOP / SR
    hz = g["hz"]
    for b in range(len(bands)):
        lab = f"{hz[b]:.0f}-{hz[b + 1]:.0f} Hz"
        for T, ls in ((1.0, "-"),):
            if T in maps:
                ax[0].plot(ctr, TH.tol_db(maps[T]["eta_kb"][:, b]), ls, color=cm(b / max(len(bands) - 1, 1)),
                           label=lab if T == Tmain else None)
    ax[0].axhline(AWARE_TOL, color="r", ls="--", label="AWARE 6 dB")
    ax[0].set(ylabel="tolerance (dB below |X|)", title=f"CASP tolerance per band, target {Tmain:g} (higher = stricter)")
    ax[0].legend(ncol=4, fontsize=7)
    frame_rms = np.sqrt(np.mean(np.abs(X[f0:f1]) ** 2, axis=0))
    ax[1].plot(t, 20 * np.log10(frame_rms + 1e-9))
    ax[1].set(xlabel="time (s)", ylabel="1-4 kHz level (dB)", title="host level in the embedding band")
    _save(fig, out, "fig3_tolerance_vs_time.png")

    # 4 ---- audibility check (the main figure) -------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    tb = np.array([(u0 + u1 - 1) / 2 for u0, u1 in blocks]) * HOP / SR
    sty = {}
    for name, A in A_by.items():
        fam = "aware" if name.startswith("AWARE") else ("shape" if "shape" in name else "casp")
        n = sty.get(fam, 0)
        sty[fam] = n + 1
        col = {"aware": "tab:red", "casp": "tab:blue", "shape": "tab:purple"}[fam]
        ax[0].plot(tb, A, [":", "-", "--", "-."][n % 4], color=col, label=name, lw=1.3)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].set(xlabel="time (s)", ylabel="audibility A = ||dIR|| / D_JND  (1 = just noticeable)",
              title="Whole-signal audibility per block, same random signs in all conditions")
    ax[0].legend(fontsize=7, ncol=2)
    names = [c["name"] for c in conds]
    ax[1].barh(names, [c["A_block_mean"] for c in conds],
               color=["tab:red" if n.startswith("AWARE") else ("tab:purple" if "shape" in n else "tab:blue")
                      for n in names])
    ax[1].axvline(1.0, color="k", lw=0.8)
    ax[1].set(xlabel="mean block audibility", title="mean over blocks")
    _save(fig, out, "fig4_audibility.png")

    # 5 ---- quality vs energy (equal-quality view) ---------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in conds:
        col = "tab:red" if c["name"].startswith("AWARE") else ("tab:purple" if "shape" in c["name"] else "tab:blue")
        if c["pesq"] is not None:
            ax[0].scatter(c["energy_db"], c["pesq"], color=col)
            ax[0].annotate(c["name"].replace("CASP target", "CASP").replace("AWARE tol", "AWARE").replace(" @AWARE-6dB energy", "@E"),
                           (c["energy_db"], c["pesq"]), fontsize=7)
        ax[1].scatter(c["energy_db"], c["A_block_mean"], color=col)
    ax[0].set(xlabel="energy of the change (dB, AWARE units)", ylabel="PESQ", title="quality vs energy")
    ax[1].set(xlabel="energy of the change (dB, AWARE units)", ylabel="mean block audibility",
              title="CASP audibility vs energy")
    _save(fig, out, "fig5_quality_energy.png")

    # 6 ---- how the solve works: D(eta) curves -------------------------------
    D = g["D"]
    K, nb, G = D.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rng = np.random.default_rng(0)
    for _ in range(min(25, K * nb)):
        k, b = rng.integers(K), rng.integers(nb)
        ax[0].loglog(g["etas"], np.maximum(D[k, b], 1e-6), alpha=0.5)
    for T in sorted(maps):
        ax[0].axhline(T * d_jnd / np.sqrt(nb), color="k", ls=":", lw=0.8)
        ax[0].text(g["etas"][0], T * d_jnd / np.sqrt(nb), f" target {T:g}", fontsize=7)
    ax[0].set(xlabel="eta (relative change)", ylabel="||dIR|| for one band x block",
              title="Probe curves (random cells); dotted = per-band target")
    flags = np.array([[np.mean(maps[T]["flag"] == -1), np.mean(maps[T]["flag"] == 1)] for T in sorted(maps)])
    xs = np.arange(len(maps))
    ax[1].bar(xs - 0.2, flags[:, 0], 0.4, label="too audible even at eta=%.2f (clipped)" % g["etas"][0])
    ax[1].bar(xs + 0.2, flags[:, 1], 0.4, label="inaudible even at eta=%.1f (capped)" % g["etas"][-1])
    ax[1].set_xticks(xs, [f"{T:g}" for T in sorted(maps)])
    ax[1].set(xlabel="audibility target", ylabel="fraction of band x block cells", title="solver range hits")
    ax[1].legend(fontsize=7)
    _save(fig, out, "fig6_probe_curves.png")
