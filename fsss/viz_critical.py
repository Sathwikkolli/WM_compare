"""
fsss/viz_critical.py -- figures for exp_v16 (critically-sampled subband embedding).

Six figures, each answering one question that is hard to see in a table:

  F1 fold diagram      WHY the decimation factor must equal the band count.
                       Schematic, drawn from CriticalPlan arithmetic alone --
                       no audio. This is the figure to put in front of someone
                       who does not believe x[::10] can be lossless.
  F2 placement         Where the strip actually goes: original spectrogram,
                       the isolated strip, and the host as AWARE reads it.
                       Makes the "relabel" concrete.
  F3 round-trip null   analyze -> synthesize with nothing embedded. If this is
                       not silent, every later figure is measuring plumbing.
  F4 watermark map     |watermarked - original| in the STFT domain. Should be
                       confined to the strip and nowhere else.
  F5 salient mask      The exp1b region mask over the host spectrogram, with the
                       achieved coverage in the title. Shows immediately whether
                       gating gated anything.
  F6 robustness curves bit_acc per attack strength, one line per config, read
                       from exp_v16's CSV.

F1 needs nothing. F2-F5 need one audio clip (and F4 needs an embed, so it is the
slow one). F6 needs fsss_out/exp_v16_critical_decimation.csv to exist.

Everything writes PNG into fsss_out/ and nothing is shown interactively, so this
is safe to run on a compute node.

    conda activate wmcompare
    python -m fsss.viz_critical                       # all figures it can make
    python -m fsss.viz_critical --figs 1,6            # just those
    python -m fsss.viz_critical --clip audio/x.wav --no-embed
"""

import os
import sys
import csv as _csv
import numpy as np

import matplotlib
matplotlib.use("Agg")                                  # no display on a compute node
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

OUT_DIR = os.path.join(BASE, "fsss_out")
CSV_NAME = "exp_v16_critical_decimation.csv"
WORK_SR = 16000
DPI = 130


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _spec_db(x, n_fft=1024, hop=256):
    """Magnitude spectrogram in dB, floored 80 dB below peak so the colour scale
    is not dominated by numerical dust."""
    import librosa
    S = np.abs(librosa.stft(np.asarray(x, dtype=np.float32), n_fft=n_fft,
                            hop_length=hop))
    db = 20.0 * np.log10(S + 1e-10)
    return np.maximum(db, db.max() - 80.0)


def _show_spec(ax, x, sr, title, n_fft=1024, hop=256):
    db = _spec_db(x, n_fft, hop)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(x) / sr, 0, sr / 2])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=7)
    return db


# --------------------------------------------------------------------------- #
#  F1 -- the fold diagram (no audio needed)
# --------------------------------------------------------------------------- #
def fig1_folds(strip=2, nbands=10, sr=WORK_SR, bad_factor=8):
    """Why the decimation factor must equal the band count.

    Sampling folds the frequency axis at every multiple of fs/2. A band inside
    one fold survives; a band straddling a fold is destroyed. Two panels: the
    correct factor (folds land exactly on the strip edges) and a wrong one (a
    fold cuts the strip in half).
    """
    W = sr / (2.0 * nbands)
    lo, hi = strip * W, (strip + 1) * W
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)

    for ax, M, verdict in ((axes[0], nbands, "CORRECT"),
                           (axes[1], bad_factor, "BROKEN")):
        fs_dec = sr / M
        creases = np.arange(0, sr / 2 + 1e-9, fs_dec / 2.0)
        cut = [c for c in creases if lo + 1e-9 < c < hi - 1e-9]

        ax.add_patch(Rectangle((lo, 0.15), hi - lo, 0.7,
                               color="#2a7fbf" if not cut else "#c0392b",
                               alpha=0.55))
        for c in creases:
            ax.axvline(c, color="0.35", lw=1.0, ls="--")
        for c in cut:
            ax.axvline(c, color="k", lw=2.4)
            ax.annotate("fold cuts the strip", xy=(c, 0.93),
                        xytext=(c + 260, 0.99), fontsize=8, color="#c0392b",
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

        ax.set_xlim(0, sr / 2)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.set_title(f"{verdict}: keep every {M}th sample  ->  folds every "
                     f"{fs_dec/2:.0f} Hz   "
                     f"(strip {strip} = {lo:.0f}-{hi:.0f} Hz)", fontsize=9)
    axes[1].set_xlabel("frequency (Hz)", fontsize=9)
    fig.suptitle("F1  Why the decimation factor must equal the band count",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "v16_f1_folds.png")


# --------------------------------------------------------------------------- #
#  F2 -- placement
# --------------------------------------------------------------------------- #
def fig2_placement(x, plan, taps):
    """Original / isolated strip / host as AWARE reads it.

    The third panel is the point: the host is plotted against sr, not against
    its true rate sr/M, because that is exactly what AWARE does -- torch.stft
    never asks for a sample rate and the mel bank was built once at sr. The
    strip's content therefore appears stretched across the full axis.
    """
    import torch
    from fsss.band_critical import t_analyze

    xt = torch.as_tensor(np.asarray(x, dtype=np.float64))
    lo, host = t_analyze(xt, plan, taps)
    hi = (xt - lo).numpy()
    host = host.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    _show_spec(axes[0], x, WORK_SR, "original")
    axes[0].axhline(plan.slot[0], color="cyan", lw=1.0)
    axes[0].axhline(plan.slot[1], color="cyan", lw=1.0)

    _show_spec(axes[1], hi, WORK_SR,
               f"strip {plan.k} isolated ({plan.slot[0]:.0f}-{plan.slot[1]:.0f} Hz, "
               f"pass {plan.f_lo:.0f}-{plan.f_hi:.0f})")

    _show_spec(axes[2], host, WORK_SR,
               f"host: decimated x{plan.M}, read as if {WORK_SR} Hz\n"
               f"{len(host)} samples, content at "
               f"{plan.perceived(plan.f_lo):.0f}-{plan.perceived(plan.f_hi):.0f} Hz")
    fig.suptitle("F2  Where the strip goes (panel 3 is what the detector sees)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "v16_f2_placement.png")


# --------------------------------------------------------------------------- #
#  F3 -- round-trip null
# --------------------------------------------------------------------------- #
def fig3_null(x, plan, taps):
    """analyze -> synthesize with nothing embedded.

    The residual is what the chain costs before any watermark exists. If it is
    not far below the signal, exp1's numbers are measuring the plumbing.
    """
    import torch
    from fsss.band_critical import t_analyze, t_synthesize

    xt = torch.as_tensor(np.asarray(x, dtype=np.float64))
    y = t_synthesize(*t_analyze(xt, plan, taps), plan, taps).numpy()
    err = np.asarray(x, dtype=float) - y

    trim = 4096
    a, b = np.asarray(x, dtype=float)[trim:-trim], y[trim:-trim]
    snr = 10.0 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-300))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _show_spec(axes[0], x, WORK_SR, "original")
    _show_spec(axes[1], err, WORK_SR,
               f"residual  x - synthesize(analyze(x))\nSNR {snr:.1f} dB "
               f"(want > 40; edges excluded)")
    fig.suptitle("F3  Round-trip null -- the chain's cost before any watermark",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "v16_f3_null.png")
    return snr


# --------------------------------------------------------------------------- #
#  F4 -- where the watermark actually landed
# --------------------------------------------------------------------------- #
def fig4_watermark(x, wm, plan, label="exp1a"):
    """|watermarked - original| in the STFT domain.

    Everything outside the strip is the untouched remainder `lo`, so the
    difference there should be numerically zero. Energy showing up outside the
    cyan lines means the synthesis filter is leaking.
    """
    d = np.asarray(wm, dtype=float)[:len(x)] - np.asarray(x, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _show_spec(axes[0], x, WORK_SR, "original")
    _show_spec(axes[1], d, WORK_SR,
               f"|watermarked - original|  ({label})\n"
               f"should be confined to {plan.slot[0]:.0f}-{plan.slot[1]:.0f} Hz")
    for ax in axes:
        ax.axhline(plan.slot[0], color="cyan", lw=1.0)
        ax.axhline(plan.slot[1], color="cyan", lw=1.0)
    fig.suptitle("F4  Where the watermark landed", fontsize=11)
    fig.tight_layout()
    _save(fig, f"v16_f4_watermark_{label}.png")


# --------------------------------------------------------------------------- #
#  F5 -- salient mask
# --------------------------------------------------------------------------- #
def fig5_mask(host, mask, stats, hop):
    """The exp1b region mask over the host spectrogram.

    Coverage is in the title because it is the number that decides whether the
    exp1a/exp1b comparison means anything. Above ~85% the mask is effectively
    fully-on and the two arms are the same experiment.
    """
    db = _spec_db(host)
    n_frames = min(db.shape[1], mask.shape[1])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.imshow(db[:, :n_frames], origin="lower", aspect="auto", cmap="magma",
              extent=[0, n_frames * hop / WORK_SR, 0, WORK_SR / 2])

    # shade the frames that are NOT writable
    off = ~mask[:, :n_frames].any(axis=0)
    t = np.arange(n_frames) * hop / WORK_SR
    ax.fill_between(t, 0, WORK_SR / 2, where=off, color="0.1", alpha=0.62,
                    step="mid", label="not writable")

    cov = stats.get("coverage", float("nan")) * 100.0
    warn = "  <-- SATURATED, gating is a no-op" if cov > 85 else ""
    ax.set_title(f"F5  exp1b salient mask on the host   "
                 f"{stats.get('n_anchors', '?')} anchors, "
                 f"{stats.get('n_regions', '?')} regions, "
                 f"coverage {cov:.0f}% of {stats.get('n_frames', n_frames)} "
                 f"frames{warn}", fontsize=10)
    ax.set_xlabel("host time (s, compressed)", fontsize=8)
    ax.set_ylabel("Hz (as read)", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "v16_f5_mask.png")


# --------------------------------------------------------------------------- #
#  F6 -- robustness curves from the CSV
# --------------------------------------------------------------------------- #
def fig6_curves(csv_path=None):
    """bit_acc per attack strength, one line per config.

    Reads exp_v16's CSV so it can be re-run without re-embedding. x is the
    attack's own strength label kept in VOX_GRID order, since the grids are not
    all numeric (dynamic_compression carries a tuple).
    """
    csv_path = csv_path or os.path.join(OUT_DIR, CSV_NAME)
    if not os.path.exists(csv_path):
        print(f"  F6 skipped: {csv_path} not found (run exp_v16 first)")
        return

    rows = list(_csv.DictReader(open(csv_path)))
    if not rows:
        print("  F6 skipped: CSV is empty")
        return

    configs, attacks = [], []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])
        if r["attack"] not in attacks and r["attack"] != "clean":
            attacks.append(r["attack"])
    if not attacks:
        print("  F6 skipped: clean-only run, nothing to sweep")
        return

    ncol = 3
    nrow = int(np.ceil(len(attacks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow),
                             squeeze=False)

    for i, attack in enumerate(attacks):
        ax = axes[i // ncol][i % ncol]
        for cfg in configs:
            sel = [r for r in rows if r["config"] == cfg and r["attack"] == attack]
            if not sel:
                continue
            labels, means = [], []
            for lab in dict.fromkeys(r["param"] for r in sel):   # preserves order
                vals = [float(r["bit_acc"]) for r in sel
                        if r["param"] == lab and r["bit_acc"] != ""]
                if vals:
                    labels.append(lab)
                    means.append(float(np.mean(vals)))
            if means:
                ax.plot(range(len(means)), means, marker="o", ms=3.5, lw=1.4,
                        label=cfg)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.axhline(0.8, color="0.5", ls="--", lw=0.9)            # detection floor
        ax.set_ylim(0.35, 1.03)
        ax.set_title(attack, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for j in range(len(attacks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    axes[0][0].set_ylabel("bit accuracy", fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(configs), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("F6  Robustness vs attack strength   "
                 "(dashed = 0.8 detection floor)", fontsize=11)
    fig.tight_layout()
    _save(fig, "v16_f6_curves.png")


# --------------------------------------------------------------------------- #
def main(argv):
    def get_arg(flag, default, cast=str):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    figs = get_arg("--figs", "1,2,3,4,5,6", str)
    want = {int(t) for t in figs.split(",") if t.strip()}
    strip = get_arg("--strip", 2, int)
    nbands = get_arg("--nbands", 10, int)
    dur = get_arg("--dur", 10.0, float)
    clip = get_arg("--clip", "", str)
    no_embed = "--no-embed" in argv

    print("exp_v16 figures ->", OUT_DIR)

    if 1 in want:
        fig1_folds(strip=strip, nbands=nbands)

    if want & {2, 3, 4, 5}:
        import torch
        from fsss.band_critical import CriticalPlan, taps_for
        from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV

        path = clip or (pick_clips(EMILIA_CSV, 1) or [""])[0]
        if not path:
            print("  F2-F5 skipped: no clip (pass --clip)")
            return
        x = load_16k(path)[:int(round(dur * WORK_SR))]
        plan = CriticalPlan(strip, num_bands=nbands, sr=WORK_SR)
        taps = taps_for(plan)
        print(f"  clip: {os.path.basename(path)}  ({len(x)/WORK_SR:.1f}s)")
        print(f"  {plan}")

        if 2 in want:
            fig2_placement(x, plan, taps)
        if 3 in want:
            fig3_null(x, plan, taps)

        if (4 in want or 5 in want) and not no_embed:
            from aware.utils.models import load
            from aware.service import embed_watermark
            from fsss.chain_embed_critical import CriticalBandAWAREEmbedder

            base, _ = load()
            bits = np.random.default_rng(0).integers(0, 2, size=20, dtype=np.int32)

            if 4 in want:
                e = CriticalBandAWAREEmbedder.from_embedder(
                    base, band_index=strip, num_bands=nbands, salient=False)
                wm = embed_watermark(x, sample_rate=WORK_SR,
                                     watermark_bits=bits, model=e)
                fig4_watermark(x, wm, plan, label="exp1a")

            if 5 in want:
                e = CriticalBandAWAREEmbedder.from_embedder(
                    base, band_index=strip, num_bands=nbands, salient=True)
                # build the mask without paying for a full embed
                host = e.to_detector_input(x)
                n_frames = 1 + len(host) // e.hop_length
                e._orig_audio = np.asarray(x, dtype=np.float32)
                e._orig_sr = WORK_SR
                M = e._build_staircase_mask(host, WORK_SR, e.frame_length // 2 + 1,
                                            n_frames)
                fig5_mask(host, M, e.last_mask_stats or {}, e.hop_length)
        elif no_embed:
            print("  F4/F5 skipped (--no-embed)")

    if 6 in want:
        fig6_curves()


if __name__ == "__main__":
    main(sys.argv[1:])
