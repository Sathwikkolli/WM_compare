"""
fsss/viz.py -- visualizations for understanding the anchor detectors.

Produces four figures (PNGs in fsss_out/viz/) using the standard conventions from
the onset-detection / MIR and desync-watermark literature:

  1. curves.png      -- waveform + each detector's DETECTION FUNCTION (its internal
                        strength curve) with picked anchors as vertical lines.
                        (the canonical onset-detection figure: MIR handbook / FMP C6)
  2. spectrogram.png -- log-STFT spectrogram with all three detectors' anchors
                        overlaid as colored vertical lines (compare WHERE each fires).
  3. repeatability.png -- clean vs attacked: which anchors survive an attack
                        (matched = green, missed = red). The before/after figure
                        used in desync-watermark papers.
  4. heatmap.png     -- detector x attack hit-rate heatmap from exp_a_summary.csv.

Run on Great Lakes in the wmcompare env:
    conda activate wmcompare
    python -m fsss.viz                                  # first Emilia clip, all figures
    python -m fsss.viz --clip audio/client_original_16k.wav
    python -m fsss.viz --attack dynamic_compression --param t-30_r8
    python -m fsss.viz --summary fsss_out/exp_a_summary.csv   # heatmap only needs this
"""

import os
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")                    # headless (no display on the cluster)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(BASE, "cascade"))

from fsss.detectors import DETECTORS, MIN_SEP_MS
from fsss.exp_a_repeatability import load_16k, pick_clips, EMILIA_CSV, WORK_SR, W_MS
import vox_attacks

OUT = os.path.join(BASE, "fsss_out", "viz")
COLORS = {"fsss": "#D85A30", "librosa_flux": "#1D9E75", "ssl_wavlm": "#7F77DD"}
NAMES = {"fsss": "fsss (energy ratio)", "librosa_flux": "librosa (spectral flux)",
         "ssl_wavlm": "wavlm (novelty)"}


# --- detection-function (strength curve) recomputation per detector --------- #
def fsss_curve(x, sr, r_ms=10.0):
    """Energy ratio E_after/E_before at every sample (log for visibility)."""
    x = x.astype(np.float64)
    e = x * x
    c = np.concatenate(([0.0], np.cumsum(e)))
    r = max(1, int(round(r_ms * sr / 1000.0)))
    idx = np.arange(r, len(x) - r)
    eb = c[idx] - c[idx - r]
    ea = c[idx + r] - c[idx]
    er = ea / (eb + 1e-12)
    return idx / sr, np.log1p(er)


def librosa_curve(x, sr, hop=256):
    import librosa
    env = librosa.onset.onset_strength(y=x.astype(np.float32), sr=sr, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    return t, env


def wavlm_curve(x, sr):
    import torch
    from fsss.detectors import _wavlm_model
    model, device = _wavlm_model()
    xf = x.astype(np.float32)
    xf = (xf - xf.mean()) / (xf.std() + 1e-8)
    with torch.no_grad():
        hs = model(torch.from_numpy(xf).float().unsqueeze(0).to(device)).last_hidden_state[0]
        hs = torch.nn.functional.normalize(hs, dim=-1)
        nov = (1.0 - (hs[1:] * hs[:-1]).sum(-1)).cpu().numpy()
    nov = np.concatenate([[0.0], nov])
    t = np.arange(len(nov)) * 320.0 / sr
    return t, nov


def picks_time(name, x, sr):
    try:
        return np.asarray(DETECTORS[name](x, sr)) / sr
    except Exception as e:
        print(f"  {name} picks unavailable: {e}")
        return None


# --- figure 1: waveform + detection functions ------------------------------- #
def fig_curves(x, sr, path):
    have_wavlm = "ssl_wavlm" in _available()
    rows = 4 if have_wavlm else 3
    fig, ax = plt.subplots(rows, 1, figsize=(11, 2.1 * rows), sharex=True)
    t_wav = np.arange(len(x)) / sr
    ax[0].plot(t_wav, x, color="#888780", lw=0.5)
    ax[0].set_ylabel("waveform")
    ax[0].set_title("detection functions: what each detector 'sees', and where it fires")

    panels = [("fsss", fsss_curve, "log(1+E_after/E_before)"),
              ("librosa_flux", librosa_curve, "spectral flux")]
    if have_wavlm:
        panels.append(("ssl_wavlm", wavlm_curve, "1 - cos(frame, prev)"))

    for i, (name, curvefn, ylab) in enumerate(panels, start=1):
        try:
            tc, sc = curvefn(x, sr)
            ax[i].plot(tc, sc, color=COLORS[name], lw=0.8)
        except Exception as e:
            ax[i].text(0.5, 0.5, f"{name}: {e}", transform=ax[i].transAxes, ha="center")
        pts = picks_time(name, x, sr)
        if pts is not None:
            for p in pts:
                ax[i].axvline(p, color=COLORS[name], lw=0.8, alpha=0.35)
        ax[i].set_ylabel(ylab)
        ax[i].legend([NAMES[name]], loc="upper right", fontsize=8)

    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# --- figure 2: spectrogram with anchors overlaid ---------------------------- #
def fig_spectrogram(x, sr, path):
    import librosa
    import librosa.display
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=256)),
                                ref=np.max)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
                             cmap="magma", ax=ax)
    handles = []
    for name in _available():
        pts = picks_time(name, x, sr)
        if pts is None:
            continue
        for p in pts:
            ax.axvline(p, color=COLORS[name], lw=1.0, alpha=0.7)
        handles.append(plt.Line2D([0], [0], color=COLORS[name], label=NAMES[name]))
    ax.set_ylim(0, 8000)
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title("where each detector places anchors, over the spectrogram")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# --- figure 3: repeatability before/after an attack ------------------------- #
def _match_mask(clean_idx, att_idx, sr, w_ms=W_MS, scale=1.0, lag=0):
    """Return boolean array over clean_idx: True if re-found in attacked within w."""
    if len(clean_idx) == 0 or len(att_idx) == 0:
        return np.zeros(len(clean_idx), dtype=bool)
    pred = clean_idx / scale + lag
    w = w_ms * sr / 1000.0
    used = np.zeros(len(att_idx), dtype=bool)
    ok = np.zeros(len(clean_idx), dtype=bool)
    order = np.argsort(pred)
    for j in order:
        d = np.abs(att_idx - pred[j])
        d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= w:
            used[k] = True
            ok[j] = True
    return ok


def fig_repeatability(x, sr, attack, param_label, path, detector="librosa_flux"):
    from fsss.match import align_lag
    grid = dict(vox_attacks.VOX_GRID.get(attack, []))
    if param_label not in grid:
        print(f"[repeat] param '{param_label}' not in {attack}; options: {list(grid)}")
        return
    param = grid[param_label]
    y2 = vox_attacks.apply(attack, param, x.astype(np.float32), sr)
    if y2 is None:
        print(f"[repeat] attack {attack} unavailable")
        return

    clean = np.asarray(DETECTORS[detector](x, sr))
    att = np.asarray(DETECTORS[detector](y2, sr))
    scale = float(param) if attack == "time_stretch" else 1.0
    lag = 0 if attack == "time_stretch" else align_lag(x, y2, sr)
    ok = _match_mask(clean, att, sr, scale=scale, lag=lag)
    hit = ok.mean() if len(ok) else float("nan")

    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    tx = np.arange(len(x)) / sr
    ty = np.arange(len(y2)) / sr
    ax[0].plot(tx, x, color="#B4B2A9", lw=0.5)
    for p, good in zip(clean / sr, ok):
        ax[0].axvline(p, color=("#1D9E75" if good else "#E24B4A"), lw=1.0)
    ax[0].set_title(f"{NAMES[detector]}: anchor survival under {attack} [{param_label}]  "
                    f"-- hit rate {hit:.2f}  (green=survived, red=lost)")
    ax[0].set_ylabel("clean")
    ax[1].plot(ty, y2, color="#B4B2A9", lw=0.5)
    for p in att / sr:
        ax[1].axvline(p, color=COLORS[detector], lw=0.8, alpha=0.6)
    ax[1].set_ylabel("attacked")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  (hit={hit:.2f})")


# --- figure 4: hit-rate heatmap from the summary CSV ------------------------ #
def fig_heatmap(summary_csv, path):
    if not os.path.exists(summary_csv):
        print(f"[heatmap] no summary at {summary_csv} -- run exp_a first")
        return
    rows = list(csv.DictReader(open(summary_csv)))
    if not rows:
        print("[heatmap] summary empty")
        return
    dets = sorted({r["detector"] for r in rows})
    attacks = sorted({r["attack"] for r in rows})
    M = np.full((len(dets), len(attacks)), np.nan)
    for r in rows:                                   # mean hit_rate over params
        i, j = dets.index(r["detector"]), attacks.index(r["attack"])
        v = float(r["hit_rate"])
        M[i, j] = v if np.isnan(M[i, j]) else (M[i, j] + v) / 2

    fig, ax = plt.subplots(figsize=(1.1 * len(attacks) + 3, 0.8 * len(dets) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([NAMES.get(d, d) for d in dets], fontsize=9)
    for i in range(len(dets)):
        for j in range(len(attacks)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="black")
    fig.colorbar(im, ax=ax, label="hit rate (mean over strengths)")
    ax.set_title("anchor repeatability: detector x attack")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# --- helpers / main --------------------------------------------------------- #
_AVAIL = None


def _available():
    global _AVAIL
    if _AVAIL is None:
        probe = (np.random.default_rng(0).standard_normal(2 * WORK_SR) * 0.1).astype("float32")
        _AVAIL = []
        for n in ("fsss", "librosa_flux", "ssl_wavlm"):
            try:
                DETECTORS[n](probe, WORK_SR)
                _AVAIL.append(n)
            except Exception:
                pass
    return _AVAIL


def get_arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    clip = get_arg(argv, "--clip", None)
    attack = get_arg(argv, "--attack", "dynamic_compression")
    param = get_arg(argv, "--param", "t-30_r8")
    summary = get_arg(argv, "--summary", os.path.join(BASE, "fsss_out", "exp_a_summary.csv"))

    if clip is None:
        clips = pick_clips(EMILIA_CSV, 1)
        if not clips:
            print("no clip found; pass --clip PATH")
            return
        clip = clips[0]
    print(f"clip: {clip}")
    print(f"detectors available: {_available()}")

    x = load_16k(clip)
    fig_curves(x, WORK_SR, os.path.join(OUT, "curves.png"))
    fig_spectrogram(x, WORK_SR, os.path.join(OUT, "spectrogram.png"))
    fig_repeatability(x, WORK_SR, attack, param, os.path.join(OUT, "repeatability.png"))
    fig_heatmap(summary, os.path.join(OUT, "heatmap.png"))


if __name__ == "__main__":
    main(sys.argv[1:])
