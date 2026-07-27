"""
fsss/viz_salient.py -- visualizations for the salient-point hopping study (exp_v11).

Three figure types, all written as PNGs to fsss_out/:

  1. fig_salient_points(...)  -> v11_salient_points_<clip>.png
     A linear-frequency spectrogram of the clip, a lane plot of where each anchor
     detector (fsss / librosa_flux / ssl_wavlm) fires, and the underlying strength
     curves with their picked peaks. Shows WHERE salient points land and how much
     the detectors agree.

  2. fig_hop_masks(...)       -> v11_hopmask_<clip>.png
     For a fixed band (default 500-4000) and N sub-bands, one panel per segmentation
     method showing the ACTUAL key-driven staircase mask M[freq x frame]: the audio
     spectrogram faint behind, each segment filled at its HMAC-chosen sub-band,
     sub-band edges and salient-point boundaries drawn on. This is the "aha" figure:
     salient points -> segments -> key -> the hopping staircase.

  3. fig_summary_heatmap(...) -> v11_summary_heatmap.png
     seg-method x band grid of mean clean detection confidence across the clips.

The staircase mask is rebuilt for display via the embedder's own helpers
(_band_rows / _segments / _hmac_band), so nothing in staircase.py or the installed
aware package is touched. Frame counts use librosa.stft framing so the mask x-axis
lines up with the displayed spectrogram (illustrative; the embedder's torch STFT may
differ by a frame or two).
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import librosa

try:
    from fsss.detectors import DETECTORS
    from fsss.staircase import StaircaseAWAREEmbedder
except ImportError:
    from detectors import DETECTORS
    from staircase import StaircaseAWAREEmbedder

# consistent colors across all figures
METHOD_COLORS = {
    "fixed":        "#888780",
    "fsss":         "#DD8452",
    "librosa_flux": "#4C72B0",
    "ssl_wavlm":    "#55A868",
}
METHOD_LABEL = {
    "fixed":        "fixed 0.5s",
    "fsss":         "fsss (energy-ratio)",
    "librosa_flux": "librosa (spectral flux)",
    "ssl_wavlm":    "wavlm (SSL novelty)",
}
BAND_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# --------------------------------------------------------------------------- #
#  small signal helpers
# --------------------------------------------------------------------------- #
def _lin_spec_db(audio, sr, n_fft, hop):
    """Linear-frequency magnitude spectrogram in dB (y axis = Hz, matches mask bins)."""
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=np.max)


def _n_frames(audio, n_fft, hop):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop).shape[1]


def _fsss_er_curve(audio, sr, hop, r_ms=10.0):
    """Frame-rate FSSS energy-ratio curve (log-compressed, normalized to [0,1])."""
    x = np.asarray(audio, dtype=np.float64)
    r = max(1, int(round(r_ms * sr / 1000.0)))
    if len(x) < 2 * r + 1:
        return np.array([]), np.array([])
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    centers = np.arange(r, len(x) - r, hop)
    eb = csum[centers] - csum[centers - r]
    ea = csum[centers + r] - csum[centers]
    er = np.log1p(ea / (eb + 1e-12))
    er = (er - er.min()) / (er.max() - er.min() + 1e-12)
    return centers / float(sr), er


def _anchor_times(audio, sr, anchor, rate):
    """Salient-point times (s) for one anchor detector; [] if it errors/unavailable."""
    try:
        idx = np.asarray(DETECTORS[anchor](audio, sr, target_rate=rate))
        return idx / float(sr)
    except Exception:
        return np.array([])


# --------------------------------------------------------------------------- #
#  figure 1 : salient points on the signal
# --------------------------------------------------------------------------- #
def fig_salient_points(audio, sr, clip_name, anchors, out_path,
                       n_fft=1024, hop=256, anchor_rate=3.5):
    """anchors: ordered list of available anchor names (subset of fsss/librosa/wavlm)."""
    dur = len(audio) / float(sr)
    S_db = _lin_spec_db(audio, sr, n_fft, hop)
    times = {a: _anchor_times(audio, sr, a, anchor_rate) for a in anchors}

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.1, 2.0], hspace=0.28)

    # -- spectrogram --------------------------------------------------------- #
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, sr / 2], vmin=S_db.max() - 80, vmax=S_db.max())
    ax0.set_ylabel("frequency (Hz)")
    ax0.set_title(f"{clip_name}   —   salient points by anchor method "
                  f"(rate ≈ {anchor_rate}/s)", fontsize=11)
    ax0.set_ylim(0, min(6000, sr / 2))
    for a in anchors:
        for t in times[a]:
            ax0.axvline(t, color=METHOD_COLORS[a], lw=0.8, alpha=0.55)

    # -- lanes (one row per detector) --------------------------------------- #
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for i, a in enumerate(anchors):
        if len(times[a]):
            ax1.eventplot(times[a], lineoffsets=i, linelengths=0.8,
                          colors=METHOD_COLORS[a], linewidths=1.6)
        ax1.text(-0.012, i, METHOD_LABEL[a], ha="right", va="center",
                 fontsize=9, color=METHOD_COLORS[a], clip_on=False,
                 transform=ax1.get_yaxis_transform())
    ax1.set_ylim(-0.6, len(anchors) - 0.4)
    ax1.set_yticks([])
    ax1.set_ylabel("anchors", fontsize=9)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)

    # -- strength curves + picked peaks ------------------------------------- #
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    if "librosa_flux" in anchors:
        oenv = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
        ot = librosa.times_like(oenv, sr=sr, hop_length=hop)
        oenv = (oenv - oenv.min()) / (oenv.max() - oenv.min() + 1e-12)
        ax2.plot(ot, oenv, color=METHOD_COLORS["librosa_flux"], lw=1.0,
                 label="librosa onset strength")
        for t in times["librosa_flux"]:
            ax2.plot(t, np.interp(t, ot, oenv), "o", ms=4,
                     color=METHOD_COLORS["librosa_flux"])
    if "fsss" in anchors:
        et, er = _fsss_er_curve(audio, sr, hop)
        if len(et):
            ax2.plot(et, er, color=METHOD_COLORS["fsss"], lw=1.0,
                     label="fsss energy-ratio (log)")
            for t in times["fsss"]:
                ax2.plot(t, np.interp(t, et, er), "o", ms=4,
                         color=METHOD_COLORS["fsss"])
    ax2.set_ylabel("strength (norm.)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax2.set_xlim(0, dur)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
#  figure 2 : the key-driven staircase mask, per segmentation method
# --------------------------------------------------------------------------- #
def _band_index_image(st, audio, sr, n_freq, n_frames):
    """(freq x frame) int image: 0 = empty, b+1 = sub-band b written there.
    Also returns the segment start frames (the hop boundaries)."""
    rows = st._band_rows(sr, n_freq)
    segs = st._segments(audio, sr, n_frames)
    img = np.zeros((n_freq, n_frames), dtype=int)
    bounds = []
    for i, (s, e) in enumerate(segs):
        b = st._hmac_band(i)
        img[rows[b], s:e] = b + 1
        if i > 0:
            bounds.append(s)
    return img, segs


def fig_hop_masks(embedder, audio, sr, clip_name, methods, out_path,
                  band=(500, 4000), key="thesis", n_bands=3, seglen=0.5,
                  anchor_rate=3.5):
    """methods: ordered list of (name, segment_mode, anchor)."""
    n_fft = embedder.frame_length
    hop = embedder.hop_length
    n_freq = n_fft // 2 + 1
    nfr = _n_frames(audio, n_fft, hop)
    dur = len(audio) / float(sr)
    S_db = _lin_spec_db(audio, sr, n_fft, hop)

    lo, hi = band
    edges = np.linspace(lo, hi, n_bands + 1)
    # discrete cmap: 0 transparent, 1..N band colors
    cmap = ListedColormap([(0, 0, 0, 0)] + [BAND_COLORS[b] for b in range(n_bands)])
    norm = BoundaryNorm(np.arange(-0.5, n_bands + 1.5, 1), cmap.N)

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes = axes[0]
    ylo, yhi = max(0, lo - 300), min(sr / 2, hi + 300)

    for ax, (mname, mode, anchor) in zip(axes, methods):
        st = StaircaseAWAREEmbedder.from_embedder(
            embedder, key=key, n_bands=n_bands, band_range=band,
            segment_mode=mode, anchor=(anchor or "librosa_flux"),
            anchor_rate=anchor_rate, segment_len_s=seglen)
        st.embedding_bands = band
        img, segs = _band_index_image(st, audio, sr, n_freq, nfr)

        ax.imshow(S_db, origin="lower", aspect="auto", cmap="gray_r", alpha=0.35,
                  extent=[0, dur, 0, sr / 2], vmin=S_db.max() - 80, vmax=S_db.max())
        ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                  extent=[0, dur, 0, sr / 2], alpha=0.85, interpolation="nearest")
        for ed in edges:                                   # sub-band boundaries
            ax.axhline(ed, color="k", lw=0.5, alpha=0.35, ls=":")
        for (s, e) in segs[1:]:                            # hop boundaries
            ax.axvline(s * hop / sr, color="k", lw=0.7, alpha=0.45, ls="--")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, dur)
        ax.set_title(f"{METHOD_LABEL[mname]}\n{len(segs)} segments",
                     fontsize=9, color=METHOD_COLORS[mname])
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")

    # legend for the band colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLORS[b]) for b in range(n_bands)]
    labels = [f"b{b}: {int(edges[b])}–{int(edges[b+1])} Hz" for b in range(n_bands)]
    fig.legend(handles, labels, loc="lower center", ncol=n_bands, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{clip_name}   —   key-driven hop staircase   "
                 f"(band {lo}-{hi}, N={n_bands}, key='{key}')", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
#  figure 3 : summary heatmap (method x band, mean clean conf)
# --------------------------------------------------------------------------- #
def fig_summary_heatmap(conf_mean, acc_mean, methods, bands, out_path):
    """conf_mean/acc_mean: dict[(method, band_name)] -> float. methods/bands: names."""
    M = np.array([[conf_mean.get((m, b), np.nan) for b in bands] for m in methods])
    A = np.array([[acc_mean.get((m, b), np.nan) for b in bands] for m in methods])

    fig, ax = plt.subplots(figsize=(1.9 * len(bands) + 2, 0.9 * len(methods) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods])
    ax.set_title("clean detection — mean confidence\n(bit_acc in parentheses)", fontsize=10)
    for i in range(len(methods)):
        for j in range(len(bands)):
            if not np.isnan(M[i, j]):
                txt = f"{M[i, j]:.2f}\n({A[i, j]:.2f})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="confidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
