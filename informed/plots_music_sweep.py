"""
informed/plots_music_sweep.py -- figures for Phase A-1.

Reads results/<run>/data/music_clip*.csv, writes results/<run>/figures/*.png.
Nothing is hardcoded: change the data, the figures change.

  crossing.png              THE headline. Detection confidence and DNSMOS on one
                            chart against SNR, with the 0.50 and 3.0 lines drawn
                            in. The shaded band between where the two curves
                            cross their thresholds IS the vulnerability window.
  metric_disagreement.png   DNSMOS vs PESQ across the same sweep. Tests the
                            parent plan's claim that PESQ would wrongly condemn
                            a normal speech-over-music mix.

Usage:
    python plots_music_sweep.py
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

DETECT_THRESHOLD = 0.50
DNSMOS_FLOOR = 3.0
PESQ_FLOOR = 2.0

C_DET = "#c2410c"      # detection
C_QUAL = "#1d4ed8"     # quality
C_PESQ = "#7c3aed"


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "music_clip*.csv")))
    if not files:
        raise SystemExit(f"no music_clip*.csv in {DATA_DIR} -- run music_sweep.py")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def curve(rows, music, arm, field):
    """(snrs, mean, sd) across clips, sorted by SNR ascending."""
    by = defaultdict(list)
    for r in rows:
        if r["music"] != music or r["arm"] != arm:
            continue
        v = fnum(r.get(field))
        if np.isfinite(v):
            by[fnum(r["snr_db"])].append(v)
    xs = sorted(k for k in by if np.isfinite(k))
    mean = np.array([np.mean(by[x]) for x in xs])
    sd = np.array([np.std(by[x], ddof=1) if len(by[x]) > 1 else 0.0 for x in xs])
    return np.array(xs), mean, sd


def cross_x(xs, ys, thr):
    """SNR where ys falls through thr as SNR decreases. nan if never."""
    order = np.argsort(-xs)
    x, y = xs[order], ys[order]
    for i in range(len(x) - 1):
        if y[i] >= thr > y[i + 1]:
            if y[i] == y[i + 1]:
                return float(x[i + 1])
            return float(x[i + 1] + (thr - y[i + 1]) * (x[i] - x[i + 1]) /
                         (y[i] - y[i + 1]))
    return float("nan")


def fig_crossing(rows, music):
    xs_c, conf, conf_sd = curve(rows, music, "wm", "conf")
    xs_q, dns, dns_sd = curve(rows, music, "wm", "dnsmos_ovrl")
    if not len(xs_c) or not len(xs_q):
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs_c, conf, "o-", color=C_DET, lw=2, ms=5, label="detection confidence")
    ax.fill_between(xs_c, conf - conf_sd, conf + conf_sd, color=C_DET, alpha=0.13)
    ax.axhline(DETECT_THRESHOLD, color=C_DET, ls=":", lw=1.2)
    ax.set_ylabel("detection confidence", color=C_DET)
    ax.tick_params(axis="y", labelcolor=C_DET)
    ax.set_ylim(-0.03, 1.05)

    ax2 = ax.twinx()
    ax2.plot(xs_q, dns, "s-", color=C_QUAL, lw=2, ms=5, label="DNSMOS (quality)")
    ax2.fill_between(xs_q, dns - dns_sd, dns + dns_sd, color=C_QUAL, alpha=0.13)
    ax2.axhline(DNSMOS_FLOOR, color=C_QUAL, ls=":", lw=1.2)
    ax2.set_ylabel("DNSMOS overall (1-5)", color=C_QUAL)
    ax2.tick_params(axis="y", labelcolor=C_QUAL)
    ax2.set_ylim(1, 5)

    s_det = cross_x(xs_c, conf, DETECT_THRESHOLD)
    s_qual = cross_x(xs_q, dns, DNSMOS_FLOOR)

    # The vulnerability window: usable audio, undetectable watermark.
    if np.isfinite(s_det):
        lo = s_qual if np.isfinite(s_qual) else float(min(xs_c))
        if s_det > lo:
            ax.axvspan(lo, s_det, color="#f59e0b", alpha=0.16, zorder=0)
            mid = (lo + s_det) / 2.0
            ax.annotate(f"usable audio,\nno detection\n({s_det - lo:.1f} dB wide)",
                        xy=(mid, 0.5), ha="center", va="center",
                        fontsize=9, color="#92400e")
        ax.axvline(s_det, color=C_DET, ls="--", lw=1)
    if np.isfinite(s_qual):
        ax.axvline(s_qual, color=C_QUAL, ls="--", lw=1)

    ax.invert_xaxis()                       # more music to the right
    ax.set_xlabel("speech-to-music SNR (dB)  —  more music →")
    ax.set_title(f"Where the watermark dies vs where the audio dies  [{music}]",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.22, lw=0.6)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=9)

    sub = ("Shaded band = attacker's room: the audio still sounds acceptable "
           "but the watermark is undetectable.")
    if not np.isfinite(s_qual):
        sub = ("DNSMOS never fell through the floor in this range — the audio "
               "stayed usable at every SNR tested.")
    fig.text(0.01, -0.02, sub, fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def fig_metric_disagreement(rows, music):
    xs_d, dns, _ = curve(rows, music, "wm", "dnsmos_ovrl")
    xs_p, pesq, _ = curve(rows, music, "wm", "pesq")
    if not len(xs_d) or not len(xs_p):
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs_d, dns, "s-", color=C_QUAL, lw=2, ms=5,
            label=f"DNSMOS (no-reference), floor {DNSMOS_FLOOR}")
    ax.axhline(DNSMOS_FLOOR, color=C_QUAL, ls=":", lw=1.2)
    ax.plot(xs_p, pesq, "^-", color=C_PESQ, lw=2, ms=5,
            label=f"PESQ (vs clean speech), floor {PESQ_FLOOR}")
    ax.axhline(PESQ_FLOOR, color=C_PESQ, ls=":", lw=1.2)

    s_d = cross_x(xs_d, dns, DNSMOS_FLOOR)
    s_p = cross_x(xs_p, pesq, PESQ_FLOOR)
    if np.isfinite(s_d) and np.isfinite(s_p) and s_p > s_d:
        ax.axvspan(s_d, s_p, color="#7c3aed", alpha=0.10, zorder=0)
        ax.annotate("PESQ says destroyed,\nDNSMOS says usable",
                    xy=((s_d + s_p) / 2, 2.6), ha="center", fontsize=9,
                    color="#5b21b6")

    ax.invert_xaxis()
    ax.set_ylim(1, 5)
    ax.set_xlabel("speech-to-music SNR (dB)  —  more music →")
    ax.set_ylabel("quality score (both on a 1-5 scale)")
    ax.set_title(f"Fidelity vs usability: the two metrics disagree  [{music}]",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.22, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(0.01, -0.02,
             "PESQ measures distance from the clean original. An attacker does "
             "not need fidelity -- only that the result sounds acceptable.",
             fontsize=8, color="#4b5563")
    fig.tight_layout()
    return fig


def main():
    rows = load()
    musics = sorted({r["music"] for r in rows})
    os.makedirs(FIG_DIR, exist_ok=True)

    for music in musics:
        suffix = "" if len(musics) == 1 else f"_{music}"
        for name, fn in (("crossing", fig_crossing),
                         ("metric_disagreement", fig_metric_disagreement)):
            fig = fn(rows, music)
            if fig is None:
                print(f"  skipped {name}{suffix} -- no data")
                continue
            out = os.path.join(FIG_DIR, f"{name}{suffix}.png")
            fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
