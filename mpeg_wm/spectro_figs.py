"""
mpeg_wm/spectro_figs.py -- spectrograms of AudioSeal, AWARE, MPEG, CASP-1 and CASP-2 on a few
clips: the watermarked audio, the watermark itself (watermarked - original), and views that
show WHERE each system puts its watermark and WHY CASP-1/2 lose robustness.

Reads the wavs the benchmark already saved (no models are loaded):
    mpeg_wm/work/2026-10-19_mpeg-vox-bench/<clip>/{host,mpeg_*,audioseal_*,aware_*}.wav
    mpeg_wm/work/2026-10-20_casp-vox-bench/<clip>/{casp1_*,casp2_*}.wav

    python spectro_figs.py                                   # 3 default clips, both operating points
    python spectro_figs.py --clips asv003 asv050 asv101 --ops match4.2
    python spectro_figs.py --no-attacks                      # skip figures 08-09 (no ffmpeg needed)

Writes results/2026-10-21_spectrograms/<clip>/<op>/:
    02_stft_linear.png      watermarked | watermark, linear frequency axis (dB)
    03_stft_logfreq.png     the same with a log frequency axis
    04_mel.png              the same as 80-band mel spectrograms
    05_watermark_to_speech_ratio.png   watermark level minus speech level per time-frequency bin
    06_energy_vs_speech_level.png      how much watermark sits under quiet vs loud speech
    07_profiles.png         average spectrum, and level over time
    08_detector_evidence.png  MPEG / CASP-1 / CASP-2 only: per-block bit evidence, clean and attacked
    09_surviving_watermark.png  what is left of each watermark after MP3 16k and 8-level quantization
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_OLD = "2026-10-19_mpeg-vox-bench"
RUN_CASP = "2026-10-20_casp-vox-bench"
SYSTEMS = [("audioseal", "AudioSeal", RUN_OLD), ("aware", "AWARE", RUN_OLD), ("mpeg", "MPEG", RUN_OLD),
           ("casp1", "CASP-1", RUN_CASP), ("casp2", "CASP-2", RUN_CASP)]
SS_SYSTEMS = ("mpeg", "casp1", "casp2")           # share the blind spread-spectrum detector
OUT = os.path.join(ROOT, "results", "2026-10-21_spectrograms")
FS = 16000                                         # analysis rate (all watermarks live below 8 kHz)
N_FFT, HOP = 512, 128
ATTACKS = [("vox_gaussian_noise", "10dB"), ("vox_mp3", "16k"), ("vox_quantization", "8lvl")]


# --------------------------------------------------------------------------- #
#  audio and spectra
# --------------------------------------------------------------------------- #
def load_wav(path):
    """PCM16 wav at 22.05 kHz -> float at 16 kHz."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, y = wavfile.read(path)
    y = y.astype(np.float64) / 32768.0 if y.dtype == np.int16 else y.astype(np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != FS:
        from math import gcd
        g = gcd(sr, FS)
        y = resample_poly(y, FS // g, sr // g)
    return y


def to16(y22):
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(y22, dtype=np.float64), 320, 441)


def power_spec(x):
    """|STFT|^2, Hann 512 / hop 128 at 16 kHz. Returns (f, t, P)."""
    from scipy.signal import stft
    f, t, z = stft(x, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    return f, t, np.abs(z) ** 2


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-14))


def mel_filters(n_mels=80, fmax=FS / 2):
    """Triangular mel filterbank (HTK mel scale), shape (n_mels, N_FFT//2+1)."""
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    edges = hz(np.linspace(mel(0), mel(fmax), n_mels + 2))
    f = np.arange(N_FFT // 2 + 1) * FS / N_FFT
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb, edges[1:-1]


# --------------------------------------------------------------------------- #
#  plotting helpers
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def spec_grid(rows, title, path, axis="linear", vranges=None):
    """rows: list of (row label, [P_col0, P_col1]) power spectrograms (None = empty panel).
    axis: 'linear' | 'log' | 'mel'. vranges: one (vmin, vmax) in dB per column."""
    plt = _plt()
    fb, mel_c = mel_filters() if axis == "mel" else (None, None)
    cols = ["watermarked audio", "watermark only (watermarked - original)"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 2.0 * len(rows)), squeeze=False, sharex=True)
    t = np.arange(rows[0][1][0].shape[1]) * HOP / FS
    f = np.arange(N_FFT // 2 + 1) * FS / N_FFT
    for r, (label, panels) in enumerate(rows):
        for c, p in enumerate(panels):
            ax = axes[r, c]
            if p is None:
                ax.axis("off")
                continue
            vmin, vmax = vranges[c]
            if axis == "mel":
                m = db(fb @ p)
                im = ax.pcolormesh(t, np.arange(len(mel_c)), m, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
                ticks = [np.argmin(abs(mel_c - hz)) for hz in (250, 500, 1000, 2000, 4000, 7000)]
                ax.set_yticks(ticks, ["250", "500", "1k", "2k", "4k", "7k"])
            else:
                im = ax.pcolormesh(t, f, db(p), shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
                if axis == "log":
                    ax.set_yscale("log")
                    ax.set_ylim(60, FS / 2)
            if c == 0:
                ax.set_ylabel(f"{label}\nHz", fontsize=8)
            if all(rows[q][1][c] is None for q in range(r)):         # first panel in this column
                ax.set_title(cols[c], fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("dB", fontsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def clip_files(cid, op):
    files = {"host": os.path.join(HERE, "work", RUN_OLD, cid, "host.wav")}
    for key, _, run in SYSTEMS:
        files[key] = os.path.join(HERE, "work", run, cid, f"{key}_{op}.wav")
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    return files, missing


def payload_and_key(cid, system, run):
    """The payload bits and key that system used on this clip (from the benchmark output)."""
    d = os.path.join(ROOT, "results", run, cid)
    key = json.load(open(os.path.join(d, "meta.json"))).get("mpeg_key")
    for row in csv.DictReader(open(os.path.join(d, "rows.csv"))):
        if row["system"] == system:
            return np.array([int(b) for b in row["payload"]]), key
    return None, key


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def fig_spectra(host, marked, cid, op, out):
    P_host = power_spec(host)[2]
    P_mark = {k: power_spec(marked[k])[2] for k in marked}
    P_wm = {k: power_spec(marked[k] - host)[2] for k in marked}
    top = db(P_host).max()
    wm_top = max(db(p).max() for p in P_wm.values())
    rows = [("Original", [P_host, None])] + [(name, [P_mark[k], P_wm[k]]) for k, name, _ in SYSTEMS if k in marked]
    vr = [(top - 80, top), (wm_top - 80, wm_top)]
    t = f"{cid}, {op}: same colour scale within each column"
    spec_grid(rows, t, os.path.join(out, "02_stft_linear.png"), "linear", vr)
    spec_grid(rows, t, os.path.join(out, "03_stft_logfreq.png"), "log", vr)
    fb, _ = mel_filters()
    mtop, mwm = db(fb @ P_host).max(), max(db(fb @ p).max() for p in P_wm.values())
    spec_grid(rows, t, os.path.join(out, "04_mel.png"), "mel", [(mtop - 80, mtop), (mwm - 80, mwm)])
    return P_host, P_wm


def fig_ratio(P_host, P_wm, cid, op, out):
    """Watermark level minus speech level per bin: where the watermark stands out."""
    plt = _plt()
    keys = [k for k, _, _ in SYSTEMS if k in P_wm]
    names = dict((k, n) for k, n, _ in SYSTEMS)
    t = np.arange(P_host.shape[1]) * HOP / FS
    f = np.arange(N_FFT // 2 + 1) * FS / N_FFT
    hdb = db(P_host)
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.1 * len(keys)), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        im = ax.pcolormesh(t, f, db(P_wm[k]) - hdb, shading="auto", cmap="RdBu_r", vmin=-40, vmax=20)
        ax.contour(t, f, hdb, levels=[hdb.max() - 30], colors="k", linewidths=0.4)
        ax.set_ylabel(f"{names[k]}\nHz", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("watermark - speech (dB)", fontsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(f"{cid}, {op}: watermark-to-speech ratio. Red = watermark close to / above the speech "
                 f"(easy to detect, risky to hear); blue = buried under the speech. Black line: loud speech.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05_watermark_to_speech_ratio.png"), dpi=110)
    plt.close(fig)


def fig_energy_vs_level(P_host, P_wm, cid, op, out):
    """Bin every time-frequency cell by how loud the speech is there, then ask:
    how loud is the watermark in those cells, and what share of its energy sits there."""
    plt = _plt()
    hdb = db(P_host) - db(P_host).max()
    edges = np.arange(-90, 1, 5)
    centre = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(hdb, edges) - 1
    names = dict((k, n) for k, n, _ in SYSTEMS)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, p in P_wm.items():
        lvl, share, ratio = [], [], []
        tot = p.sum()
        for i in range(len(centre)):
            m = which == i
            if m.sum() < 20:
                lvl.append(np.nan), share.append(np.nan), ratio.append(np.nan)
                continue
            lvl.append(db(p[m].mean()) - db(P_host).max())
            share.append(100 * p[m].sum() / tot)
            ratio.append(db(p[m].mean()) - db(P_host[m].mean()))
        a1.plot(centre, lvl, marker="o", ms=3, label=names[k])
        a2.plot(centre, share, marker="o", ms=3, label=names[k])
        a3.plot(centre, ratio, marker="o", ms=3, label=names[k])
    a1.set(xlabel="speech level in the cell (dB re loudest)", ylabel="watermark level (dB re loudest speech)",
           title="how loud the watermark is")
    a2.set(xlabel="speech level in the cell (dB re loudest)", ylabel="% of the watermark's energy",
           title="where the watermark's energy goes")
    a3.set(xlabel="speech level in the cell (dB re loudest)", ylabel="watermark - speech (dB)",
           title="watermark-to-speech ratio (what a blind detector sees)")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(f"{cid}, {op}: a blind detector treats the speech as noise, so energy under loud "
                 f"speech (right side) buys little detection", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_energy_vs_speech_level.png"), dpi=110)
    plt.close(fig)


def fig_profiles(P_host, P_wm, cid, op, out):
    plt = _plt()
    names = dict((k, n) for k, n, _ in SYSTEMS)
    f = np.arange(N_FFT // 2 + 1) * FS / N_FFT
    t = np.arange(P_host.shape[1]) * HOP / FS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7))
    a1.semilogx(f[1:], db(P_host.mean(axis=1))[1:], "k", lw=2, label="speech")
    a2.plot(t, db(P_host.sum(axis=0)), "k", lw=2, label="speech")
    for k, p in P_wm.items():
        a1.semilogx(f[1:], db(p.mean(axis=1))[1:], lw=1, label=names[k])
        a2.plot(t, db(p.sum(axis=0)), lw=1, label=names[k])
    a1.set(xlabel="frequency (Hz)", ylabel="average level (dB)", title="average spectrum")
    a2.set(xlabel="time (s)", ylabel="level per frame (dB)", title="level over time (pauses = dips in the black line)")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7, ncol=6)
    fig.suptitle(f"{cid}, {op}: watermark vs speech", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "07_profiles.png"), dpi=110)
    plt.close(fig)


def _attack(cfg, y22):
    import bm_core as B
    return B.run_attack(cfg, y22)


def _cfg(family, label):
    import bm_core as B
    for c in B.attack_configs([family]):
        if c["label"] == label:
            return c
    raise KeyError((family, label))


def fig_detector(host22, marked22, cid, op, out):
    """MPEG / CASP-1 / CASP-2 share one blind detector: per block, how strongly it votes for
    the RIGHT bit, before and after attacks."""
    plt = _plt()
    import detect as D
    names = dict((k, n) for k, n, _ in SYSTEMS)
    keys = [k for k in SS_SYSTEMS if k in marked22]
    conds = [("clean", None)] + [(f"{fam.replace('vox_', '')} {lab}", _cfg(fam, lab)) for fam, lab in ATTACKS]
    fig, axes = plt.subplots(len(conds), len(keys), figsize=(5 * len(keys), 2.3 * len(conds)),
                             sharex=True, squeeze=False)
    env = np.abs(to16(host22))
    for j, k in enumerate(keys):
        run = RUN_OLD if k == "mpeg" else RUN_CASP
        bits, key = payload_and_key(cid, k, run)
        if bits is None:
            continue
        signs = 2 * bits - 1
        for i, (cname, cfg) in enumerate(conds):
            y = marked22[k] if cfg is None else _attack(cfg, marked22[k])
            if y is None:
                continue
            r = D.block_scores(to16(y), key)
            ev = r * signs[np.arange(len(r)) % len(signs)]
            ev = ev / (np.sqrt(np.mean(r ** 2)) + 1e-30)
            got, _, p = D.decode(to16(y), key, len(bits))
            acc = np.mean(got == bits)
            tb = (np.arange(len(r)) * 256 + 256) / FS
            ax = axes[i, j]
            ax.fill_between(tb, 0, ev, where=ev >= 0, color="tab:green", step="mid", alpha=0.8)
            ax.fill_between(tb, 0, ev, where=ev < 0, color="tab:red", step="mid", alpha=0.8)
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(env)) / FS, env, color="0.85", lw=0.4)
            ax2.set_yticks([])
            ax.set_zorder(ax2.get_zorder() + 1)                   # votes in front of the waveform
            ax.patch.set_visible(False)
            ax.set_title(f"{names[k]}, {cname}: bit acc {acc:.2f}", fontsize=8)
            ax.axhline(0, color="k", lw=0.5)
            if j == 0:
                ax.set_ylabel("vote for the\nright bit", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{cid}, {op}: per-block detector votes (green = right bit, red = wrong); grey = speech waveform",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "08_detector_evidence.png"), dpi=110)
    plt.close(fig)


def fig_surviving(host22, marked22, cid, op, out):
    """attack(watermarked) - attack(original): the part of each watermark an attack leaves."""
    plt = _plt()
    names = dict((k, n) for k, n, _ in SYSTEMS)
    conds = [(f"{fam.replace('vox_', '')} {lab}", _cfg(fam, lab)) for fam, lab in ATTACKS[1:]]
    keys = [k for k, _, _ in SYSTEMS if k in marked22]
    h_att = [to16(_attack(cfg, host22)) for _, cfg in conds]
    fig, axes = plt.subplots(len(keys), len(conds) + 1, figsize=(5 * (len(conds) + 1), 2.0 * len(keys)),
                             sharex=True, squeeze=False)
    f = np.arange(N_FFT // 2 + 1) * FS / N_FFT
    panels = {}
    for k in keys:
        panels[(k, 0)] = power_spec(to16(marked22[k]) - to16(host22))[2]
        for c, (_, cfg) in enumerate(conds):
            m = to16(_attack(cfg, marked22[k]))
            n = min(len(m), len(h_att[c]))
            panels[(k, c + 1)] = power_spec(m[:n] - h_att[c][:n])[2]
    top = max(db(p).max() for p in panels.values())
    for r, k in enumerate(keys):
        for c in range(len(conds) + 1):
            p = panels[(k, c)]
            t = np.arange(p.shape[1]) * HOP / FS
            im = axes[r, c].pcolormesh(t, f, db(p), shading="auto", cmap="magma", vmin=top - 80, vmax=top)
            if r == 0:
                axes[r, c].set_title("watermark before attack" if c == 0 else f"left after {conds[c - 1][0]}",
                                     fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel(f"{names[k]}\nHz", fontsize=8)
        fig.colorbar(im, ax=axes[r, -1], fraction=0.02, pad=0.01)
    fig.suptitle(f"{cid}, {op}: attacked(watermarked) - attacked(original). Codecs drop what they "
                 f"consider inaudible: watermark hidden under the speech goes first.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "09_surviving_watermark.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["asv000", "asv040", "asv080"])
    ap.add_argument("--ops", nargs="+", default=["match4.2", "ship"],
                    help="operating points: 'match4.2' (equal PESQ) and/or 'ship' (default strength)")
    ap.add_argument("--no-attacks", action="store_true", help="skip figures 08-09")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not a.no_attacks:
        code = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(code), "casp"))
        import bm_core  # noqa: F401  (puts cascade/ on sys.path for the attacks)
        sys.path.insert(0, code)
    from scipy.io import wavfile

    for cid in a.clips:
        for op in a.ops:
            files, missing = clip_files(cid, op)
            if "host" in missing:
                print(f"[{cid} {op}] no host.wav under mpeg_wm/work/{RUN_OLD}/{cid}; skipped")
                continue
            if missing:
                print(f"[{cid} {op}] missing {missing} (clip not in that run?); plotting the rest")
            out = os.path.join(a.out, cid, op)
            os.makedirs(out, exist_ok=True)
            host = load_wav(files["host"])
            marked = {}
            for k, _, _ in SYSTEMS:
                if k in missing:
                    continue
                y = load_wav(files[k])
                n = min(len(y), len(host))
                marked[k] = y[:n]
            n = min(len(v) for v in marked.values())
            host = host[:n]
            marked = {k: v[:n] for k, v in marked.items()}
            P_host, P_wm = fig_spectra(host, marked, cid, op, out)
            fig_ratio(P_host, P_wm, cid, op, out)
            fig_energy_vs_level(P_host, P_wm, cid, op, out)
            fig_profiles(P_host, P_wm, cid, op, out)
            if not a.no_attacks:
                def raw(p):                                     # 22.05 kHz for the attacks
                    sr, y = wavfile.read(p)
                    return (y.astype(np.float32) / 32768.0) if y.dtype == np.int16 else y.astype(np.float32)
                host22 = raw(files["host"])
                m22 = {k: raw(files[k])[:len(host22)] for k, _, _ in SYSTEMS if k not in missing}
                fig_detector(host22, m22, cid, op, out)
                fig_surviving(host22, m22, cid, op, out)
            print(f"[{cid} {op}] -> {out}", flush=True)


if __name__ == "__main__":
    main()
