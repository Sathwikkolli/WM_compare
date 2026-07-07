#!/usr/bin/env python3
"""
AURA watermark robustness battery.

Embeds an AURA 32-bit watermark into a full-length audio file (by tiling the
model's fixed 2-second / 48 kHz window), then applies a battery of attacks that
mirror the standard 8-case test criteria (Quality / Compression / Format chain /
Editing / Signal processing / Platform simulation / Re-recording simulation) and
runs AURA detection on every produced file.

Output: a results folder containing every watermarked+attacked file plus
`results.csv`, `results.json`, and `results.md` — a table in the same shape as
the reference sheet (Detection Result / Detection Probability / decoded bits).

Run on Great Lakes (needs the checkpoint + a GPU):

    python aura_battery.py \
        --checkpoint ../../aura_watermark/checkpoints/run_002/step_0200000_final.pt \
        --input      ../audio/client_original.mp3 \
        --outdir     aura_battery_out

AURA is a fixed-window model with no sync layer, so length/tempo-changing
attacks (pitch shift, time-stretch) desync the windows — expect those to fail,
exactly as they do for the reference system's "Signal processing" case.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

# ── make the aura_watermark package importable regardless of CWD ──────────────
_HERE = Path(__file__).resolve().parent
for cand in (_HERE / ".." / ".." / "aura_watermark",          # WM_compare/../aura_watermark
             Path.home() / "projects" / "aura_watermark"):     # Great Lakes layout
    cand = cand.resolve()
    if (cand / "aura_watermark").is_dir():
        sys.path.insert(0, str(cand))
        break

# Reuse the project's own inference helpers so we never drift from training code.
from infer import (  # noqa: E402
    load_model, embed_watermark, detect_watermark,
    bits_to_str, str_to_bits, compute_snr,
)
import torchaudio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aura.battery")

SR = 48_000          # AURA sample rate
WIN = 96_000         # AURA window = 2 s
N_BITS = 32
DETECT_THRESHOLD = 0.70   # bit-accuracy at/above which we call it "Detected"


# ═════════════════════════════════════════════════════════════════════════════
# ffmpeg helpers  (codecs / format conversions)
# ═════════════════════════════════════════════════════════════════════════════
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. On Great Lakes: `module load ffmpeg` "
            "or `conda install -c conda-forge ffmpeg`."
        )
    return exe


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _transcode(src: Path, dst: Path, *codec_args: str) -> None:
    """src -> dst via ffmpeg with the given output codec args."""
    _run([_ffmpeg(), "-y", "-i", str(src), *codec_args, str(dst)])


# ═════════════════════════════════════════════════════════════════════════════
# audio I/O
# ═════════════════════════════════════════════════════════════════════════════
def load_full(path: Path) -> torch.Tensor:
    """Load any file -> mono, 48 kHz, peak-normalised, full length. Returns [1, T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak
    return wav  # [1, T]


def save_wav(wav: torch.Tensor, path: Path) -> None:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.cpu().clamp(-1, 1), SR)


# ═════════════════════════════════════════════════════════════════════════════
# embed / detect across the whole file (2-s window tiling)
# ═════════════════════════════════════════════════════════════════════════════
def embed_full(embedder, audio: torch.Tensor, message: torch.Tensor,
               device: torch.device) -> torch.Tensor:
    """Watermark a full clip by tiling non-overlapping 2-s windows. Returns [1, T]."""
    T = audio.shape[-1]
    n_win = max(1, T // WIN)
    out = audio.clone()
    for i in range(n_win):
        s = i * WIN
        chunk = audio[:, s:s + WIN].to(device)               # [1, WIN]
        wm = embed_watermark(embedder, chunk, message.to(device))
        out[:, s:s + WIN] = wm.detach().cpu()
    return out  # tail shorter than one window is left un-watermarked


def detect_full(embedder, detector, audio: torch.Tensor, message: torch.Tensor,
                device: torch.device) -> Tuple[float, str, int]:
    """
    Decode every 2-s window, compare to the embedded message.

    Returns (mean_bit_accuracy, majority_voted_bitstring, n_windows).
    """
    T = audio.shape[-1]
    n_win = max(1, T // WIN)
    tgt = message.long().flatten()
    accs: List[float] = []
    votes = torch.zeros(N_BITS, dtype=torch.long)
    for i in range(n_win):
        s = i * WIN
        chunk = audio[:, s:s + WIN]
        if chunk.shape[-1] < WIN:                            # pad final short window
            chunk = torch.nn.functional.pad(chunk, (0, WIN - chunk.shape[-1]))
        _, bits = detect_watermark(embedder, detector, chunk.to(device))
        bits = bits.long().flatten().cpu()
        accs.append((bits == tgt).float().mean().item())
        votes += bits
    decoded = (votes >= (n_win / 2)).long()                  # per-bit majority vote
    return float(sum(accs) / len(accs)), bits_to_str(decoded), n_win


# ═════════════════════════════════════════════════════════════════════════════
# attacks  →  each returns the path to a produced file
# ═════════════════════════════════════════════════════════════════════════════
def a_quality(wm_wav: Path, out: Path) -> Path:
    dst = out / "test1_quality.wav"
    shutil.copy(wm_wav, dst)
    return dst


def _mp3(wm_wav: Path, out: Path, kbps: int) -> Path:
    dst = out / f"test2_mp3_{kbps}k.mp3"
    _transcode(wm_wav, dst, "-codec:a", "libmp3lame", "-b:a", f"{kbps}k")
    return dst


def a_format_chain(wm_wav: Path, out: Path) -> Path:
    """wav -> mp3(128k) -> wav -> flac -> wav  (lossy round-trip through formats)."""
    dst = out / "test3_format_chain.wav"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _transcode(wm_wav, td / "a.mp3", "-codec:a", "libmp3lame", "-b:a", "128k")
        _transcode(td / "a.mp3", td / "b.wav", "-ar", str(SR))
        _transcode(td / "b.wav", td / "c.flac")
        _transcode(td / "c.flac", dst, "-ar", str(SR))
    return dst


def a_editing(wm_wav: Path, out: Path) -> Path:
    """Editing: trim 0.25 s head, insert 0.5 s silence mid-clip, drop a 0.25 s slice."""
    dst = out / "test4_editing.wav"
    wav = load_full(wm_wav)
    T = wav.shape[-1]
    head = int(0.25 * SR)
    wav = wav[:, head:]                                       # trim head
    mid = wav.shape[-1] // 2
    sil = torch.zeros(1, int(0.5 * SR))
    wav = torch.cat([wav[:, :mid], sil, wav[:, mid:]], dim=-1)   # insert silence
    cut = int(0.25 * SR)
    q = wav.shape[-1] // 4
    wav = torch.cat([wav[:, :q], wav[:, q + cut:]], dim=-1)      # drop a slice
    save_wav(wav, dst)
    return dst


def a_signal(wm_wav: Path, out: Path) -> Path:
    """Signal processing: EQ (hp+lp+peak) + pitch shift +1 semitone + normalize."""
    dst = out / "test5_signal.wav"
    wav = load_full(wm_wav)
    # EQ
    wav = torchaudio.functional.highpass_biquad(wav, SR, 120.0)
    wav = torchaudio.functional.lowpass_biquad(wav, SR, 12_000.0)
    wav = torchaudio.functional.equalizer_biquad(wav, SR, 3_000.0, gain=4.0, Q=1.0)
    # pitch shift +1 semitone (length preserved -> desyncs AURA windows)
    wav = torchaudio.functional.pitch_shift(wav, SR, n_steps=1)
    # normalize
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak * 0.98
    save_wav(wav, dst)
    return dst


def a_platform_opus(wm_wav: Path, out: Path) -> Path:
    """Platform simulation: Opus 24 kbps round-trip (Discord/WhatsApp-style)."""
    dst = out / "test6_platform_opus.wav"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _transcode(wm_wav, td / "a.opus", "-codec:a", "libopus", "-b:a", "24k")
        _transcode(td / "a.opus", dst, "-ar", str(SR))
    return dst


def a_rerecord(wm_wav: Path, out: Path) -> Path:
    """Re-recording sim: room reverb (synthetic RIR) + band-limit + noise + gain."""
    dst = out / "test7_rerecord_sim.wav"
    wav = load_full(wm_wav)
    # synthetic exponential-decay RIR, ~200 ms
    rir_len = int(0.2 * SR)
    t = torch.arange(rir_len, dtype=torch.float32)
    rir = torch.randn(rir_len) * torch.exp(-t / (0.05 * SR))
    rir[0] = 1.0
    rir = rir / rir.abs().max()
    wet = torchaudio.functional.fftconvolve(wav, rir.unsqueeze(0))[:, : wav.shape[-1]]
    # mic band-limit
    wet = torchaudio.functional.highpass_biquad(wet, SR, 90.0)
    wet = torchaudio.functional.lowpass_biquad(wet, SR, 10_000.0)
    # ambient noise @ ~35 dB SNR
    sig_p = wet.pow(2).mean().clamp(min=1e-10)
    noise = torch.randn_like(wet) * (sig_p.sqrt() * 10 ** (-35 / 20))
    wet = wet + noise
    peak = wet.abs().max()
    if peak > 1e-6:
        wet = wet / peak * 0.9
    save_wav(wet, dst)
    return dst


# (label, remarks, fn) — order matches the reference sheet
ATTACKS: List[Tuple[str, str, Callable[[Path, Path], Path]]] = [
    ("Quality",               "watermarked audio, no attack",        a_quality),
    ("Compression",           "MP3 64 kbps",                          lambda w, o: _mp3(w, o, 64)),
    ("Compression",           "MP3 128 kbps",                         lambda w, o: _mp3(w, o, 128)),
    ("Compression",           "MP3 320 kbps",                         lambda w, o: _mp3(w, o, 320)),
    ("Format chain",          "wav->mp3->wav->flac->wav",             a_format_chain),
    ("Editing",               "trim + insert silence + cut slice",    a_editing),
    ("Signal processing",     "EQ + pitch shift + normalize",         a_signal),
    ("Platform simulation",   "Opus 24 kbps round-trip",              a_platform_opus),
    ("Re-recording simulation","reverb + band-limit + noise",         a_rerecord),
]


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="AURA watermark robustness battery")
    ap.add_argument("--checkpoint", required=True, help="AURA .pt checkpoint")
    ap.add_argument("--input", required=True, help="source audio file")
    ap.add_argument("--outdir", default="aura_battery_out", help="results folder")
    ap.add_argument("--bits", default=None, help="32-char 0/1 message (default: fixed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", args.checkpoint)
    embedder, detector, cfg = load_model(args.checkpoint, device)

    if args.bits:
        message = str_to_bits(args.bits, N_BITS)
    else:
        # fixed, reproducible payload
        g = torch.Generator().manual_seed(1234)
        message = torch.randint(0, 2, (N_BITS,), generator=g, dtype=torch.long)
    log.info("Embedded message: %s", bits_to_str(message))

    # ── embed into the full-length clip ──────────────────────────────────────
    src = load_full(Path(args.input))
    wm = embed_full(embedder, src, message, device)
    snr = compute_snr(src[..., : wm.shape[-1]], wm)
    wm_master = out / "_watermarked_master.wav"
    save_wav(wm, wm_master)
    log.info("Watermarked master saved (%.1f s, SNR %.1f dB)", wm.shape[-1] / SR, snr)

    # ── run the battery ──────────────────────────────────────────────────────
    rows: List[Dict] = []
    for label, remark, fn in ATTACKS:
        try:
            fpath = fn(wm_master, out)
            att = load_full(fpath)
            acc, decoded, n_win = detect_full(embedder, detector, att, message, device)
            detected = acc >= DETECT_THRESHOLD
            log.info("%-24s %-30s acc=%.4f  %s", label, remark, acc,
                     "DETECTED" if detected else "not detected")
        except Exception as e:  # noqa: BLE001
            log.error("%-24s FAILED: %s", label, e)
            fpath, acc, decoded, n_win, detected = None, 0.0, "-", 0, False
        rows.append({
            "test_criteria": label,
            "file": fpath.name if fpath else "-",
            "remarks": remark,
            "detection_result": "Detected" if detected else "Not Detected",
            "detection_probability": round(acc, 4),
            "threshold_used": DETECT_THRESHOLD,
            "watermark_detected": bool(detected),
            "decoded_message_bits": decoded,
            "n_windows": n_win,
        })

    summary = {
        "input": str(Path(args.input).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embedded_message_bits": bits_to_str(message),
        "watermark_snr_db": round(snr, 2),
        "detect_threshold": DETECT_THRESHOLD,
        "results": rows,
    }

    (out / "results.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out / "results.csv", rows)
    _write_md(out / "results.md", summary, rows)
    log.info("Done. Results in %s/", out)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    import csv
    cols = ["test_criteria", "file", "remarks", "detection_result",
            "detection_probability", "threshold_used", "decoded_message_bits", "n_windows"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_md(path: Path, summary: Dict, rows: List[Dict]) -> None:
    lines = [
        "# AURA watermark robustness battery",
        "",
        f"- **Input:** `{summary['input']}`",
        f"- **Checkpoint:** `{summary['checkpoint']}`",
        f"- **Embedded message:** `{summary['embedded_message_bits']}`",
        f"- **Watermark SNR:** {summary['watermark_snr_db']} dB",
        f"- **Detect threshold:** {summary['detect_threshold']} bit-accuracy",
        "",
        "| Test Criteria | File | Remarks | Detection Result | Detection Probability | Decoded Bits |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['test_criteria']} | {r['file']} | {r['remarks']} | "
            f"{r['detection_result']} | {r['detection_probability'] * 100:.2f}% | "
            f"`{r['decoded_message_bits']}` |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
