"""
masking/common.py -- shared pieces for the masking-threshold-in-AWARE experiments.

Everything runs at 16 kHz mono (AWARE's native rate). Paths are Great Lakes
defaults; every one can be overridden with an environment variable.
"""
import os
import sys
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SR = 16000
N_BITS = 20                       # AWARE full_length payload (config_full_length.yaml)
AWARE_MODEL = os.environ.get("AWARE_MODEL", "AWARE")

# ASVspoof2019 LA eval, bona fide only -- same paths as deepfake_probe/run_probe.py
# and Narrow-Frequency-Range-Clone/eval/run_suite.sbatch. Already 16 kHz FLAC.
ASV_AUDIO = os.environ.get(
    "ASV_AUDIO",
    "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac")
ASV_PROTOCOL = os.environ.get(
    "ASV_PROTOCOL", os.path.expanduser("~/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt"))

RUN = os.environ.get("MASK_RUN", "2026-10-06_masking-e0-e1")
RESULTS = os.path.join(ROOT, "results", RUN)
DATA = os.path.join(RESULTS, "data")
FIGS = os.path.join(RESULTS, "figures")
WORK = os.path.join(HERE, "work", RUN)        # audio; gitignored
CLIPS_JSON = os.path.join(HERE, "clips.json")


# --------------------------------------------------------------------------- #
#  AWARE import
# --------------------------------------------------------------------------- #
def import_aware():
    """The repo root holds an empty `aware/` submodule checkout that shadows the
    pip-installed package as a namespace package (see probe_budget_utilization.py).
    Drop the repo root and this folder from sys.path before importing."""
    bad = {os.path.abspath(ROOT), os.path.abspath(HERE)}
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in bad]
    from aware.utils.models import load
    from aware.service import embed_watermark, detect_watermark
    return load, embed_watermark, detect_watermark


class OptimizeCapture:
    """Wraps AWAREEmbedder._optimize to keep its inputs and output, without
    changing the library. After embed(): initial, final, upper (all 1-D, the
    flattened (freq_bin x frame) coefficients of the 1-4 kHz band) and n_freq."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.orig = embedder._optimize
        self.data = {}
        embedder._optimize = self._wrapped

    def _wrapped(self, initial_coeffs, stft_magnitude, watermark_pattern,
                 freq_indices, not_freq_indices, bounds, stft_phase):
        out = self.orig(initial_coeffs, stft_magnitude, watermark_pattern,
                        freq_indices, not_freq_indices, bounds, stft_phase)
        self.data = dict(
            initial=initial_coeffs.detach().cpu().numpy().astype("float32").copy(),
            final=out.detach().cpu().numpy().astype("float32").copy(),
            upper=np.array([b[1] for b in bounds], dtype="float32"),
            lower=np.array([b[0] for b in bounds], dtype="float32"),
            freq_indices=np.asarray(freq_indices),
        )
        return out


# --------------------------------------------------------------------------- #
#  audio io
# --------------------------------------------------------------------------- #
def load_16k(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype("float32")


def write_wav(path, y):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(y, dtype="float32"), SR, subtype="FLOAT")


def bit_acc(decoded, truth):
    d = np.asarray(decoded).astype(int).ravel()[:len(truth)]
    if len(d) == 0:
        return float("nan")
    return float(np.mean(d == np.asarray(truth)[:len(d)]))


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


# --------------------------------------------------------------------------- #
#  attacks -- the "starter" set: overlaps Wang et al. 2025 (DSP 160:105025) and
#  this repo's earlier AWARE benchmarks. All take/return a 16 kHz float array.
# --------------------------------------------------------------------------- #
def _ffmpeg_roundtrip(y, enc_args, ext):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH")
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        src, enc, dec = (os.path.join(td, n) for n in ("in.wav", "enc." + ext, "out.wav"))
        sf.write(src, y, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src] + enc_args + [enc], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc,
                        "-ar", str(SR), "-ac", "1", dec], check=True)
        z, _ = sf.read(dec, dtype="float32")
    return z


def _awgn(y, snr_db, seed=0):
    p = np.sqrt(np.mean(y ** 2)) / (10 ** (snr_db / 20))
    return (y + np.random.RandomState(seed).randn(len(y)).astype("float32") * p).astype("float32")


def _resample_rt(y, sr_mid):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(SR, sr_mid)
    z = resample_poly(resample_poly(y, sr_mid // g, SR // g), SR // g, sr_mid // g)
    return z[:len(y)].astype("float32")


def _lowpass(y, fc, order=8):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(order, fc, fs=SR, output="sos"), y).astype("float32")


def _requant(y, bits):
    L = 2 ** (bits - 1)
    return (np.round(y * L) / L).astype("float32")


def _median(y, k):
    from scipy.signal import medfilt
    return medfilt(y, k).astype("float32")


# name -> (fn, keeps_alignment). Amplitude attacks are not clipped (float wav),
# matching Wang et al.'s AS_0.6 / AS_1.4; AWARE normalises amplitude anyway.
ATTACKS = {
    "clean":        (lambda y: y, True),
    "mp3_128":      (lambda y: _ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "128k"], "mp3"), True),
    "mp3_64":       (lambda y: _ffmpeg_roundtrip(y, ["-codec:a", "libmp3lame", "-b:a", "64k"], "mp3"), True),
    "awgn_25":      (lambda y: _awgn(y, 25), True),
    "resample_8k":  (lambda y: _resample_rt(y, 8000), True),
    "lowpass_6k":   (lambda y: _lowpass(y, 6000), True),
    "amp_0.6":      (lambda y: (y * 0.6).astype("float32"), True),
    "amp_1.4":      (lambda y: (y * 1.4).astype("float32"), True),
    "requant_8bit": (lambda y: _requant(y, 8), True),
    "median_3":     (lambda y: _median(y, 3), True),
    "crop_2048":    (lambda y: y[2048:], False),     # first 128 ms removed
    "crop_6144":    (lambda y: y[6144:], False),     # first 384 ms removed
}


# --------------------------------------------------------------------------- #
#  quality (watermarked vs original)
# --------------------------------------------------------------------------- #
def quality(ref, deg):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    out = dict(pesq=None, stoi=None, snr_db=None, si_snr_db=None, odg=None)
    try:
        from pesq import pesq
        out["pesq"] = float(pesq(SR, ref, deg, "wb"))
    except Exception as e:
        print(f"  pesq failed: {e}")
    try:
        from pystoi import stoi
        out["stoi"] = float(stoi(ref, deg, SR, extended=False))
    except Exception as e:
        print(f"  stoi failed: {e}")
    e = deg - ref
    out["snr_db"] = float(10 * np.log10(np.sum(ref ** 2) / (np.sum(e ** 2) + 1e-12)))
    r, d = ref - ref.mean(), deg - deg.mean()
    proj = (np.dot(d, r) / (np.dot(r, r) + 1e-12)) * r
    out["si_snr_db"] = float(10 * np.log10(np.sum(proj ** 2) / (np.sum((d - proj) ** 2) + 1e-12)))
    out["odg"] = peaq_odg(ref, deg)
    return out


def peaq_odg(ref, deg):
    """PEAQ basic-version ODG via the GstPEAQ command-line tool (`peaq`), if it is
    installed. PEAQ is defined at 48 kHz, so both signals are upsampled first.
    Returns None when the tool is absent -- E0 still runs, the column stays empty."""
    exe = os.environ.get("PEAQ_BIN") or shutil.which("peaq")
    if not exe:
        return None
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "ref.wav"), os.path.join(td, "test.wav")
            sf.write(a, resample_poly(ref, 3, 1), 48000, subtype="PCM_16")
            sf.write(b, resample_poly(deg, 3, 1), 48000, subtype="PCM_16")
            txt = subprocess.run([exe, "--basic", a, b], capture_output=True,
                                 text=True, check=True).stdout
        for line in txt.splitlines():
            if "Objective Difference Grade" in line:
                return float(line.split(":")[-1])
    except Exception as e:
        print(f"  peaq failed: {e}")
    return None
