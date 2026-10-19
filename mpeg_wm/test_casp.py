"""Tests for the CASP ear model and CASP placement. No data or model weights needed.

    python test_casp.py          (or: python -m pytest test_casp.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casp_model as C       # noqa: E402
import casp_place as CP      # noqa: E402
import detect as D           # noqa: E402
import embed as E            # noqa: E402
from test_mpeg import speechlike   # noqa: E402

FS = 16000


def _tone(f, level_db, seconds=0.3):
    t = np.arange(int(seconds * FS)) / FS
    return 10 ** ((level_db - C.FULL_SCALE_DB) / 20) * np.sin(2 * np.pi * f * t)


def test_full_scale_sine_is_96_db_spl():
    p = C.to_pascals(_tone(1000, 96.0))
    assert abs(20 * np.log10(np.sqrt(np.mean(p ** 2)) / C.P_REF) - 96.0) < 0.01


def test_channels_are_half_an_erb_apart():
    cf = C.erb_channels()
    assert len(cf) == 53 and abs(cf[0] - 250) < 1e-6 and cf[-1] <= 8000
    assert np.allclose(np.diff(C.hz_to_erb_rate(cf)), 0.5)


def test_cochlea_is_compressive_at_mid_levels():
    cf = C.erb_channels()
    k = int(np.argmin(abs(cf - 1000)))
    def level(db):
        y = C.drnl(C.outer_middle_ear(C.to_pascals(_tone(1000, db)), FS), FS, cf[k:k + 1])
        return 20 * np.log10(np.sqrt(np.mean(y[0, FS // 10:] ** 2)))
    low, mid = level(30) - level(20), level(70) - level(60)
    print(f"  cochlea growth: {low:.1f} dB per 10 dB at 20-30 dB SPL, {mid:.1f} at 60-70")
    assert 9 < low < 11 and mid < 6


def test_hair_cell_rests_near_minus_64_mv_and_saturates():
    g0 = C.met_conductance(np.array(0.0))
    v = (g0 * C.IHC_EP + C.IHC_GKF * C.IHC_EKF + C.IHC_GKS * C.IHC_EKS) / (g0 + C.IHC_GKF + C.IHC_GKS)
    assert -0.066 < v < -0.061
    assert C.met_conductance(np.array(200e-9)) > 0.99 * C.IHC_GMAX


def test_adaptation_gives_forward_masking():
    cf = C.erb_channels()
    k = int(np.argmin(abs(cf - 1000)))
    m = C.adaptation_min_levels(cf, FS)[k:k + 1]
    n = int(0.2 * FS)
    x = np.concatenate([np.full(n, m[0] * 1e3), np.full(int(0.6 * FS), m[0] * 10)])[None]
    y = C.adaptation(x, FS, m)[0]
    steady = C.adaptation(np.full((1, int(0.8 * FS)), m[0] * 10), FS, m)[0][-1]
    assert y[n + int(0.02 * FS)] < 0 < steady          # probe pushed down 20 ms after the burst
    assert abs(y[n + int(0.4 * FS)] - steady) < 0.2 * steady   # recovered by 400 ms


def test_modulation_centres():
    assert [round(f, 1) for f in C.modulation_centres(1000)] == \
        [0.0, 5.0, 10.0, 16.7, 27.8, 46.3, 77.2, 128.6, 214.3]
    assert max(C.modulation_centres(8000)) <= 1000


def test_calibration_follows_webers_law():
    s = C.check_calibration(FS)
    print(f"  calibration: 2 dB step = {s['two_db']:.2f} x JND, 1 dB at 40 / 80 dB = "
          f"{s['at_40db']:.2f} / {s['at_80db']:.2f} x JND at 60 dB")
    assert 1.6 < s["two_db"] < 2.4 and 0.7 < s["at_40db"] < 1.4 and 0.7 < s["at_80db"] < 1.4


def test_identical_signals_have_zero_dprime():
    x = speechlike(1.0)
    a = C.internal_representation(x, FS)
    assert C.dprime(a["ir"], a["ir"], a["fs_ir"]) == 0.0


def test_way1_without_gains_is_exactly_the_mpeg_watermark():
    host = speechlike(2.0, seed=1)
    bits = np.random.default_rng(2).integers(0, 2, 16)
    levels, weights = CP.base_levels(host, "mpeg")
    wm = CP.synthesise(len(host), bits, 77, levels, weights)
    ref = E.embed_watermark(host, bits, 77).astype(float) - host
    assert np.max(np.abs(wm - ref)) < 1e-6


def test_casp_systems_keep_mpeg_energy_decode_and_shuffle_is_a_permutation():
    host = speechlike(2.0, seed=3)
    bits = np.random.default_rng(4).integers(0, 2, 16)
    key = 1234
    target = CP.mpeg_energy(host, bits, key)
    for way in (1, 2):
        marked, info = CP.embed_casp(host, bits, key, way=way, n_rounds=2)
        shuf, _ = CP.embed_casp(host, bits, key, way=way, shuffle=True, gain_db=info["gain_db"])
        h = info["history"]
        print(f"  way {way}: CASP d' {h[0]['d_total']:.2f} -> {h[-1]['d_total']:.2f}, "
              f"gain spread {h[-1]['spread_db']:.1f} dB")
        for m in (marked, shuf):
            e = float(np.sum((m.astype(float) - host) ** 2))
            assert abs(e / target - 1) < 1e-3
            got, _, p = D.decode(m, key, 16)
            assert np.array_equal(got, bits) and p < 1e-4
        g = info["gain_db"]
        assert np.allclose(np.sort(CP.shuffle_gains(g, key).ravel()), np.sort(g.ravel()))
        assert h[-1]["d_max_tile"] <= h[0]["d_max_tile"] * 1.05     # loudest tile got quieter


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
