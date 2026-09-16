"""
informed/plot_lowpass_result.py -- one figure for the lowpass result.

Detection rate (1% false alarms on 200 clean clips, 50 watermarked clips) for
blind AWARE, informed before the fix, and informed after the fix, across
lowpass cutoffs. Numbers are from
results/2026-09-11_aware-lowpass-null/null_probe_summary.md.

    python plot_lowpass_result.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("WM_COMPARE_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "results", "2026-09-11_aware-lowpass-null", "lowpass_result.png")

LABELS = ["9.9k", "8.8k", "6.6k", "4.4k", "3.3k", "2.2k", "1.5k", "1.1k", "662", "441"]
FIXED = [100, 100, 100, 100, 100, 84, 58, 4, 0, 0]   # informed16 (16 kHz, whole clip)
BLIND = [100, 100, 100, 100, 100, 86, 28, 4, 0, 0]   # AWARE conf
OLD = [78, 72, 68, 60, 56, 50, 32, 2, 0, 0]          # informed, Phase B method

C_FIXED, C_BLIND, C_OLD = "#2a78d6", "#eb6834", "#1baf7a"

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(LABELS))

ax.plot(x, OLD, "--o", color=C_OLD, lw=2, ms=5, label="Informed, before fix")
ax.plot(x, BLIND, "-o", color=C_BLIND, lw=2.4, ms=6, label="Blind AWARE")
ax.plot(x, FIXED, "-o", color=C_FIXED, lw=2.8, ms=7, label="Informed, fixed (16 kHz, whole clip)")

ax.annotate("58%", (6, 58), xytext=(10, 4), textcoords="offset points",
            color=C_FIXED, fontsize=12, fontweight="bold")
ax.annotate("28%", (6, 28), xytext=(-34, -16), textcoords="offset points",
            color=C_BLIND, fontsize=12, fontweight="bold")
ax.annotate("78%", (0, 78), xytext=(8, -16), textcoords="offset points",
            color=C_OLD, fontsize=11, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(LABELS)
ax.set_xlabel("Lowpass keeps audio below (Hz)   →  stronger filtering", fontsize=11)
ax.set_ylabel("Watermarked clips detected (%)", fontsize=11)
ax.set_ylim(-4, 106)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", color="#e5e4dd", lw=1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=10)

ax.set_title("Watermark detection under lowpass filtering", fontsize=14,
             fontweight="bold", loc="left", pad=22)
ax.text(0, 1.02, "1% false alarms on 200 clean clips · 50 watermarked clips · AWARE",
        transform=ax.transAxes, fontsize=9.5, color="#52514e")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(f"wrote {OUT}")
