"""
fsss/compare_exp1_exp2.py -- one table with BOTH experiments side by side.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
fsss/exp1_critical.py and fsss/exp2_bandsteer.py run as two independent jobs,
each writing its own CSV. Neither can see the other's numbers. This script reads
both CSVs and prints one combined table, so all five configs can be compared in
a single view:

    aware    original AWARE, the shared baseline
    exp1a    strip, shrunk by keeping every 10th sample
    exp1b    same, written only during salient moments
    exp2a    strip, moved onto AWARE's window by shift + resample
    exp2b    same, written only during salient moments

It computes nothing new -- it only re-reads and re-groups what the two
experiments already measured. So it is instant, and it can be re-run any number
of times without touching the GPU.

A NOTE ON THE 'aware' COLUMN
----------------------------
Both experiments watermark with plain AWARE as their baseline, using the same
clips, the same seed and the same tolerance. So the two CSVs each contain an
'aware' row set, and they should agree exactly. This script uses experiment 1's
copy and CHECKS that experiment 2's matches; a warning means something differed
between the runs and the comparison should not be trusted.

HOW TO RUN
----------
    conda activate wmcompare
    python -m fsss.exp1_critical        # writes fsss_out/exp1_critical.csv
    python -m fsss.exp2_bandsteer       # writes fsss_out/exp2_bandsteer.csv
    python -m fsss.compare_exp1_exp2    # this script

Writes fsss_out/compare_exp1_exp2.png and prints the table.
"""

import os
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save figures to disk; no display on a compute node
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)

OUT_DIR = os.path.join(BASE, "fsss_out")
EXP1_CSV = os.path.join(OUT_DIR, "exp1_critical.csv")
EXP2_CSV = os.path.join(OUT_DIR, "exp2_bandsteer.csv")

# The order columns appear in. Baseline first, then the two experiments.
CONFIG_ORDER = ["aware", "exp1a", "exp1b", "exp2a", "exp2b"]

DETECT_THRESHOLD = 0.5


# =========================================================================== #
#  Reading the CSVs
# =========================================================================== #

def read_csv(path):
    """{(config, attack): [(ber, conf, pesq), ...]} plus the attack order.

    Attack order is taken from the file rather than hard-coded, so the table
    rows always match whatever the experiments actually ran.
    """
    if not os.path.exists(path):
        print(f"missing: {path}")
        print("  run the matching experiment first")
        return None, []

    rows = defaultdict(list)
    attacks = []
    with open(path) as handle:
        for record in csv.DictReader(handle):
            config = record["config"]
            attack = record["attack"]
            if attack not in attacks:
                attacks.append(attack)

            def number(field):
                text = record.get(field, "")
                try:
                    return float(text)
                except (TypeError, ValueError):
                    return float("nan")

            rows[(config, attack)].append(
                (number("ber_percent"), number("confidence"), number("pesq")))
    return rows, attacks


def merge(exp1_rows, exp2_rows):
    """Combine both experiments, keeping experiment 1's copy of 'aware'.

    Returns (merged rows, warning text or None).
    """
    merged = dict(exp1_rows)
    warning = None

    for (config, attack), values in exp2_rows.items():
        if config == "aware":
            continue                       # keep exp1's copy; checked below
        merged[(config, attack)] = values

    # Consistency check: the shared baseline must agree between the two runs.
    for (config, attack), values in exp2_rows.items():
        if config != "aware":
            continue
        other = exp1_rows.get(("aware", attack))
        if other is None:
            continue
        a = np.nanmean([v[0] for v in values])
        b = np.nanmean([v[0] for v in other])
        if abs(a - b) > 0.01:
            warning = (f"'aware' differs between the two runs "
                       f"(BER {b:.2f} in exp1 vs {a:.2f} in exp2 on '{attack}'). "
                       f"The runs did not use identical settings.")
            break
    return merged, warning


def average(rows, config, attack):
    """(mean BER, mean confidence, mean PESQ, detected, tried)."""
    values = rows.get((config, attack), [])
    if not values:
        return float("nan"), float("nan"), float("nan"), 0, 0
    bers = [v[0] for v in values if not np.isnan(v[0])]
    confs = [v[1] for v in values if not np.isnan(v[1])]
    pesqs = [v[2] for v in values if not np.isnan(v[2])]
    detected = sum(1 for v in values if v[1] >= DETECT_THRESHOLD)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return mean(bers), mean(confs), mean(pesqs), detected, len(values)


# =========================================================================== #
#  The combined table
# =========================================================================== #

def print_combined_table(rows, attacks, configs):
    width = 22 + 22 * len(configs)
    print("\n" + "=" * width)
    print("COMBINED TABLE -- experiment 1 and experiment 2 side by side")
    print("  BER%  percent of bits wrong. Lower is better, 0 = perfect.")
    print("  conf  detector confidence 0-1. Higher is better, >= 0.5 = detected.")
    print("  PESQ  quality of the file vs the clean original. Higher is better.")
    print("        On attack rows the attack dominates PESQ, so compare configs")
    print("        ACROSS a row rather than comparing rows to each other.")
    print("=" * width)
    print(f"  {'attack':20s}" + "".join(f"{c:>22s}" for c in configs))
    print(f"  {'':20s}" + "".join(f"{'BER%':>7s}{'conf':>7s}{'PESQ':>8s}"
                                  for _ in configs))
    print("  " + "-" * (width - 2))
    for attack in attacks:
        line = f"  {attack:20s}"
        for config in configs:
            ber, conf, pesq, _, tried = average(rows, config, attack)
            if tried == 0:
                line += "-".rjust(22)
            else:
                pesq_text = "   -" if np.isnan(pesq) else f"{pesq:8.2f}"
                line += f"{ber:>7.2f}{conf:>7.3f}{pesq_text:>8s}"
        print(line)
    print("=" * width)

    # detection rate, same layout
    print("\n" + "=" * width)
    print("DETECTION RATE -- files where confidence >= 0.5")
    print("=" * width)
    print(f"  {'attack':20s}" + "".join(f"{c:>22s}" for c in configs))
    print("  " + "-" * (width - 2))
    for attack in attacks:
        line = f"  {attack:20s}"
        for config in configs:
            _, _, _, detected, tried = average(rows, config, attack)
            line += "-".rjust(22) if tried == 0 else f"{f'{detected}/{tried}':>22s}"
        print(line)
    print("=" * width)


def print_summary(rows, attacks, configs):
    """One line per config, averaged over every attack. The 'at a glance' view."""
    attacks_only = [a for a in attacks if a != "clean"]
    print("\n" + "=" * 74)
    print("SUMMARY -- averaged over every attack")
    print("=" * 74)
    print(f"  {'config':10s}{'clean BER%':>12s}{'attack BER%':>13s}"
          f"{'attack conf':>13s}{'clean PESQ':>12s}{'detected':>12s}")
    print("  " + "-" * 72)
    for config in configs:
        clean_ber, _, clean_pesq, _, clean_tried = average(rows, config, "clean")
        bers, confs, detected, tried = [], [], 0, 0
        for attack in attacks_only:
            b, c, _, d, t = average(rows, config, attack)
            if t:
                bers.append(b)
                confs.append(c)
                detected += d
                tried += t
        if not clean_tried and not tried:
            continue
        rate = f"{detected}/{tried}" if tried else "-"
        print(f"  {config:10s}{clean_ber:>12.2f}{np.mean(bers):>13.2f}"
              f"{np.mean(confs):>13.3f}{clean_pesq:>12.2f}{rate:>12s}")
    print("=" * 74)


# =========================================================================== #
#  Figure
# =========================================================================== #

def figure_combined(rows, attacks, configs):
    """Three stacked panels -- BER, confidence and PESQ -- all five configs."""
    x = np.arange(len(attacks))
    bar_width = 0.8 / len(configs)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    panels = [
        (0, "bit error rate (lower is better)", "BER (%)", None),
        (1, "detector confidence (higher is better)", "confidence", DETECT_THRESHOLD),
        (2, "PESQ (higher is better)", "PESQ", None),
    ]

    for ax, (index, title, ylabel, line) in zip(axes, panels):
        for i, config in enumerate(configs):
            values = [average(rows, config, a)[index] for a in attacks]
            ax.bar(x + i * bar_width - 0.4 + bar_width / 2, values,
                   bar_width, label=config)
        if line is not None:
            ax.axhline(line, color="0.35", ls="--", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(fontsize=9, ncol=len(configs))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(attacks, rotation=35, ha="right", fontsize=9)
    fig.suptitle("Experiment 1 and Experiment 2 compared against original AWARE",
                 fontsize=12)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "compare_exp1_exp2.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {path}")


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    print("=" * 74)
    print("COMBINED REPORT -- experiment 1 vs experiment 2 vs original AWARE")
    print("=" * 74)

    exp1_rows, exp1_attacks = read_csv(EXP1_CSV)
    exp2_rows, exp2_attacks = read_csv(EXP2_CSV)
    if exp1_rows is None and exp2_rows is None:
        return

    rows = {}
    attacks = []
    warning = None
    if exp1_rows is not None and exp2_rows is not None:
        rows, warning = merge(exp1_rows, exp2_rows)
        attacks = exp1_attacks + [a for a in exp2_attacks if a not in exp1_attacks]
        print(f"read both CSVs from {OUT_DIR}")
    else:
        # one experiment is missing -- still print what we have, clearly labelled
        rows = exp1_rows if exp1_rows is not None else exp2_rows
        attacks = exp1_attacks if exp1_rows is not None else exp2_attacks
        print("WARNING: only one experiment's CSV was found; the table is partial.")

    if warning:
        print(f"\nWARNING: {warning}")

    # only show configs that actually have data
    configs = [c for c in CONFIG_ORDER
               if any(key[0] == c for key in rows)]
    print(f"configs      : {', '.join(configs)}")
    print(f"attacks      : {len(attacks)} rows")

    print_combined_table(rows, attacks, configs)
    print_summary(rows, attacks, configs)
    figure_combined(rows, attacks, configs)

    print("\nDone.")


if __name__ == "__main__":
    main()
