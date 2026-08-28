"""
informed/score_screen.py -- which attacks break detection while the audio survives?

Reads results/<run>/data/screen_clip*.csv. Writes summary_screen.md,
data/screen_metrics.csv, and **selected_attacks.json** -- the Phase B input.

THE CLASSIFICATION

For each attack, walk its strengths from weakest to strongest and find s*, the
strongest setting whose mean quality is still at or above the usability floor.
Then ask what detection is doing at s*:

  VULNERABLE     detection is already below threshold at s*.
                 Usable audio, no detection. This is a real weakness, and the
                 attacker gets it for free.
  SECURE         detection still holds at s*. To beat the watermark the attacker
                 must push past the floor and wreck the audio -- which defeats
                 their purpose. This is what 2026-08-17_attack-damage-control
                 found for high-pass and quantisation.
  NO_FLOOR       quality never fell below the floor anywhere in the swept range.
                 If detection also never failed, SECURE across the whole range;
                 if detection failed, VULNERABLE and the sweep did not go far
                 enough to find the quality limit.
  UNAVAILABLE    the attack could not run here. Not a result either way.

`margin_db` is not available in general (strengths are not comparable across
attacks), so the width of the weakness is reported as `n_vulnerable_configs`:
how many swept settings sit in the usable-but-undetected quadrant.

Only VULNERABLE attacks go into selected_attacks.json. Attacks that break
alignment are flagged there but excluded -- Phase B cannot subtract what it
cannot align.

Usage:
    python score_screen.py
    python score_screen.py --floor 3.5        # strict usability
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.environ.get("WM_COMPARE_BASE", ROOT)
sys.path.insert(0, HERE)

RUN_SLUG = "2026-08-28_informed-detection"
RESULTS_DIR = os.path.join(BASE, "results", RUN_SLUG)
DATA_DIR = os.path.join(RESULTS_DIR, "data")

DETECT_THRESHOLD = 0.50
DNSMOS_FLOOR = 3.0
BITACC_USEFUL = 0.85      # below this, informed detection has little to recover


def get_arg(argv, flag, default, cast=str):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "screen_clip*.csv")))
    if not files:
        raise SystemExit(f"no screen_clip*.csv in {DATA_DIR} -- run screen_sweep.py")
    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("conf", "bit_acc", "dnsmos_ovrl", "pesq", "runtime_s"):
                    r[k] = fnum(r.get(k))
                r["ok"] = r.get("ok") == "1"
                r["alignment_breaking"] = r.get("alignment_breaking") == "1"
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} file(s)")
    return rows


def mean(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def fmt(v, nd=2, dash="-"):
    return f"{v:.{nd}f}" if np.isfinite(v) else dash


def main(argv):
    floor = get_arg(argv, "--floor", DNSMOS_FLOOR, float)
    rows = load()

    wm = [r for r in rows if r["arm"] == "wm"]
    un = [r for r in rows if r["arm"] == "unwm"]
    attacks = sorted({r["attack"] for r in rows})
    clips = sorted({r["clip_id"] for r in rows})
    print(f"  {len(attacks)} attacks, {len(clips)} clips")

    # ---- collapse to (attack, param) means across clips --------------------
    cfg = defaultdict(dict)
    order = defaultdict(list)
    for r in wm:
        key = (r["attack"], r["param"])
        cfg[key].setdefault("conf", []).append(r["conf"])
        cfg[key].setdefault("dns", []).append(r["dnsmos_ovrl"])
        cfg[key].setdefault("pesq", []).append(r["pesq"])
        cfg[key].setdefault("bacc", []).append(r["bit_acc"])
        cfg[key].setdefault("ok", []).append(r["ok"])
        cfg[key]["category"] = r["category"]
        cfg[key]["breaks_align"] = r["alignment_breaking"]
        if r["param"] not in order[r["attack"]]:
            order[r["attack"]].append(r["param"])

    per_cfg, per_attack = [], []
    for atk in attacks:
        params = order[atk]
        recs = []
        for p in params:
            c = cfg[(atk, p)]
            recs.append({
                "attack": atk, "param": p, "category": c.get("category", "?"),
                "breaks_align": bool(c.get("breaks_align")),
                "conf": mean(c["conf"]), "dnsmos": mean(c["dns"]),
                "pesq": mean(c["pesq"]), "bit_acc": mean(c["bacc"]),
                "n_ok": sum(1 for v in c["ok"] if v), "n": len(c["ok"]),
            })
        per_cfg.extend(recs)

        runnable = [r for r in recs if r["n_ok"] > 0 and np.isfinite(r["dnsmos"])]
        if not runnable:
            per_attack.append({"attack": atk, "category": recs[0]["category"],
                               "verdict": "UNAVAILABLE", "n_vulnerable": 0,
                               "breaks_align": recs[0]["breaks_align"],
                               "s_star": "", "conf_at_s_star": float("nan"),
                               "dnsmos_at_s_star": float("nan"),
                               "bit_acc_at_s_star": float("nan")})
            continue

        usable = [r for r in runnable if r["dnsmos"] >= floor]
        vulnerable = [r for r in runnable
                      if r["dnsmos"] >= floor and np.isfinite(r["conf"])
                      and r["conf"] < DETECT_THRESHOLD]

        if not usable:
            verdict, s = "SECURE", None      # every setting already wrecks audio
        else:
            # s* = the usable setting with the LOWEST confidence: the best an
            # attacker can do without pushing past the floor.
            s = min(usable, key=lambda r: r["conf"] if np.isfinite(r["conf"]) else 9e9)
            all_usable = len(usable) == len(runnable)
            if vulnerable:
                verdict = "VULNERABLE"
            elif all_usable:
                verdict = "NO_FLOOR"         # never got the audio bad enough
            else:
                verdict = "SECURE"

        per_attack.append({
            "attack": atk, "category": recs[0]["category"], "verdict": verdict,
            "n_vulnerable": len(vulnerable), "n_configs": len(runnable),
            "breaks_align": recs[0]["breaks_align"],
            "s_star": s["param"] if s else "",
            "conf_at_s_star": s["conf"] if s else float("nan"),
            "dnsmos_at_s_star": s["dnsmos"] if s else float("nan"),
            "bit_acc_at_s_star": s["bit_acc"] if s else float("nan"),
        })

    L = []
    w = L.append
    w("# Phase A — the 27-attack boundary screen\n")
    w(f"Run `{RUN_SLUG}`. {len(rows)} rows, {len(attacks)} attacks, "
      f"{len(clips)} clips. Detection threshold **{DETECT_THRESHOLD}**, "
      f"usability floor **DNSMOS {floor}**.\n")
    w("Classification is documented at the top of `score_screen.py`. "
      "`s*` is the strongest setting whose audio is still usable.\n")

    # ---- headline ---------------------------------------------------------
    order_v = {"VULNERABLE": 0, "NO_FLOOR": 1, "SECURE": 2, "UNAVAILABLE": 3}
    per_attack.sort(key=lambda r: (order_v.get(r["verdict"], 9),
                                   -r["n_vulnerable"], r["attack"]))

    w("\n## 1. Verdict per attack  <- THE SCREEN\n")
    w("| attack | category | verdict | vulnerable configs | s* | conf @ s* | DNSMOS @ s* | bit acc @ s* |")
    w("|---|---|---|---|---|---|---|---|")
    for r in per_attack:
        flag = " ⚠align" if r["breaks_align"] else ""
        nv = f"{r['n_vulnerable']}/{r.get('n_configs', 0)}"
        w(f"| `{r['attack']}`{flag} | {r['category']} | **{r['verdict']}** | {nv} "
          f"| {r['s_star']} | {fmt(r['conf_at_s_star'])} "
          f"| {fmt(r['dnsmos_at_s_star'])} | {fmt(r['bit_acc_at_s_star'], 3)} |")
    w("\n⚠align = the bake-off found no aligner handles this attack, so Phase B "
      "cannot subtract an original even where the screen flags a weakness.")

    counts = defaultdict(int)
    for r in per_attack:
        counts[r["verdict"]] += 1
    w(f"\n**{counts['VULNERABLE']} vulnerable, {counts['SECURE']} secure, "
      f"{counts['NO_FLOOR']} never reached the quality floor, "
      f"{counts['UNAVAILABLE']} unavailable.**")

    # ---- is there anything for Phase B to fix? ----------------------------
    w("\n## 2. Is the watermark still readable where detection fails?\n")
    w("Informed detection recovers a *masked* watermark. If the bits are already "
      f"gone at s*, there is nothing to recover. Threshold: **{BITACC_USEFUL}**.\n")
    w("| attack | bit acc @ s* | recoverable? |")
    w("|---|---|---|")
    for r in per_attack:
        if r["verdict"] != "VULNERABLE":
            continue
        b = r["bit_acc_at_s_star"]
        verdict = ("yes" if np.isfinite(b) and b >= BITACC_USEFUL else
                   "unlikely — bits already lost" if np.isfinite(b) else "unknown")
        w(f"| `{r['attack']}` | {fmt(b, 3)} | {verdict} |")

    # ---- by category ------------------------------------------------------
    w("\n## 3. By category\n")
    w("Additive attacks should respond to informed detection (the watermark is "
      "masked); codec and filter attacks should not (rebuilt or deleted).\n")
    w("| category | attacks | vulnerable |")
    w("|---|---|---|")
    bycat = defaultdict(lambda: [0, 0])
    for r in per_attack:
        bycat[r["category"]][0] += 1
        bycat[r["category"]][1] += 1 if r["verdict"] == "VULNERABLE" else 0
    for cat in sorted(bycat):
        n, v = bycat[cat]
        w(f"| {cat} | {n} | {v} |")

    # ---- arm control ------------------------------------------------------
    w("\n## 4. Did the watermark itself cost quality?\n")
    gaps = []
    byk = defaultdict(dict)
    for r in wm:
        byk[(r["clip_id"], r["attack"], r["param"])]["wm"] = r["dnsmos_ovrl"]
    for r in un:
        byk[(r["clip_id"], r["attack"], r["param"])]["unwm"] = r["dnsmos_ovrl"]
    for v in byk.values():
        if "wm" in v and "unwm" in v and np.isfinite(v["wm"]) and np.isfinite(v["unwm"]):
            gaps.append(v["unwm"] - v["wm"])
    if gaps:
        a = np.array(gaps)
        w(f"Mean DNSMOS gap (unwatermarked − watermarked) over {len(gaps)} matched "
          f"conditions: **{a.mean():+.3f}** (sd {a.std(ddof=1):.3f}, "
          f"max |gap| {np.abs(a).max():.3f}).")
        w("\n`2026-08-17_attack-damage-control` measured 0.06 PESQ for the same "
          "question. A small gap here means the attacks do the damage and the "
          "control arm can be dropped from Phase B.")
    else:
        w("No matched pairs — the control arm did not run.")

    # ---- integrity --------------------------------------------------------
    w("\n## Data integrity\n")
    n_skip = sum(1 for r in rows if not r["ok"])
    w(f"- rows where the attack could not run: **{n_skip}** "
      f"({100.0*n_skip/max(1,len(rows)):.1f}%) — recorded as skips, never as "
      f"watermark results")
    if n_skip:
        by_atk = defaultdict(int)
        for r in rows:
            if not r["ok"]:
                by_atk[r["attack"]] += 1
        worst = sorted(by_atk.items(), key=lambda kv: -kv[1])[:6]
        w("  - " + ", ".join(f"`{k}` {v}" for k, v in worst))
    n_noq = sum(1 for r in wm if not np.isfinite(r["dnsmos_ovrl"]) and r["ok"])
    w(f"- runnable rows with no DNSMOS: **{n_noq}** — if not ~0 the quality axis "
      f"is incomplete and section 1 is unreliable")

    w("\n## Conclusion\n")
    w("*(write this by hand after reading the tables — results/README.md rule 2)*")

    # ---- outputs ----------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    if per_cfg:
        cols = sorted({k for d in per_cfg for k in d})
        with open(os.path.join(DATA_DIR, "screen_metrics.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(per_cfg)

    # THE PHASE B INPUT. Data, not code: Phase A chooses the attacks by
    # evidence, and Phase B reads that choice rather than hardcoding one.
    selected = [r for r in per_attack
                if r["verdict"] == "VULNERABLE" and not r["breaks_align"]]
    sel = {
        "run": RUN_SLUG,
        "floor": floor,
        "detect_threshold": DETECT_THRESHOLD,
        "selected": [{"attack": r["attack"], "param": r["s_star"],
                      "category": r["category"],
                      "conf_at_s_star": r["conf_at_s_star"],
                      "dnsmos_at_s_star": r["dnsmos_at_s_star"],
                      "bit_acc_at_s_star": r["bit_acc_at_s_star"],
                      "recoverable": bool(np.isfinite(r["bit_acc_at_s_star"])
                                          and r["bit_acc_at_s_star"] >= BITACC_USEFUL)}
                     for r in selected],
        "excluded_alignment_breaking": [r["attack"] for r in per_attack
                                        if r["verdict"] == "VULNERABLE"
                                        and r["breaks_align"]],
        "note": "Phase B input. Only attacks that break detection while the "
                "audio stays usable, and that an aligner can handle.",
    }
    with open(os.path.join(HERE, "selected_attacks.json"), "w") as f:
        json.dump(sel, f, indent=2)

    txt = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary_screen.md"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {os.path.join(RESULTS_DIR, 'summary_screen.md')}")
    print(f"wrote {os.path.join(DATA_DIR, 'screen_metrics.csv')}")
    print(f"wrote {os.path.join(HERE, 'selected_attacks.json')}  "
          f"({len(sel['selected'])} attacks for Phase B)")
    if not sel["selected"]:
        print("\nNO VULNERABLE ATTACKS. If this holds, AWARE outlives usability "
              "across the whole screen and informed detection has nothing to "
              "fix — a real result about the watermark, and the point at which "
              "the parent plan should stop.")


if __name__ == "__main__":
    main(sys.argv[1:])
