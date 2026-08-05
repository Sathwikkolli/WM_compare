# Results Archive

Permanent home for **every** experiment run in this project. Nothing here is
"regenerated on the cluster" — if it isn't in this folder, it didn't happen.

## Why this exists

Before 2026-08-05 this repo treated results as disposable: `.gitignore` excluded
`*_results.csv` and `*.png` with the note "regenerated on the cluster." That
means every number ever produced was lost the moment the scratch directory was
cleaned, and no result could be cited later without re-running the job.

From now on results are **artifacts, not by-products**.

## Layout

```
results/
    README.md                     <- this file
    INDEX.md                      <- running table of every experiment; update it
    YYYY-MM-DD_<slug>/
        README.md                 <- what, why, how; decisions; conclusions
        params.json               <- exact config needed to reproduce
        data/                     <- raw numbers (CSV/NPZ). Never overwrite.
        figures/                  <- PNG/SVG visualizations
        summary.md                <- the headline numbers + what they mean
```

One directory per experiment run. Date-prefixed so ordering is chronological.
If an experiment is re-run with changed parameters, that is a **new directory**,
not an edit to the old one.

## Rules

1. **Never overwrite `data/`.** A re-run gets a new dated directory. Old numbers
   stay citable.
2. **`summary.md` must state the conclusion in words**, not just the table. Six
   months from now the table alone will not be readable.
3. **`params.json` must be sufficient to reproduce.** Include: git commit,
   random seeds, clip list, attack grid, package versions, Slurm job ID.
4. **Every directory gets a row in `INDEX.md`** with date, slug, one-line
   finding, and status.
5. **Record negative and inconclusive results too.** A method that failed is
   information; deleting it means someone re-tries it in three months.
6. **This folder is NOT gitignored.** CSVs and figures here are committed. If
   raw audio is genuinely too large, store a manifest + checksums here and the
   audio in `/scratch`, and say so in the run's README.

## Naming

`YYYY-MM-DD_<short-slug>` — e.g. `2026-08-05_align-method-bakeoff`.
Lowercase, hyphens, no spaces.
