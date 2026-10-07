"""
summarize_replicates.py -- mean and SD over the model-seed replicates and over the
permutation replicates.

Two kinds of replicate, kept apart (see config.py):

  model seeds   config.SEEDS (42, 43, 44). Same data; the train/validation split, the
                ICA/NMF initialisation and subsample, and the neural-network
                initialisation and batch order change. Run every seed-dependent step
                once per seed with LATENT2GENESETS_SEED=<n>; seed n != 42 writes
                *_seed<n> files.
  permutations  controls/permutation_null.py --perm p, p = 0 .. N_PERMUTATIONS - 1.
                The data are re-shuffled for each p; the model seed stays 42.

Every replicate is first reduced exactly as a single run is reported (mean AUROC over
the latent dimensionalities per dataset x database x model x readout), then the mean
and the SD (ddof = 1) are taken across replicates.

ICA is skipped at a dimensionality where it does not converge (pipeline/02_readouts),
so an ICA cell can be absent from some replicate. Such a cell is averaged over the
replicates that have it and its count is written out (n_seeds / n_dims); a cell of any
other model missing from a replicate is an error.

Reads   evaluation/per_dim_auroc_cells<suffix>[_seed<n>].csv        (03_evaluate/auroc_per_dim.py)
        evaluation/auroc_recurrence_cells<suffix>[_seed<n>].csv     (03_evaluate/auroc_recurrence.py)
        phenotype/auroc_<group>_<label>[_seed<n>].csv               (04_phenotype/build_auroc.py)
        controls/permutation_null_cells[_perm<p>].csv               (controls/permutation_null.py --summarize)
Writes  evaluation/auroc_seed_summary<suffix>.csv                   mean, SD, per-seed values
        evaluation/per_dim_auroc_cells<suffix>_seedmean.csv         per-dimension mean over seeds
        evaluation/auroc_recurrence_seed_summary<suffix>.csv        mean, SD per recurrence threshold k
        phenotype/auroc_seed_summary.csv                            per case study x model x readout
        controls/permutation_null_summary.csv                       mean, SD over permutations

Usage:
    python pipeline/05_replicates/summarize_replicates.py
    python pipeline/05_replicates/summarize_replicates.py --seeds 42,43,44 --perms 0-9
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse

import pandas as pd

from config import (SEEDS, N_PERMUTATIONS, RECURRENCE_K, seed_tag, perm_tag,
                    EVALUATION_DIR, PHENOTYPE_DIR, CONTROLS_DIR)
from latent2genesets import ora as _ora

KEY = ["dataset", "db", "method", "readout"]


def parse_ints(spec):
    """'42,43,44' or '0-9' -> sorted list of ints"""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(x) for x in spec.split(",")})


def load_replicates(paths):
    """{replicate id: path} -> {replicate id: DataFrame}; every file must exist."""
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("replicate files not found:\n  " + "\n  ".join(missing))
    return {r: pd.read_csv(p) for r, p in paths.items()}


def across(per_rep, label):
    """{replicate id: Series} -> DataFrame with one column per replicate plus mean, sd, n.
    Only ICA rows may be short of a replicate."""
    t = pd.concat({f"{label}{r}": s for r, s in per_rep.items()}, axis=1)
    cols = list(t.columns)
    n = t[cols].notna().sum(axis=1)
    short = t[n < len(cols)]
    if len(short):
        not_ica = short.index.get_level_values("method") != "ICA"
        if not_ica.any():
            raise ValueError(f"non-ICA cells missing from a replicate:\n{short[not_ica].head()}")
        print(f"  note: {len(short)} ICA cells present in fewer than {len(cols)} replicates (ICA did not converge)")
    t["mean"], t["sd"], t[f"n_{label}s"] = t[cols].mean(axis=1), t[cols].std(axis=1, ddof=1), n
    return t


def auroc_seeds(seeds, stem):
    """Pan-dataset AUROC: per-seed mean over dims, then mean / SD over seeds."""
    cells = load_replicates({s: EVALUATION_DIR / f"{stem}{seed_tag(s)}.csv" for s in seeds})
    out = across({s: c.groupby(KEY).auroc.mean() for s, c in cells.items()}, "seed")
    out.reset_index().to_csv(EVALUATION_DIR / f"auroc_seed_summary{stem[len('per_dim_auroc_cells'):]}.csv", index=False)
    # per-dimension mean over seeds, for the dimensionality-trend plots
    per_dim = across({s: c.set_index(KEY + ["dim"]).auroc for s, c in cells.items()}, "seed")
    per_dim = per_dim.rename(columns={"mean": "auroc"})[["auroc", "sd", "n_seeds"]]
    per_dim.reset_index().to_csv(EVALUATION_DIR / f"{stem}_seedmean.csv", index=False)
    print(f"\n===== AUROC, mean ± SD over seeds {seeds} (mean over dims) =====")
    print(out[["mean", "sd", "n_seeds"]].round(3).to_string())
    return out


def recurrence_seeds(seeds, suffix):
    """k-of-N sensitivity: for each recurrence threshold k, per-seed mean over dims, then
    mean / SD over seeds. Nonlinear-model weights are computed but not reported in the paper."""
    paths = {s: EVALUATION_DIR / f"auroc_recurrence_cells{suffix}{seed_tag(s)}.csv" for s in seeds}
    if not paths[seeds[0]].exists():
        print("\n[skip] no evaluation/auroc_recurrence_cells*.csv -- run pipeline/03_evaluate/auroc_recurrence.py first")
        return
    cells = load_replicates(paths)
    parts = []
    for k in RECURRENCE_K:
        t = across({s: c.groupby(KEY)[f"auroc_k{k}"].mean() for s, c in cells.items()}, "seed")
        npos = {s: c.groupby(KEY)[f"n_positive_k{k}"].first() for s, c in cells.items()}
        first = npos[seeds[0]]
        for s in seeds[1:]:  # the reference does not depend on the seed
            both = first.index.intersection(npos[s].index)
            assert (first[both] == npos[s][both]).all(), f"k={k}: positives differ between seeds {seeds[0]} and {s}"
        parts.append(pd.DataFrame({f"mean_k{k}": t["mean"], f"sd_k{k}": t["sd"],
                                   f"n_positive_k{k}": first.reindex(t.index)}))
    out = pd.concat(parts, axis=1)
    out.reset_index().to_csv(EVALUATION_DIR / f"auroc_recurrence_seed_summary{suffix}.csv", index=False)
    print(f"\n===== AUROC by reference recurrence threshold k, mean ± SD over seeds {seeds} =====")
    shown = pd.DataFrame({f"k>={k}": out[f"mean_k{k}"].map("{:.3f}".format) + " ± " + out[f"sd_k{k}"].map("{:.3f}".format)
                          for k in RECURRENCE_K})
    print(shown.to_string())


def phenotype_seeds(seeds):
    """Case studies: per-seed mean over dims and databases per model x readout."""
    suf = _ora.OUT_SUFFIX
    jobs = sorted(p.name[len("auroc_"):-len(f"{suf}.csv")] for p in PHENOTYPE_DIR.glob(f"auroc_*{suf}.csv")
                  if "_seed" not in p.name and "summary" not in p.name and (suf or "_annotU" not in p.name))
    if not jobs:
        print("\n[skip] no phenotype/auroc_<group>_<label>.csv -- run pipeline/04_phenotype first")
        return
    frames = []
    for job in jobs:
        cells = load_replicates({s: PHENOTYPE_DIR / f"auroc_{job}{suf}{seed_tag(s)}.csv" for s in seeds})
        t = across({s: c.groupby(["method", "readout"]).auroc.mean() for s, c in cells.items()}, "seed")
        frames.append(t.reset_index().assign(job=job))
    out = pd.concat(frames, ignore_index=True)
    out = out[["job"] + [c for c in out.columns if c != "job"]]
    out.to_csv(PHENOTYPE_DIR / f"auroc_seed_summary{suf}.csv", index=False)
    print(f"\n===== case studies: wrote {len(jobs)} jobs x {out.groupby('job').size().iloc[0]} model-readout "
          f"combinations to auroc_seed_summary{suf}.csv =====")


def permutation_null(perms):
    """Permutation null: per-permutation mean over dims, then mean / SD over permutations."""
    paths = {p: CONTROLS_DIR / f"permutation_null_cells{perm_tag(p)}.csv" for p in perms}
    if not paths[perms[0]].exists():
        print("\n[skip] no controls/permutation_null_cells.csv -- run controls/permutation_null.py first")
        return
    cells = load_replicates(paths)
    out = across({p: c.groupby(KEY).auroc.mean() for p, c in cells.items()}, "perm")
    # how many dimensionalities produced a model in each permutation (ICA often fails on shuffled data)
    dims = pd.concat({p: c.groupby(KEY).dim.nunique() for p, c in cells.items()}, axis=1)
    out["min_dims"], out["max_dims"] = dims.min(axis=1), dims.max(axis=1)
    out.reset_index().to_csv(CONTROLS_DIR / "permutation_null_summary.csv", index=False)
    print(f"\n===== permutation null, mean ± SD over permutations {perms} (mean over dims) =====")
    print(out[["mean", "sd", "n_perms", "min_dims", "max_dims"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="model seeds, e.g. 42,43,44")
    ap.add_argument("--perms", default=f"0-{N_PERMUTATIONS - 1}", help="permutation replicates, e.g. 0-9")
    args = ap.parse_args()
    seeds, perms = parse_ints(args.seeds), parse_ints(args.perms)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)

    suffix = _ora.OUT_SUFFIX
    auroc_seeds(seeds, f"per_dim_auroc_cells{suffix}")
    recurrence_seeds(seeds, suffix)
    phenotype_seeds(seeds)
    permutation_null(perms)
