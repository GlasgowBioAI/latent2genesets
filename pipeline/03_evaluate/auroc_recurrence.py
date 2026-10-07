"""
auroc_recurrence.py -- sensitivity of the AUROC to the reference recurrence
threshold (k-of-N).

The primary reference (auroc_per_dim.py) calls a gene set positive if it is differentially
expressed in at least ONE cancer type / tissue, which makes most Hallmark gene sets
positive. Here the same scores are evaluated against stricter references: positive if
differentially expressed in at least k groups, k in config.RECURRENCE_K (1, 2, 3, 5).
k = 1 reproduces auroc_per_dim.py exactly.

Nothing else changes: per (dataset, db, model, readout, dim) the score of a gene set is
-log10(min(dim * min p over latents, 1)), from the per-latent ORA p-values written by
pipeline/02_readouts/run_readouts.py. No model is fit here.

Output: results/evaluation/auroc_recurrence_cells<suffix>[_seed<n>].csv
    columns: dataset, db, method, readout, dim, auroc_k<k>, n_positive_k<k> for each k

Usage:
    python pipeline/03_evaluate/auroc_recurrence.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import READOUT_DIR, EVALUATION_DIR, SEED_SUFFIX, RECURRENCE_K, readout_ora_path
from latent2genesets import ora as _ora
from latent2genesets.evaluation import build_recurrence_reference

USECOLS = ["method", "dim", "readout", "db", "pathway", "p"]


def process_dataset(ds, recurrence):
    t0 = time.time()
    fname = readout_ora_path(ds, f"{_ora.OUT_SUFFIX}{SEED_SUFFIX}")
    print(f"[{ds}] reading {fname.name} ({fname.stat().st_size/1e6:.0f}MB)...", flush=True)
    raw = pd.read_csv(fname, usecols=USECOLS)
    # min(p) over the latents within each dim, as in auroc_per_dim.py
    best = (raw.groupby(["method", "dim", "readout", "db", "pathway"], observed=True)["p"]
               .min().reset_index())
    best["dataset"] = ds
    merged = best.merge(recurrence, on=["dataset", "db", "pathway"], how="inner")
    merged["score"] = -np.log10((merged["p"] * merged["dim"]).clip(upper=1.0).clip(lower=1e-300))

    rows = []
    for key, sub in merged.groupby(["dataset", "db", "method", "readout", "dim"], observed=True):
        row = dict(zip(["dataset", "db", "method", "readout", "dim"], key))
        for k in RECURRENCE_K:
            y = (sub["n_groups_detected"].values >= k).astype(int)
            row[f"auroc_k{k}"] = roc_auc_score(y, sub["score"].values) if 0 < y.sum() < len(y) else np.nan
            row[f"n_positive_k{k}"] = int(y.sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"[{ds}] done in {time.time()-t0:.0f}s, {out.shape}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    recurrence = build_recurrence_reference(f"final_reference{_ora.OUT_SUFFIX}.csv")
    cells = pd.concat([process_dataset(ds, recurrence) for ds in ["TCGA", "GTEX"]], ignore_index=True)
    path = EVALUATION_DIR / f"auroc_recurrence_cells{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv"
    cells.to_csv(path, index=False)
    print(f"wrote {cells.shape} -> {path}")


if __name__ == "__main__":
    main()
