"""
auroc_per_dim.py -- per-dimensionality AUROC against the pooled
differential-expression reference.

Reads the per-latent Fisher's exact P-values written by
pipeline/02_readouts/run_readouts.py. For each gene set and dimensionality D, the
minimum P-value over the D latent dimensions is Bonferroni-corrected over D and
scored as -log10(p_Bonf); AUROC of this score against the pooled reference labels
is computed separately for each dataset, database, method, readout and
dimensionality. No significance threshold is applied.

One model seed per run (config.SEED); pipeline/05_replicates averages the seeds.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import READOUT_DIR, EVALUATION_DIR, SEED_SUFFIX, readout_ora_path
from latent2genesets import ora as _ora
from latent2genesets.evaluation import build_pooled_reference

FREQ_THRESH = 0.5
USECOLS = ["dataset", "method", "dim", "latent_idx", "readout", "db", "pathway", "p"]




def process_dataset(ds, pooled_ref):
    t0 = time.time()
    fname = readout_ora_path(ds, f"{_ora.OUT_SUFFIX}{SEED_SUFFIX}")
    print(f"[{ds}] reading {fname.name} ({fname.stat().st_size/1e6:.0f}MB)...", flush=True)
    raw = pd.read_csv(fname, usecols=USECOLS)
    print(f"[{ds}] loaded {raw.shape} in {time.time()-t0:.0f}s", flush=True)

    # min(p) over latent_idx WITHIN each dim -- continuous score, no cutoff.
    best = (raw.groupby(["method", "dim", "readout", "db", "pathway"], observed=True)["p"]
               .min().reset_index())
    best["dataset"] = ds
    print(f"[{ds}] collapsed to {best.shape} (dim, method, readout, db, pathway) rows "
          f"({time.time()-t0:.0f}s)", flush=True)

    merged = best.merge(pooled_ref, on=["dataset", "db", "pathway"], how="inner")
    # Bonferroni across the D latents within this dim: p_bonf = min(p * dim, 1).
    # D is constant across every pathway in a given (dim, method, readout, db)
    # group, so this is a uniform rescaling and barely moves AUROC (verified:
    # 0.6425 raw vs 0.6417 Bonferroni on TCGA/hallmark/PCA/pearson/D=50, the
    # ~0.001 gap coming only from ties introduced where p*dim clips to 1).
    # Applied anyway to match the main pipeline's correction convention.
    merged["p_bonf"] = (merged["p"] * merged["dim"]).clip(upper=1.0)
    merged["score"] = -np.log10(merged["p_bonf"].clip(lower=1e-300))

    rows = []
    for (ds_, db, method, readout, dim), sub in merged.groupby(
            ["dataset", "db", "method", "readout", "dim"], observed=True):
        y = sub["pooled_reference_binary"].values
        if y.sum() == 0 or y.sum() == len(y):
            auc = np.nan
        else:
            auc = roc_auc_score(y, sub["score"].values)
        rows.append({"dataset": ds_, "db": db, "method": method, "readout": readout,
                      "dim": dim, "n_pathways": len(sub), "n_positive": int(y.sum()), "auroc": auc})
    out = pd.DataFrame(rows)
    print(f"[{ds}] done in {time.time()-t0:.0f}s, {out.shape}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    suffix = _ora.OUT_SUFFIX + SEED_SUFFIX
    pooled_ref = build_pooled_reference(f"final_reference{_ora.OUT_SUFFIX}.csv")
    frames = [process_dataset(ds, pooled_ref) for ds in ["TCGA", "GTEX"]]
    cell_df = pd.concat(frames, ignore_index=True)
    cell_path = EVALUATION_DIR / f"per_dim_auroc_cells{suffix}.csv"
    cell_df.to_csv(cell_path, index=False)
    print(f"wrote {cell_df.shape} -> {cell_path}")


if __name__ == "__main__":
    main()
