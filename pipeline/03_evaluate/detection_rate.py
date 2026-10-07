"""
Pathway-level detection rate under Benjamini-Hochberg (BH/FDR) correction across
pathways, replacing the flat 0.05 cutoff on the existing per-pathway p-value
(adj_p = min_over_latents(p) * dim, i.e. Bonferroni-corrected for having taken the
best of D latent directions) with a BH q-value computed within each
(dataset, db, method, dim, readout) batch -- i.e. within one actual ORA run against
one db's own pathway pool at one model dimensionality. adj_p (not raw min_p) is fed
into BH so the existing D-latent multiplicity correction is preserved and BH only
adds the previously-missing across-pathway correction. hallmark/kegg are never
pooled into the same BH batch, matching how this pipeline actually ran ORA
(run_readouts.py tests each db against its own pathway pool separately).

Writes:
  detection_by_method_readout_BH.csv  (detection rate over dimensionalities)
  detection_by_dim_BH.csv              (detection rate over gene sets, per dimensionality)
  dim_robustness_BH.csv                (step-to-step retention/kappa across dimensionalities)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import time
import numpy as np
import sys
import pandas as pd

# ORA background-universe mode ("all" default / "annotated"): picks which
# readout ORA / detection variant to read and how to name outputs.
from latent2genesets import ora as _ora
from statsmodels.stats.multitest import multipletests

from config import READOUT_DIR, EVALUATION_DIR, SEED_SUFFIX, readout_ora_path
USECOLS = ["method", "dim", "readout", "db", "pathway", "p"]

METHOD_READOUTS = {
    "PCA": ["pearson", "spearman", "weights"],
    "ICA": ["pearson", "spearman", "weights"],
    "NMF": ["pearson", "spearman", "weights"],
    "AE":  ["pearson", "spearman", "weights", "ig", "deeplift", "inputxgrad"],
    "DAE": ["pearson", "spearman", "weights", "ig", "deeplift", "inputxgrad"],
    "VAE": ["pearson", "spearman", "weights", "ig", "deeplift", "inputxgrad"],
}
READOUT_ORDER = ["pearson", "spearman", "weights", "ig", "deeplift", "inputxgrad"]


def cohen_kappa(s_small, s_large, n_pathways):
    a, b = len(s_small), len(s_large)
    both = len(s_small & s_large)
    neither = n_pathways - len(s_small | s_large)
    p_obs = (both + neither) / n_pathways
    pa, pb = a / n_pathways, b / n_pathways
    p_exp = pa * pb + (1 - pa) * (1 - pb)
    if p_exp >= 1.0:
        return None
    return (p_obs - p_exp) / (1 - p_exp)


def bh_transform(s):
    return multipletests(s.values, method="fdr_bh")[1]


by_method_rows, by_dim_rows, robust_rows = [], [], []

for ds in ["TCGA", "GTEX"]:
    t0 = time.time()
    path = readout_ora_path(ds, f"{_ora.OUT_SUFFIX}{SEED_SUFFIX}")
    print(f"[{ds}] loading {path}", flush=True)
    df = pd.read_csv(path, usecols=USECOLS,
                      dtype={"method": "category", "readout": "category",
                             "db": "category", "pathway": "category"})
    print(f"[{ds}]   loaded {df.shape} in {time.time()-t0:.0f}s", flush=True)

    g = df.groupby(["method", "dim", "readout", "db", "pathway"], observed=True)["p"].min().reset_index()
    del df
    g["adj_p"] = (g["p"] * g["dim"]).clip(upper=1.0)  # existing Bonferroni-over-D-latents p
    print(f"[{ds}]   collapsed to {g.shape[0]} (method,dim,readout,db,pathway) rows", flush=True)

    t1 = time.time()
    g["q_bh"] = g.groupby(["method", "dim", "readout", "db"], observed=True)["adj_p"].transform(bh_transform)
    g["detected_bh"] = g["q_bh"] < 0.05
    g["detected_bonf"] = g["adj_p"] < 0.05
    print(f"[{ds}]   BH done in {time.time()-t1:.0f}s -- "
          f"Bonferroni {g.detected_bonf.mean()*100:.1f}% vs BH {g.detected_bh.mean()*100:.1f}% detected overall",
          flush=True)

    # ---- rate over dim, per (db,method,readout,pathway) ----
    rate1 = g.groupby(["method", "readout", "db", "pathway"], observed=True)["detected_bh"].mean().reset_index()
    rate1["detection_rate_pct"] = rate1["detected_bh"] * 100
    rate1["dataset"] = ds
    by_method_rows.append(rate1[["dataset", "db", "method", "readout", "pathway", "detection_rate_pct"]])

    # ---- rate over pathway, per (db,method,dim,readout) ----
    rate2 = g.groupby(["method", "dim", "readout", "db"], observed=True)["detected_bh"].mean().reset_index()
    rate2["detection_rate_pct"] = rate2["detected_bh"] * 100
    rate2["dataset"] = ds
    by_dim_rows.append(rate2[["dataset", "db", "method", "dim", "readout", "detection_rate_pct"]])

    # ---- step-to-step retention / kappa across dims, pooled over db ----
    det = g[g.detected_bh]
    n_pathways = g.pathway.nunique()
    for ro in READOUT_ORDER:
        methods = [m for m, ros in METHOD_READOUTS.items() if ro in ros]
        per_pair, per_pair_k = {}, {}
        for m in methods:
            sub = det[(det.method == m) & (det.readout == ro)]
            dims = sorted(g[(g.method == m) & (g.readout == ro)].dim.unique())
            sets = {D: set(sub[sub.dim == D].pathway) for D in dims}
            for i in range(len(dims) - 1):
                s_small, s_large = sets[dims[i]], sets[dims[i + 1]]
                if len(s_small) == 0:
                    continue
                r = len(s_small & s_large) / len(s_small)
                per_pair.setdefault(dims[i + 1], []).append(r)
                k = cohen_kappa(s_small, s_large, n_pathways)
                if k is not None:
                    per_pair_k.setdefault(dims[i + 1], []).append(k)
        for dim_large, vals in per_pair.items():
            kvals = per_pair_k.get(dim_large, [])
            robust_rows.append({"dataset": ds, "readout": ro, "dim": dim_large,
                               "retention_pct": 100 * np.mean(vals),
                               "kappa": np.mean(kvals) if kvals else np.nan,
                               "n_methods": len(vals)})
    del g, det
    print(f"[{ds}] done in {time.time()-t0:.0f}s total", flush=True)

by_method_out = pd.concat(by_method_rows, ignore_index=True)
by_method_out.to_csv(f"{EVALUATION_DIR}/detection_by_method_readout_BH{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv", index=False)
print("wrote", by_method_out.shape, "-> detection_by_method_readout_BH.csv", flush=True)

by_dim_out = pd.concat(by_dim_rows, ignore_index=True)
by_dim_out.to_csv(f"{EVALUATION_DIR}/detection_by_dim_BH{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv", index=False)
print("wrote", by_dim_out.shape, "-> detection_by_dim_BH.csv", flush=True)

robust_out = pd.DataFrame(robust_rows)
robust_out.to_csv(f"{EVALUATION_DIR}/dim_robustness_BH{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv", index=False)
print("wrote", robust_out.shape, "-> dim_robustness_BH.csv", flush=True)

print("ALL DONE", flush=True)
