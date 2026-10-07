"""
build_detection_bh.py -- per-pathway BH detection rate across the 28 dims,
computed from run_group_eval.py's raw group_readout_ora_<TAG>_<db>.csv (one model fit per
group, shared by all labels within it). Same convention as
pipeline/03_evaluate/detection_rate.py: within each (method, dim, readout, db) batch,
min p over latent_idx -> adj_p = min_p * dim (Bonferroni over the D latents,
existing pipeline convention) -> BH across pathways in that batch -> detected
if q_bh < 0.05 -> detection_rate_pct = % of the 28 dims detected, per
(method, readout, db, pathway).

Output: results/detection_bh_<TAG>_<db>.csv
    columns: group, db, method, readout, pathway, detection_rate_pct

Usage:
    python build_detection_bh.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
from config import PHENOTYPE_DIR, SEED_SUFFIX, group_readout_ora_path
OUT = PHENOTYPE_DIR
from latent2genesets.data import group_tag as tag

# ORA background-universe mode ("all" default / "annotated"): decides which
# input+output filename variant this script reads and writes.
from latent2genesets import ora as _ora

# (dataset, group) pairs -- one group_readout_ora_<TAG>_<db>.csv per group (dataset+group
# together determine the file, since TAG = tag(dataset, group)).
GROUPS = [
    ("tcga", "BRCA"), ("tcga", "COAD"), ("tcga", "HNSC"), ("tcga", "THCA"), ("tcga", "LUAD"),
    ("gtex", "Muscle - Skeletal"), ("gtex", "Artery - Tibial"),
]
DBS = ["hallmark", "kegg"]
USECOLS = ["method", "dim", "latent_idx", "readout", "eval_pathway", "p"]


def bh_transform(s):
    return multipletests(s.values, method="fdr_bh")[1]


def run(dataset, group, db):
    tg = tag(dataset, group)
    path = group_readout_ora_path(tg, db, f"{_ora.OUT_SUFFIX}{SEED_SUFFIX}")
    print(f"[{tg}/{db}] loading {path.name}", flush=True)
    df = pd.read_csv(path, usecols=USECOLS,
                      dtype={"method": "category", "readout": "category",
                             "eval_pathway": "category"})
    print(f"[{tg}/{db}]   loaded {df.shape}", flush=True)

    g = (df.groupby(["method", "dim", "readout", "eval_pathway"], observed=True)["p"]
         .min().reset_index())
    del df
    g["adj_p"] = (g["p"] * g["dim"]).clip(upper=1.0)
    g["q_bh"] = g.groupby(["method", "dim", "readout"], observed=True)["adj_p"].transform(bh_transform)
    g["detected_bh"] = g["q_bh"] < 0.05

    rate = (g.groupby(["method", "readout", "eval_pathway"], observed=True)["detected_bh"]
            .mean().reset_index())
    rate["detection_rate_pct"] = rate["detected_bh"] * 100
    rate["group"] = group
    rate["db"] = db
    rate = rate.rename(columns={"eval_pathway": "pathway"})
    out_path = OUT / f"detection_bh_{tg}_{db}{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv"
    rate[["group", "db", "method", "readout", "pathway", "detection_rate_pct"]].to_csv(out_path, index=False)
    print(f"[{tg}/{db}] wrote {rate.shape} -> {out_path.name}", flush=True)


if __name__ == "__main__":
    for dataset, group in GROUPS:
        for db in DBS:
            run(dataset, group, db)
