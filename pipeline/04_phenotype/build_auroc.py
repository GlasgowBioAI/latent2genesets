"""
build_auroc.py -- per real-label job (BRCA-ER, BRCA-HER2, COAD-MSI), combine
that job's group-level group_readout_ora_<TAG>_<db>.csv (run_group_eval.py, shared
across labels within the same group) with its own
reference_<TAG>_<label>_<db>.csv (build_reference.py) and compute AUROC
per (db, method, dim, readout). Positive = pathway with adj_p_bh < 0.05 in
that job's own DE reference; negative = every other pathway tested in the
same db. Score = -log10(min p over latent_idx within a dim), same convention
as pipeline/03_evaluate/auroc_per_dim.py -- no cutoff on the unsupervised side.

The 3 jobs are reported SEPARATELY, not pooled -- BRCA-ER and BRCA-HER2
share the same model fit (not independent samples), and n_case differs a lot
across jobs (72 to 601), so a naive pool would let the biggest job dominate.

Usage:
    python build_auroc.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
from config import PHENOTYPE_DIR, SEED_SUFFIX, group_readout_ora_path
OUT = PHENOTYPE_DIR
from latent2genesets.data import group_tag as tag

# ORA background-universe mode ("all" default / "annotated"): decides which
# input+output filename variant this script reads and writes.
from latent2genesets import ora as _ora
                                  # GTEx tissue names must match group_readout_ora_<TAG>_*.csv
                                  # filenames written by run_group_eval.py; a local
                                  # copy of tag() here previously did not sanitize,
                                  # which happened to be harmless for TCGA cancer
                                  # codes (no spaces) but would silently mismatch
                                  # every GTEx job)

JOBS = [
    ("TCGA", "BRCA", "er"),
    ("TCGA", "BRCA", "her2"),
    ("TCGA", "BRCA", "pr"),
    ("TCGA", "COAD", "msi"),
    ("TCGA", "HNSC", "hpv"),
    ("TCGA", "THCA", "braf"),
    ("TCGA", "LUAD", "egfr"),
    ("GTEX", "Muscle - Skeletal", "age"),
    ("GTEX", "Muscle - Skeletal", "dthhrdy"),
    ("GTEX", "Artery - Tibial", "age"),
]
DBS = ["hallmark", "kegg"]


def run_job(dataset, group, label):
    tg = tag(dataset, group)
    frames = []
    for db in DBS:
        unsup_path = group_readout_ora_path(tg, db, f"{_ora.OUT_SUFFIX}{SEED_SUFFIX}")
        gt_path = OUT / f"reference_{tg}_{label}_{db}{_ora.OUT_SUFFIX}.csv"
        if not unsup_path.exists() or not gt_path.exists():
            print(f"[SKIP] {tg}/{label}/{db} -- missing {unsup_path.name if not unsup_path.exists() else gt_path.name}")
            continue
        unsup = pd.read_csv(unsup_path)
        gt = pd.read_csv(gt_path)
        # gt now carries one row per (pathway, direction) since reference
        # covers both up and down; a pathway is positive if EITHER direction
        # is significant. A plain dict(zip(gt.pathway, ...)) would silently
        # keep only whichever direction's row happens to sort last, dropping
        # real hits whose other-direction row is non-significant.
        gt_lut = gt.groupby("pathway")["adj_p_bh"].min().lt(0.05).to_dict()
        unsup = unsup[unsup.eval_pathway.isin(gt_lut)].copy()
        unsup["is_positive"] = unsup.eval_pathway.map(gt_lut).astype(int)
        unsup["db"] = db
        frames.append(unsup)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)

    best = (all_df.groupby(["db", "method", "dim", "readout", "eval_pathway"], observed=True)
                  .agg(p=("p", "min"), is_positive=("is_positive", "max"))
                  .reset_index())
    best["score"] = -np.log10(best["p"].clip(lower=1e-300))

    rows = []
    for (db, method, dim, readout), sub in best.groupby(["db", "method", "dim", "readout"], observed=True):
        y = sub["is_positive"].values
        if y.sum() == 0 or y.sum() == len(y):
            auc = np.nan
        else:
            auc = roc_auc_score(y, sub["score"].values)
        rows.append({"db": db, "method": method, "dim": dim, "readout": readout,
                     "auroc": auc, "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

    out = pd.DataFrame(rows).sort_values(["db", "method", "readout", "dim"])
    out_path = OUT / f"auroc_{tg}_{label}{_ora.OUT_SUFFIX}{SEED_SUFFIX}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n=== {tg}/{label} -> wrote {out.shape} -> {out_path.name} ===", flush=True)

    summary = out.groupby(["db", "method", "readout"])["auroc"].max().reset_index()
    summary = summary.sort_values("auroc", ascending=False)
    print(summary.to_string(index=False))
    return out


def main():
    # 14 fully independent jobs (own files in, own file out) -- run in
    # separate processes rather than one at a time.
    with ProcessPoolExecutor(max_workers=len(JOBS)) as ex:
        futs = {ex.submit(run_job, dataset, group, label): (dataset, group, label)
                for dataset, group, label in JOBS}
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
