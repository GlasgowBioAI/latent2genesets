"""
build_reference.py -- within-group case/control differential-expression
reference for one phenotype contrast. Uses the group's predefined training
partition only, splits those samples by an external label from data/labels/,
runs Welch's t-test case vs control and ORA + BH across every gene set in each
database separately.

Training samples only, so the reference never uses the test samples on which
run_group_eval.py computes the gene-level readouts.

Output: results/reference_<TAG>_<LABEL>_<db>.csv
    columns: pathway, pathway_size, n_query, hit, raw_p, adj_p_bh, rank

Usage:
    python build_reference.py --dataset tcga --group BRCA --label er
    python build_reference.py --dataset tcga --group BRCA --label her2
    python build_reference.py --dataset tcga --group COAD --label msi
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE, GMTS, PHENOTYPE_DIR, LABELS_DIR
from latent2genesets import ora
from latent2genesets.de import welch_de, bh_adjust, ora_one_ranking
from latent2genesets.data import load_group, group_tag as tag

OUT = PHENOTYPE_DIR
OUT.mkdir(parents=True, exist_ok=True)

# (group, label) -> (csv filename, column, case value, control value)
LABEL_SPECS = {
    ("BRCA", "er"):   ("brca_er_pr_her2.csv", "er_binary", "Positive", "Negative"),
    ("BRCA", "her2"): ("brca_er_pr_her2.csv", "her2_binary", "Positive", "Negative"),
    ("BRCA", "pr"):   ("brca_er_pr_her2.csv", "pr_binary", "Positive", "Negative"),
    ("COAD", "msi"):  ("coad_msi.csv", "msi_binary", "MSI-H", "MSS"),
    ("HNSC", "hpv"):  ("hnsc_hpv.csv", "hpv_binary", "HPV+", "HPV-"),
    ("THCA", "braf"): ("thca_braf.csv", "braf_binary", "Mutant", "WT"),
    ("LUAD", "egfr"): ("luad_egfr.csv", "egfr_binary", "Mutant", "WT"),
    # GTEx within-tissue covariates. Group = GTEx SMTSD string.
    ("Muscle - Skeletal", "age"):     ("gtex_age.csv", "age_binary", "Young", "Old"),
    ("Muscle - Skeletal", "dthhrdy"): ("gtex_dthhrdy.csv", "dthhrdy_binary", "Fast", "Slow"),
    ("Artery - Tibial", "age"):       ("gtex_age.csv", "age_binary", "Young", "Old"),
}


def build_case_mask(group, label, sample_ids):
    fname, col, case_val, control_val = LABEL_SPECS[(group, label)]
    df = pd.read_csv(LABELS_DIR / fname)
    lut = dict(zip(df.sample_id, df[col]))
    vals = np.array([lut.get(s) for s in sample_ids])
    case_mask = vals == case_val
    control_mask = vals == control_val
    keep = case_mask | control_mask
    return keep, case_mask[keep]


def run_label(dataset, group, label, verbose=True):
    tg = tag(dataset, group)
    X_tr, X_te, gene_ids, train_s, test_s = load_group(dataset, group)
    X_all = X_tr
    sample_ids = train_s

    keep, case_mask = build_case_mask(group, label, sample_ids)
    X_all = X_all[keep]
    n_case, n_control = int(case_mask.sum()), int((~case_mask).sum())
    if verbose:
        print(f"[{tg}/{label}] {keep.sum()}/{len(keep)} samples labeled -- "
              f"n_case={n_case} n_control={n_control}", flush=True)
    if n_case < 2 or n_control < 2:
        raise ValueError(f"{tg}/{label}: not enough labeled samples (case={n_case}, control={n_control})")

    Xa, Xb = X_all[case_mask], X_all[~case_mask]
    t_stat, _ = welch_de(Xa, Xb)

    out_paths = {}
    for db, path in GMTS.items():
        pw, _ = ora.load_pathways([path], gene_ids, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE)
        pw_sets = {name: (db, genes) for name, genes in pw.items()}
        U, restrict = ora.universe_for(pw, X_all.shape[1])
        res = ora_one_ranking(t_stat, pw_sets, universe_size=U, restrict=restrict)
        df = pd.DataFrame(res)
        # BH per direction (matches pipeline/01_reference/run_pairwise_de.py).
        # Both 'up' (elevated in case) and 'down' (elevated in control) are
        # kept; a gene set is a reference positive if significant in either
        # direction.
        df["adj_p_bh"] = df.groupby("direction")["p"].transform(bh_adjust)
        df = df.rename(columns={"p": "raw_p"})
        df["direction"] = pd.Categorical(df["direction"], categories=["up", "down"], ordered=True)
        df = df.sort_values(["direction", "adj_p_bh"]).reset_index(drop=True)
        df["rank"] = df.groupby("direction", observed=True).cumcount() + 1
        df.insert(0, "label", label)
        df.insert(0, "group", group)
        df.insert(0, "dataset", dataset.upper())
        out_path = OUT / f"reference_{tg}_{label}_{db}{ora.OUT_SUFFIX}.csv"
        df.to_csv(out_path, index=False)
        out_paths[db] = out_path

        sig = df[df.adj_p_bh < 0.05]
        if verbose:
            print(f"[{tg}/{label}/{db}] {len(sig)}/{len(df)} pathways BH-significant "
                  f"(adj_p_bh<0.05):", flush=True)
            for _, row in sig.head(15).iterrows():
                print(f"    {row.pathway:55s} adj_p_bh={row.adj_p_bh:.3g}", flush=True)
            if len(sig) == 0:
                print(f"    [WARN] nothing significant -- check label coding / sample sizes", flush=True)
    return out_paths


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gtex"])
    ap.add_argument("--group", required=True, choices=[
        "BRCA", "COAD", "HNSC", "THCA", "LUAD",
        "Muscle - Skeletal", "Artery - Tibial",
    ])
    ap.add_argument("--label", required=True, choices=[
        "er", "her2", "pr", "msi", "hpv", "braf", "egfr", "age", "dthhrdy",
    ])
    args = ap.parse_args()
    run_label(args.dataset, args.group, args.label)
