"""
run_pairwise_de.py — enumerate all pairwise group comparisons within TCGA
(cancer type vs cancer type) and GTEx (tissue vs tissue), run Welch DE + ORA
for each pair, and write one row per (pair, db, statistic, pathway).

BH (per db, matching build_detection_bh.py's convention of grouping by db)
is computed within each pair's pathway list, separately per statistic and
per direction ('up' = high in group_a, 'down' = low in group_a).

Usage:
    python run_pairwise_de.py --dataset tcga --pilot     # first 5 groups only
    python run_pairwise_de.py --dataset tcga
    python run_pairwise_de.py --dataset gtex --min-n 50
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from latent2genesets.de import (
    load_expression_by_group, load_pathway_sets, welch_de,
    ora_one_ranking, ora_by_db, bh_adjust, group_counts,
)

from latent2genesets import ora as _ora

from config import REFERENCE_DIR
OUT = REFERENCE_DIR
OUT.mkdir(exist_ok=True)

STATISTICS = ["t", "logfc"]


def run_dataset(dataset: str, min_n: int, pilot: bool):
    t0 = time.time()
    print(f"[{dataset}] loading expression by group (min_n={min_n})...", flush=True)
    by_group, gene_ids = load_expression_by_group(dataset, min_n=min_n)

    counts = group_counts(dataset)
    groups = sorted(by_group, key=lambda g: -counts.get(g, 0))
    if pilot:
        groups = groups[:5]
    print(f"[{dataset}] {len(groups)} groups (n>={min_n}): {groups}", flush=True)

    pw_sets = load_pathway_sets(gene_ids)
    universe_size = len(gene_ids)
    print(f"[{dataset}] {len(pw_sets)} pathways, universe={universe_size} "
          f"({time.time()-t0:.0f}s)", flush=True)

    pairs = list(itertools.combinations(groups, 2))
    print(f"[{dataset}] {len(pairs)} pairs to run", flush=True)

    tag = "pilot" if pilot else "full"
    out_path = OUT / f"pairwise_{dataset.upper()}_{tag}{_ora.OUT_SUFFIX}.csv"

    rows = []
    for i, (ga, gb) in enumerate(pairs):
        Xa, Xb = by_group[ga], by_group[gb]
        t_stat, logfc = welch_de(Xa, Xb)
        stat_vecs = {"t": t_stat, "logfc": logfc}

        for stat_name, vec in stat_vecs.items():
            res = ora_by_db(vec, pw_sets, universe_size)
            df = pd.DataFrame(res)
            df["adj_p_bh"] = df.groupby(["db", "direction"])["p"].transform(bh_adjust)
            df["detected_raw"] = df["p"] < 0.05
            df["detected_bh"] = df["adj_p_bh"] < 0.05
            df["dataset"] = dataset.upper()
            df["group_a"] = ga
            df["group_b"] = gb
            df["n_a"] = Xa.shape[0]
            df["n_b"] = Xb.shape[0]
            df["statistic"] = stat_name
            rows.append(df)

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            print(f"  [{dataset}] {i+1}/{len(pairs)} pairs done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    full = pd.concat(rows, ignore_index=True)
    cols = ["dataset", "group_a", "group_b", "n_a", "n_b", "statistic", "db",
            "direction", "pathway", "pathway_size", "n_query", "hit", "p",
            "adj_p_bh", "detected_raw", "detected_bh"]
    full = full[cols]
    full.to_csv(out_path, index=False)
    print(f"[{dataset}] wrote {full.shape} -> {out_path} ({time.time()-t0:.0f}s total)",
          flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tcga", "gtex", "both"], default="both")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--pilot", action="store_true",
                     help="Only run the first 5 largest groups (sanity check)")
    args = ap.parse_args()

    which = ["tcga", "gtex"] if args.dataset == "both" else [args.dataset]
    for ds in which:
        run_dataset(ds, args.min_n, args.pilot)


if __name__ == "__main__":
    main()
