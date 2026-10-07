"""
run_onevsrest_de.py — group vs. pooled rest-of-groups DE + ORA, one
comparison per group (not (n-1) pairwise).

Why (GTEx): GTEx has no tumor/normal axis to exploit the way TCGA does
(every sample is 'normal'), so this doesn't fix a PI3K/AKT-style blind spot
the way run_tumor_normal_de.py does for TCGA. What it does fix is the
correlated-siblings problem in the pairwise reference: e.g. 13 GTEx brain
sub-regions are highly similar to each other, so most of a brain region's
(n-1) pairwise comparisons are against other brain regions, diluting its
freq_bh for genuinely brain-specific pathways. Comparing each tissue against
ALL other kept tissues pooled (one test, bigger/more heterogeneous
reference) sidesteps that.

Why (TCGA): the direct counterpart to tumor_normal. tumor_normal asks "does
this cancer type differ from its own normal tissue"; one_vs_rest asks "does
this cancer type differ from the pan-cancer average" -- a pathway that only
wins a majority of individual pairwise comparisons because a handful of
unrelated comparators happen to sit low on some shared background axis
(e.g. hallmark_estrogen_response_* winning against non-epithelial cancers
like SARC/GBM/LAML for PRAD) will not clear a
single pooled test the same way, since the pooled reference averages that
axis out.

Usage:
    python run_onevsrest_de.py --dataset gtex --min-n 50
    python run_onevsrest_de.py --dataset tcga --min-n 40
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import time
from pathlib import Path

import pandas as pd

from latent2genesets.de import (
    load_one_vs_rest_groups, load_pathway_sets, welch_de,
    ora_one_ranking, ora_by_db, bh_adjust,
)

from latent2genesets import ora as _ora

from config import REFERENCE_DIR
OUT = REFERENCE_DIR
OUT.mkdir(exist_ok=True)

STATISTICS = ["t", "logfc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tcga", "gtex"], default="gtex")
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.dataset.upper()} one-vs-rest groups (min_n={args.min_n})...", flush=True)
    groups, gene_ids = load_one_vs_rest_groups(args.dataset, min_n=args.min_n)
    print(f"{len(groups)} groups ({time.time()-t0:.0f}s)", flush=True)

    pw_sets = load_pathway_sets(gene_ids)
    universe_size = len(gene_ids)
    print(f"{len(pw_sets)} pathways, universe={universe_size}", flush=True)

    rows = []
    for i, (tissue, (Xa, Xb)) in enumerate(sorted(groups.items())):
        t_stat, logfc = welch_de(Xa, Xb)
        stat_vecs = {"t": t_stat, "logfc": logfc}

        for stat_name, vec in stat_vecs.items():
            res = ora_by_db(vec, pw_sets, universe_size)
            df = pd.DataFrame(res)
            df["adj_p_bh"] = df.groupby(["db", "direction"])["p"].transform(bh_adjust)
            df["detected_raw"] = df["p"] < 0.05
            df["detected_bh"] = df["adj_p_bh"] < 0.05
            df["dataset"] = args.dataset.upper()
            df["group"] = tissue
            df["n_group"] = Xa.shape[0]
            df["n_rest"] = Xb.shape[0]
            df["statistic"] = stat_name
            rows.append(df)
        if (i + 1) % 10 == 0 or (i + 1) == len(groups):
            print(f"  [{i+1}/{len(groups)}] done ({time.time()-t0:.0f}s)", flush=True)

    full = pd.concat(rows, ignore_index=True)
    cols = ["dataset", "group", "n_group", "n_rest", "statistic", "db", "direction",
            "pathway", "pathway_size", "n_query", "hit", "p", "adj_p_bh",
            "detected_raw", "detected_bh"]
    full = full[cols]
    out_path = OUT / f"onevsrest_{args.dataset.upper()}_full{_ora.OUT_SUFFIX}.csv"
    full.to_csv(out_path, index=False)
    print(f"wrote {full.shape} -> {out_path} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
