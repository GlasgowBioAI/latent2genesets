"""
run_tumor_normal_de.py — TCGA tumor vs. its own matched 'Solid Tissue Normal'
DE + ORA, one comparison per cancer type (not (n-1) pairwise).

Why: pairwise cancer-vs-cancer (run_pairwise_de.py) can only detect pathways
that DIFFER between two cancer types. A pathway shared by most cancers
(e.g. hallmark_pi3k_akt_mtor_signaling) never clears |z|>2 in any pairwise
comparison, because both sides are similarly elevated -- it silently drops
out of the reference even though it's a real, well-known oncogenic
program. Comparing each cancer type against its own matched normal tissue
(same GDC PanCanAtlas cohort/pipeline, so no cross-dataset batch confound)
gives a fixed external baseline that surfaces these pan-cancer programs.

Only run for cancer types with >= --min-n 'Solid Tissue Normal' samples
(~15-18 of 33 at min_n=10; see latent2genesets.de.load_tcga_tumor_normal_groups).

Usage:
    python run_tumor_normal_de.py --min-n 10
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import time
from pathlib import Path

import pandas as pd

from latent2genesets.de import (
    load_tcga_tumor_normal_groups, load_pathway_sets, welch_de,
    ora_one_ranking, ora_by_db, bh_adjust,
)

from latent2genesets import ora as _ora

from config import REFERENCE_DIR
OUT = REFERENCE_DIR
OUT.mkdir(exist_ok=True)

STATISTICS = ["t", "logfc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=10,
                    help="minimum # of matched normal samples required")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading TCGA tumor/normal groups (min_n={args.min_n})...", flush=True)
    groups, gene_ids = load_tcga_tumor_normal_groups(min_n=args.min_n)
    print(f"{len(groups)} cancer types with >= {args.min_n} matched normal samples: "
          f"{sorted(groups)} ({time.time()-t0:.0f}s)", flush=True)

    pw_sets = load_pathway_sets(gene_ids)
    universe_size = len(gene_ids)
    print(f"{len(pw_sets)} pathways, universe={universe_size}", flush=True)

    rows = []
    for i, (ct, (Xa, Xb)) in enumerate(sorted(groups.items())):
        t_stat, logfc = welch_de(Xa, Xb)
        stat_vecs = {"t": t_stat, "logfc": logfc}

        for stat_name, vec in stat_vecs.items():
            res = ora_by_db(vec, pw_sets, universe_size)
            df = pd.DataFrame(res)
            df["adj_p_bh"] = df.groupby(["db", "direction"])["p"].transform(bh_adjust)
            df["detected_raw"] = df["p"] < 0.05
            df["detected_bh"] = df["adj_p_bh"] < 0.05
            df["dataset"] = "TCGA"
            df["group"] = ct
            df["n_tumor"] = Xa.shape[0]
            df["n_normal"] = Xb.shape[0]
            df["statistic"] = stat_name
            rows.append(df)
        print(f"  [{i+1}/{len(groups)}] {ct} (n_tumor={Xa.shape[0]}, n_normal={Xb.shape[0]}) done "
              f"({time.time()-t0:.0f}s)", flush=True)

    full = pd.concat(rows, ignore_index=True)
    cols = ["dataset", "group", "n_tumor", "n_normal", "statistic", "db", "direction",
            "pathway", "pathway_size", "n_query", "hit", "p", "adj_p_bh",
            "detected_raw", "detected_bh"]
    full = full[cols]
    out_path = OUT / f"tumor_normal_TCGA_full{_ora.OUT_SUFFIX}.csv"
    full.to_csv(out_path, index=False)
    print(f"wrote {full.shape} -> {out_path} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
