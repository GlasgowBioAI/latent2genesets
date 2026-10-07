"""
aggregate_pairwise.py — collapse pairwise DE-ORA results into a per-group
(cancer type / tissue type) pairwise gene-set table.

For group X and pathway P: X participates in (n_groups - 1) pairwise
comparisons (every pair is stored once, as an unordered (group_a, group_b)
row; direction ('up'/'down') is stored relative to group_a, so it gets
flipped when the row is exploded onto group_b — a pathway uniformly low in
group_a and high in everything else must land in group_a's 'down' bucket,
not get counted as if it were group_a's representative pathway). This
script explodes each pairwise row into both of its member groups and
aggregates, separately per direction:

    detection_frequency = (# pairs where pathway detected) / (# pairs total)

reported separately for BH-adjusted and raw (uncorrected) detection, and
separately per db/statistic/direction. No single hard threshold is imposed —
the frequency itself is the reference signal; pick a cutoff downstream
once its distribution has been inspected (see PLAN.md section 4). The
'up' direction (this group is elevated relative to the rest) is what
answers "what pathways does this cancer/tissue characteristically have".

Usage:
    python aggregate_pairwise.py --dataset tcga --tag pilot
    python aggregate_pairwise.py --dataset gtex --tag full
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import REFERENCE_DIR
OUT = REFERENCE_DIR

# ORA background-universe mode ("all" default / "annotated"): decides which
# input+output filename variant this script reads and writes.
from latent2genesets import ora as _ora


def aggregate(dataset: str, tag: str):
    in_path = OUT / f"pairwise_{dataset.upper()}_{tag}{_ora.OUT_SUFFIX}.csv"
    df = pd.read_csv(in_path)

    # direction is stored relative to group_a ('up' = high in group_a).
    # From group_b's side that same row means the opposite: flip it so
    # 'up'/'down' always means "relative to this row's own group", not
    # left stuck relative to whichever group happened to be group_a.
    flip = {"up": "down", "down": "up"}
    long = pd.concat([
        df.assign(group=df["group_a"]),
        df.assign(group=df["group_b"], direction=df["direction"].map(flip)),
    ], ignore_index=True)

    g = long.groupby(["group", "db", "statistic", "direction", "pathway"], observed=True)
    agg = g.agg(
        pathway_size=("pathway_size", "first"),
        n_pairs=("detected_bh", "size"),
        n_detected_bh=("detected_bh", "sum"),
        n_detected_raw=("detected_raw", "sum"),
        median_hit_frac=("hit", lambda s: (s / long.loc[s.index, "pathway_size"]).median()),
    ).reset_index()
    agg["freq_bh"] = agg["n_detected_bh"] / agg["n_pairs"]
    agg["freq_raw"] = agg["n_detected_raw"] / agg["n_pairs"]

    out_path = OUT / f"pairwise_{dataset.upper()}_{tag}{_ora.OUT_SUFFIX}.csv"
    # 'up' (this group's own representative/elevated pathways) sorts before
    # 'down' (pathways this group is characteristically low in) within each group.
    agg["direction"] = pd.Categorical(agg["direction"], categories=["up", "down"], ordered=True)
    agg = agg.sort_values(["group", "db", "statistic", "direction", "freq_bh"],
                           ascending=[True, True, True, True, False])
    agg.to_csv(out_path, index=False)
    print(f"[{dataset}] wrote {agg.shape} -> {out_path}", flush=True)

    # quick distribution summary to help pick a threshold later
    print(f"\n[{dataset}] freq_bh distribution (per group x db x statistic x pathway):")
    print(agg["freq_bh"].describe(), flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tcga", "gtex", "both"], default="both")
    ap.add_argument("--tag", choices=["pilot", "full"], default="pilot")
    args = ap.parse_args()
    which = ["tcga", "gtex"] if args.dataset == "both" else [args.dataset]
    for ds in which:
        aggregate(ds, args.tag)


if __name__ == "__main__":
    main()
