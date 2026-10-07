"""
build_reference.py -- merges the differential-expression views into one
per-group gene-set table.

  pairwise      cancer-vs-cancer / tissue-vs-tissue (run_pairwise_de.py +
                aggregate_pairwise.py). freq_bh (direction='up') >= 0.5
                across a group's (n-1) pairwise comparisons.
  tumor_normal  TCGA only (run_tumor_normal_de.py): tumour vs. the same
                cancer type's matched normal tissue. adj_p_bh < 0.05.
  one_vs_rest   TCGA + GTEx (run_onevsrest_de.py): group vs. the pooled
                remaining groups. adj_p_bh < 0.05.

'final_detected' = detected by any view.
'final_detected_strict' = tumor_normal_detected OR one_vs_rest_detected.
The pooled evaluation reference (latent2genesets/evaluation.py) uses
'final_detected_strict'; pairwise evidence alone is not sufficient.

Only statistic='t' is used.

Usage:
    python build_reference.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import pandas as pd

# ORA background-universe mode ("all" default / "annotated"): decides which
# input+output filename variant this script reads and writes.
from latent2genesets import ora as _ora
from config import REFERENCE_DIR

OUT = REFERENCE_DIR
FREQ_THRESH = 0.5


def load_pairwise():
    frames = []
    for ds in ["TCGA", "GTEX"]:
        df = pd.read_csv(OUT / f"pairwise_{ds}_full{_ora.OUT_SUFFIX}.csv")
        df = df[(df.statistic == "t") & (df.direction == "up")].copy()
        df["dataset"] = ds
        frames.append(df[["dataset", "group", "db", "pathway", "pathway_size", "freq_bh"]]
                       .rename(columns={"freq_bh": "pairwise_freq_bh"}))
    return pd.concat(frames, ignore_index=True)


def load_tumor_normal():
    df = pd.read_csv(OUT / f"tumor_normal_TCGA_full{_ora.OUT_SUFFIX}.csv")
    df = df[(df.statistic == "t") & (df.direction == "up")].copy()
    df["dataset"] = "TCGA"
    return df[["dataset", "group", "db", "pathway", "pathway_size",
               "adj_p_bh", "detected_bh"]].rename(
        columns={"adj_p_bh": "tumor_normal_adj_p_bh", "detected_bh": "tumor_normal_detected"})


def load_one_vs_rest():
    frames = []
    for ds in ["TCGA", "GTEX"]:
        df = pd.read_csv(OUT / f"onevsrest_{ds}_full{_ora.OUT_SUFFIX}.csv")
        df = df[(df.statistic == "t") & (df.direction == "up")].copy()
        df["dataset"] = ds
        frames.append(df[["dataset", "group", "db", "pathway", "pathway_size",
                           "adj_p_bh", "detected_bh"]])
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={"adj_p_bh": "one_vs_rest_adj_p_bh", "detected_bh": "one_vs_rest_detected"})


def main():
    pairwise = load_pairwise()
    tumor_normal = load_tumor_normal()
    one_vs_rest = load_one_vs_rest()

    key = ["dataset", "group", "db", "pathway"]
    merged = pairwise.merge(tumor_normal.drop(columns="pathway_size"), on=key, how="outer")
    merged = merged.merge(one_vs_rest.drop(columns="pathway_size"), on=key, how="outer")

    merged["pairwise_detected"] = (merged["pairwise_freq_bh"] >= FREQ_THRESH).fillna(False)
    merged["tumor_normal_detected"] = merged["tumor_normal_detected"].fillna(False).astype(bool)
    merged["one_vs_rest_detected"] = merged["one_vs_rest_detected"].fillna(False).astype(bool)

    def sources(row):
        s = []
        if row["pairwise_detected"]:
            s.append("pairwise")
        if row["tumor_normal_detected"]:
            s.append("tumor_normal")
        if row["one_vs_rest_detected"]:
            s.append("one_vs_rest")
        return "+".join(s)

    merged["sources"] = merged.apply(sources, axis=1)
    merged["final_detected"] = merged["sources"] != ""
    merged["final_detected_strict"] = merged["tumor_normal_detected"] | merged["one_vs_rest_detected"]

    cols = ["dataset", "group", "db", "pathway", "pathway_size",
            "pairwise_freq_bh", "pairwise_detected",
            "tumor_normal_adj_p_bh", "tumor_normal_detected",
            "one_vs_rest_adj_p_bh", "one_vs_rest_detected",
            "final_detected", "final_detected_strict", "sources"]
    merged = merged[cols].sort_values(["dataset", "group", "final_detected"],
                                      ascending=[True, True, False])

    out_path = OUT / f"final_reference{_ora.OUT_SUFFIX}.csv"
    merged.to_csv(out_path, index=False)
    print(f"wrote {merged.shape} -> {out_path}")

    strict_hits = merged[merged.final_detected_strict]
    print(f"{len(strict_hits)} (group, gene set) reference hits "
          f"(tumor_normal OR one_vs_rest)")
    print(strict_hits["sources"].value_counts().to_string())


if __name__ == "__main__":
    main()
