"""
evaluation.py -- the pooled differential-expression reference used for every
pan-dataset AUROC, and the file bookkeeping shared by the controls.

A gene set is positive if it is detected by tumour-vs-normal or one-vs-rest
differential expression (final_detected_strict in final_reference.csv) in at
least one cancer type / tissue.
"""
import re

import pandas as pd

from config import REFERENCE_DIR


def build_pooled_reference(ref_file="final_reference.csv"):
    """Pool final_detected_strict (tumour_normal OR one_vs_rest; see
    pipeline/01_reference/build_reference.py) across groups: a gene set is
    positive in a dataset/database if it is detected in any group."""
    df = pd.read_csv(REFERENCE_DIR / ref_file)
    agg = df.groupby(["dataset", "db", "pathway"])["final_detected_strict"].max().reset_index()
    agg["pooled_reference_binary"] = agg["final_detected_strict"].astype(int)
    return agg.drop(columns="final_detected_strict")


def build_recurrence_reference(ref_file="final_reference.csv"):
    """How many cancer types / tissues each gene set is detected in
    (final_detected_strict), per (dataset, db, pathway) -> column n_groups_detected.
    Thresholding it at k gives the k-of-N reference; k = 1 is
    build_pooled_reference()'s pooled_reference_binary."""
    df = pd.read_csv(REFERENCE_DIR / ref_file)
    return (df.groupby(["dataset", "db", "pathway"])["final_detected_strict"].sum()
              .astype(int).rename("n_groups_detected").reset_index())


def replicate_shards(directory, prefix, replicate_tag=""):
    """The <prefix>*<replicate_tag>_minp.csv.gz files of exactly ONE replicate.

    A control run may be split into several shards (--tag _partA ...), and each
    model seed / permutation writes its own set, tagged _seed<n> / _perm<p>
    (config.seed_tag, config.perm_tag). `replicate_tag` is that tag ("" for the
    primary seed and permutation 0). Shards of any other seed or permutation
    are left out, so replicates are never pooled by accident."""
    end = f"{replicate_tag}_minp.csv.gz"
    out = []
    for f in sorted(directory.glob(f"{prefix}*_minp.csv.gz")):
        if not f.name.endswith(end):
            continue
        head = f.name[len(prefix):len(f.name) - len(end)]   # dataset, models, split tag
        if re.search(r"_(seed|perm)\d+", head):
            continue
        out.append(f)
    return out
