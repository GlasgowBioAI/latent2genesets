"""
de.py — DE-based ORA building blocks for the reference gene sets.

Per pair of groups (TCGA cancer types or GTEx tissue types):
  1. Welch t-test per gene between the two groups -> t statistic and logFC.
  2. z-score each statistic across genes; z > Z_THRESH is the 'up' query set
     (high in group_a) and z < -Z_THRESH is the 'down' query set (low in
     group_a), scored separately so direction is not lost the way a single
     |z| > Z_THRESH query would lose it (cf. latent2genesets/ora.py, which does not
     need direction since latent-component sign has no biological reference).
  3. Fisher's exact test (alternative="greater") per pathway per direction,
     same convention as latent2genesets/ora.py's ora_one_latent — no cross-latent
     min-p/×D step since there is only one ranking per pair (D=1).
  4. BH (Benjamini-Hochberg) across the pathway axis, done separately per
     direction, plus the raw p kept for a sensitivity check.

Reuses latent2genesets/ora.py::load_pathways for pathway parsing so pathway
definitions are byte-identical to the main pipeline.
"""

import numpy as np
import pandas as pd

from config import BIOBOMBE_DATA, GMTS
from latent2genesets.ora import load_pathways
from latent2genesets.data import load_dataset, sample_group_map, group_counts  # noqa: F401 (re-exported)

Z_THRESH = 2.0
MIN_SET_SIZE = 15
MAX_SET_SIZE = 500







# ----------------------------------------------------------------
# Expression matrices grouped by label (train+test combined)
# ----------------------------------------------------------------

_EXPR_CACHE: dict = {}


def load_expression_by_group(dataset: str, min_n: int = 50) -> dict:
    """Returns {group_name: (n_samples, n_genes) float32 array} and the
    shared gene_ids list, for all groups with >= min_n samples."""
    if dataset in _EXPR_CACHE:
        return _EXPR_CACHE[dataset]

    Xtr, Xte, gene_ids, train_s, test_s = load_dataset(
        dataset, n_genes=None, log_transform=True, scale=None
    )
    gene_ids = [str(g) for g in gene_ids]
    gmap = sample_group_map(dataset)

    all_samples = list(train_s) + list(test_s)
    X_all = np.concatenate([Xtr, Xte], axis=0)
    labels = np.array([gmap.get(s) for s in all_samples])

    counts = pd.Series(labels).value_counts()
    keep_groups = [g for g in counts.index if g is not None and counts[g] >= min_n]

    by_group = {}
    for g in keep_groups:
        by_group[g] = X_all[labels == g]

    result = (by_group, gene_ids)
    _EXPR_CACHE[dataset] = result
    return result


def load_tcga_tumor_normal_groups(min_n: int = 10) -> dict:
    """Returns {cancer_type: (Xa=tumor, Xb=matched normal)} for TCGA cancer
    types with >= min_n 'Solid Tissue Normal' samples (from
    tcga_sample_identifiers.tsv's sample_type column) -- a fixed external
    baseline (same GDC PanCanAtlas cohort/pipeline as the tumor samples, so
    no cross-dataset batch confound), unlike pairwise cancer-vs-cancer which
    can never see a pathway shared across most cancers (see PI3K/AKT/mTOR
    case)."""
    Xtr, Xte, gene_ids, train_s, test_s = load_dataset(
        "tcga", n_genes=None, log_transform=True, scale=None
    )
    gene_ids = [str(g) for g in gene_ids]
    all_samples = list(train_s) + list(test_s)
    X_all = np.concatenate([Xtr, Xte], axis=0)

    ident = pd.read_csv(BIOBOMBE_DATA / "tcga_sample_identifiers.tsv", sep="\t")
    ident = ident.set_index("sample_id")
    lab = ident.reindex(all_samples)
    cancer_type = lab["cancer_type"].values
    sample_type = lab["sample_type"].values

    is_tumor = sample_type == "Primary Solid Tumor"
    is_normal = sample_type == "Solid Tissue Normal"

    groups = {}
    for ct in pd.unique(cancer_type[~pd.isna(cancer_type)]):
        normal_mask = is_normal & (cancer_type == ct)
        if normal_mask.sum() < min_n:
            continue
        tumor_mask = is_tumor & (cancer_type == ct)
        groups[ct] = (X_all[tumor_mask], X_all[normal_mask])
    return groups, gene_ids


def load_one_vs_rest_groups(dataset: str, min_n: int = 50) -> dict:
    """Returns {group: (Xa=this group, Xb=pooled all other kept groups)} for
    every group kept by load_expression_by_group (n >= min_n) -- one test per
    group instead of (n_groups-1) pairwise tests.

    For GTEx this avoids the correlated-siblings issue (e.g. 13 brain
    sub-regions inflating/diluting each other's pairwise frequency). For TCGA
    it's the direct counterpart to tumor_normal: tumor_normal asks "does this
    cancer type differ from its own normal tissue", one_vs_rest asks "does
    this cancer type differ from the pan-cancer average" -- a pathway that
    only wins a majority of individual pairwise comparisons because a handful
    of unrelated comparators happen to sit low on some shared background axis
    (e.g. hallmark_estrogen_response_* winning against non-epithelial
    cancers like SARC/GBM/LAML) will not clear a single pooled
    test the same way, since the pooled reference averages that axis out."""
    by_group, gene_ids = load_expression_by_group(dataset, min_n=min_n)
    groups = {}
    for g in by_group:
        rest = np.concatenate([X for other, X in by_group.items() if other != g], axis=0)
        groups[g] = (by_group[g], rest)
    return groups, gene_ids


def load_gtex_one_vs_rest_groups(min_n: int = 50) -> dict:
    """Back-compat wrapper -- see load_one_vs_rest_groups."""
    return load_one_vs_rest_groups("gtex", min_n=min_n)


def load_pathway_sets(gene_ids):
    """Returns {name: (db, set(gene_index))} for hallmark+kegg, matching
    latent2genesets/ora.py sizing conventions."""
    pw_sets = {}
    for db, path in GMTS.items():
        pw, _ = load_pathways([path], gene_ids, MIN_SET_SIZE, MAX_SET_SIZE)
        for name, idx in pw.items():
            pw_sets[name] = (db, idx)
    return pw_sets


def ora_by_db(scores: np.ndarray, pw_sets: dict, n_genes: int):
    """ora_one_ranking with the background universe resolved per db.

    In the default "all" universe mode this is one call over the merged
    hallmark+kegg pw_sets, i.e. identical to a single
    ORA run over the combined gene-set collection.
    In "annotated" mode each db gets its own annotated background (they
    differ: ~4.2k genes for hallmark vs ~4.4k for kegg) and the query is
    intersected with it, so the two dbs cannot borrow each other's coverage.
    """
    from latent2genesets import ora as _ora
    if _ora.UNIVERSE_MODE == "all":
        return ora_one_ranking(scores, pw_sets, n_genes)
    out = []
    for db in sorted({d for d, _ in pw_sets.values()}):
        sub = {n: (d, g) for n, (d, g) in pw_sets.items() if d == db}
        U, restrict = _ora.universe_for({n: g for n, (_, g) in sub.items()}, n_genes)
        out += ora_one_ranking(scores, sub, U, restrict=restrict)
    return out


# ----------------------------------------------------------------
# Welch DE
# ----------------------------------------------------------------

def welch_de(Xa: np.ndarray, Xb: np.ndarray, eps: float = 1e-12):
    """Per-gene Welch t-test between two sample groups.

    Returns:
        t     : (n_genes,) Welch t statistic
        logfc : (n_genes,) mean(Xa) - mean(Xb)  (data already log2 scale)
    """
    mean_a, mean_b = Xa.mean(axis=0), Xb.mean(axis=0)
    var_a, var_b = Xa.var(axis=0, ddof=1), Xb.var(axis=0, ddof=1)
    na, nb = Xa.shape[0], Xb.shape[0]
    se = np.sqrt(var_a / na + var_b / nb)
    t = (mean_a - mean_b) / (se + eps)
    logfc = mean_a - mean_b
    return t.astype(np.float64), logfc.astype(np.float64)


# ----------------------------------------------------------------
# ORA on one ranking vector (D=1, so no min-p/×D step)
# ----------------------------------------------------------------

def ora_one_ranking(scores: np.ndarray, pw_sets: dict, universe_size: int, restrict=None):
    """scores: (n_genes,) statistic (t or logfc).

    Direction is relative to group_a (the ranking's sign convention):
    'up'   query = z > Z_THRESH   (high in group_a)
    'down' query = z < -Z_THRESH  (low in group_a, i.e. high in group_b)
    Each direction gets its own Fisher exact (alternative='greater') per
    pathway, so a pathway that is uniformly low in group_a and high in
    group_b lands in 'down', not 'up' — no cross-ranking aggregation,
    this is the only ranking for this pair.

    `restrict` (see ora.universe_for) confines both directions' queries to the
    annotated background when the annotated-universe mode is active.

    Returns list of dicts: pathway, db, direction, pathway_size, n_query,
    hit, p (raw p).
    """
    from scipy.stats import fisher_exact

    z = (scores - scores.mean()) / (scores.std() + 1e-12)
    queries = {
        "up": set(np.where(z > Z_THRESH)[0]),
        "down": set(np.where(z < -Z_THRESH)[0]),
    }
    if restrict is not None:
        queries = {d: q & restrict for d, q in queries.items()}

    out = []
    for direction, query in queries.items():
        n_query = len(query)
        for name, (db, genes) in pw_sets.items():
            a = len(query & genes)
            size = len(genes)
            if a == 0 or n_query == 0:
                p = 1.0
            else:
                b = n_query - a
                c = size - a
                d = universe_size - a - b - c
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            out.append({
                "pathway": name, "db": db, "direction": direction,
                "pathway_size": size, "n_query": n_query, "hit": a, "p": p,
            })
    return out


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out
