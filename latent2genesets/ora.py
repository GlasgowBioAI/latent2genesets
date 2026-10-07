"""
ora.py — Over-representation analysis (Fisher's exact) on latent-feature readouts.

Replaces the Java-GSEA evaluation with a fast pure-Python ORA. Per the S2/S3
redesign (docs/07_ORA_REDESIGN.md):

  - Gene selection is z-score ADAPTIVE, not fixed top-N: standardize each
    latent's readout vector, take genes with |z| > Z_THRESH as the query set.
    A concentrated latent yields few query genes, a diffuse one yields many --
    the count follows each latent's own distribution instead of a hand-picked
    top-N.
  - Universe: selectable, see UNIVERSE_MODE below.
  - NOTHING is collapsed: every (pathway, method, dim, latent_idx, readout)
    produces its own row. No min-p, no within-D correction, no cross-D/method
    aggregation.
"""

import os

import numpy as np
from scipy.special import gammaln
from scipy.stats import fisher_exact

Z_THRESH = 2.0          # |z| cutoff defining a latent's high-weight query gene set
MIN_SET_SIZE = 15       # pathway must have >= this many genes present in universe
MAX_SET_SIZE = 500

# ----------------------------------------------------------------
# Fisher background universe
# ----------------------------------------------------------------
# "all" (the original, and still the default so every existing result file
# reproduces byte-for-byte): universe = every measured gene, query = every
# gene passing |z| > Z_THRESH. Only ~4.2k of the 16148 measured genes carry
# any hallmark/KEGG-legacy annotation, so the other ~12k can never be a hit
# for any pathway; parking them in the 2x2 table's "neither" cell deflates
# the expected overlap and inflates enrichment. Measured on BRCA-HER2:
# hallmark_angiogenesis goes from 5.1x/p=7.7e-3 to 3.0x/p=4.4e-2 once they
# are removed -- enough to move borderline sets across BH.
#
# "annotated": universe = genes annotated in the db under test, query
# intersected with the same set. This is the clusterProfiler/DAVID/GOseq
# convention. Selected with ORA_UNIVERSE=annotated; runners append
# OUT_SUFFIX to their output filenames so the two modes never collide.
UNIVERSE_MODE = os.environ.get("ORA_UNIVERSE", "all").lower()
if UNIVERSE_MODE not in ("all", "annotated"):
    raise ValueError(f"ORA_UNIVERSE must be 'all' or 'annotated', got {UNIVERSE_MODE!r}")
OUT_SUFFIX = "" if UNIVERSE_MODE == "all" else "_annotU"


def universe_for(pathways, n_genes):
    """(universe_size, restrict) for the active UNIVERSE_MODE.

    `restrict` is None in "all" mode and the set of annotated gene indices in
    "annotated" mode; pass it straight through to ora_one_latent/ora_one_ranking.
    """
    if UNIVERSE_MODE == "all":
        return n_genes, None
    annotated = set().union(*pathways.values()) if pathways else set()
    return len(annotated), annotated


def load_pathways(gmt_paths, gene_ids, min_size=MIN_SET_SIZE, max_size=MAX_SET_SIZE):
    """Parse GMT files into {pathway_name: set(gene_index)} restricted to genes
    present in `gene_ids`, keeping only pathways within [min_size, max_size]."""
    gene_idx = {g: i for i, g in enumerate(gene_ids)}
    gene_set = set(gene_ids)
    pathways, sizes = {}, {}
    for gmt_path in gmt_paths:
        with open(gmt_path) as fh:
            for line in fh:
                p = line.strip().split("\t")
                if len(p) < 3:
                    continue
                name, members = p[0].lower(), p[2:]
                present = gene_set.intersection(members)
                if not (min_size <= len(present) <= max_size):
                    continue
                pathways[name] = {gene_idx[g] for g in present}
                sizes[name] = len(present)
    return pathways, sizes


def ora_one_latent(scores, pathways, universe_size, restrict=None):
    """ORA for a single latent's readout vector against all pathways.

    scores : 1-D array (n_genes,), the readout for this latent
    pathways : {name: set(gene_index)}
    restrict : optional set of gene indices the query is confined to (the
        annotated background, see universe_for); None keeps every gene.
    Returns list of dicts, one per pathway (only pathways whose query overlap
    makes them testable; a pathway with 0 overlap gets p=1 -> neg_log_p=0).
    """
    z = (scores - scores.mean()) / (scores.std() + 1e-12)
    query = set(np.where(np.abs(z) > Z_THRESH)[0])
    if restrict is not None:
        query &= restrict
    n_query = len(query)
    out = []
    if n_query == 0:
        # degenerate latent (no gene stands out): every pathway is p=1
        for name, genes in pathways.items():
            out.append({"pathway": name, "n_query": 0, "hit": 0,
                        "p": 1.0, "neg_log_p": 0.0})
        return out
    for name, genes in pathways.items():
        a = len(query & genes)                 # in query AND in pathway
        b = n_query - a                        # in query, not in pathway
        c = len(genes) - a                     # in pathway, not in query
        d = universe_size - a - b - c          # neither
        if a == 0:
            p = 1.0
        else:
            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        out.append({"pathway": name, "n_query": n_query, "hit": a,
                    "p": p, "neg_log_p": float(-np.log10(max(p, 1e-300)))})
    return out


# ----------------------------------------------------------------
# Vectorised ORA (used by the controls; equal to ora_one_latent to ~1e-12)
# ----------------------------------------------------------------

def hypergeom_sf_fast(a, M, k_arr, N):
    """P(X>=a) for X~Hypergeom(M,k_arr[p],N), elementwise. a: (R,P) int
    array, k_arr: (P,), M/N scalars."""
    P = k_arr.shape[0]
    Xmax = int(min(N, k_arr.max()))
    x = np.arange(Xmax + 1)
    lo = np.maximum(0, N + k_arr - M)
    hi = np.minimum(k_arr, N)
    xg = x[:, None].astype(np.float64)
    kk = k_arr[None, :].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        log_pmf = (gammaln(kk + 1) - gammaln(xg + 1) - gammaln(kk - xg + 1)
                   + gammaln(M - kk + 1) - gammaln(N - xg + 1) - gammaln(M - kk - N + xg + 1)
                   - (gammaln(M + 1) - gammaln(N + 1) - gammaln(M - N + 1)))
        pmf = np.exp(log_pmf)
    valid = (x[:, None] >= lo[None, :]) & (x[:, None] <= hi[None, :])
    pmf = np.where(valid & np.isfinite(pmf), pmf, 0.0)
    sf_table = np.cumsum(pmf[::-1, :], axis=0)[::-1, :]
    a_clip = np.clip(a, 0, Xmax)
    out = sf_table[a_clip, np.arange(P)[None, :]]
    return np.where(a > Xmax, 0.0, out)


def colnorm(M):
    M = M - M.mean(axis=0, keepdims=True)
    return M / (np.sqrt((M ** 2).sum(axis=0, keepdims=True)) + 1e-10)


class Scorer:
    """Vectorised equivalent of ora.ora_one_latent over a whole readout matrix,
    for both universe modes at once. Returns min p over latents per pathway."""

    def __init__(self, gene_ids, n_genes):
        from config import GMTS, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE
        self.dbs = {}
        for db, path in GMTS.items():
            pw, _ = load_pathways([path], gene_ids, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE)
            names = list(pw.keys())
            Y = np.zeros((len(names), n_genes), dtype=np.float32)
            for i, n in enumerate(names):
                Y[i, sorted(pw[n])] = 1.0
            annotated = Y.max(0) > 0
            self.dbs[db] = dict(names=names, Y=Y, k=Y.sum(1).astype(np.int64),
                                univ={"all": (n_genes, None),
                                      "annotated": (int(annotated.sum()), annotated)})
            print(f"  {db}: {len(names)} pathways, annotated universe={int(annotated.sum())}", flush=True)

    def min_p(self, mat):
        z = (mat - mat.mean(0, keepdims=True)) / (mat.std(0, keepdims=True) + 1e-12)
        Q = (np.abs(z) > Z_THRESH)
        out = {}
        for db, d in self.dbs.items():
            for mode, (U, mask) in d["univ"].items():
                Qm = Q if mask is None else (Q & mask[:, None])
                nq = Qm.sum(0)
                A = (d["Y"] @ Qm.astype(np.float32)).round().astype(np.int64)  # (P, D)
                mp = np.ones(len(d["names"]))
                for li in range(mat.shape[1]):
                    if nq[li] == 0:
                        continue
                    a = A[:, li]
                    p = hypergeom_sf_fast(a[None, :], U, d["k"], int(nq[li]))[0]
                    p = np.where(a == 0, 1.0, p)
                    np.minimum(mp, p, out=mp)
                out[(mode, db)] = mp
        return out
