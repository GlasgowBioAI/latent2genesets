"""
permutation_null.py -- gene-wise permutation null: every gene's expression is permuted
independently across samples (train and test separately), which destroys all
gene-gene co-expression while keeping each gene's marginal distribution. All
models are then RE-FIT on the shuffled data and read out exactly as in
pipeline/02_readouts/run_readouts.py (same preprocessing, models, readouts incl. XAI, |z|>2 ORA).

The reference is untouched (still built from the real data), so any AUROC above
0.5 would have to come from the pipeline itself.

Per-pathway min p over latents is saved (universe = all, as in the main analysis) and
scored against the pooled DE reference by --summarize.

Replicates. --perm p (p = 0 .. config.N_PERMUTATIONS - 1) draws permutation
replicate p: the data are shuffled with RandomState(777 + p) and every model is
re-fit. The permutation changes the DATA; the model seed stays config.SEED (42
unless LATENT2GENESETS_SEED is set), so permutation replicates and model-seed
replicates are separate things and are never pooled. p = 0 writes untagged
files, p > 0 tags them _perm<p>. pipeline/05_replicates/summarize_replicates.py
reports the mean and SD over the permutations.

Output: <CONTROLS_DIR>/permutation_null_<DS>_<models><tag><seed><perm>_minp.csv.gz
        <CONTROLS_DIR>/permutation_null_cells<seed><perm>.csv  (--summarize)
Usage:
  python controls/permutation_null.py --dataset tcga --models PCA,ICA,NMF
  python controls/permutation_null.py --dataset tcga --models AE,DAE,VAE --dims 2,5,10 --tag _partA
  python controls/permutation_null.py --summarize
  python controls/permutation_null.py --dataset tcga --models PCA,ICA,NMF --perm 3
  python controls/permutation_null.py --summarize --perm 3
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root: config + latent2genesets

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import FastICA, NMF
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import SEED, SEED_SUFFIX, perm_tag, DIMENSIONS, CONTROLS_DIR, EVALUATION_DIR
from latent2genesets.models import AE as AEModel, DAE as DAEModel, VAE as VAEModel, train_model, extract_latent
from latent2genesets.models import fit_pca_gpu as _fit_pca_gpu
from latent2genesets.readouts import compute_readouts, compute_xai_readout, XAI_METHODS
from latent2genesets.ora import Scorer
from latent2genesets.data import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES = CONTROLS_DIR
NN = {"AE": AEModel, "DAE": DAEModel, "VAE": VAEModel}


def shuffle_genes(X, rng):
    Xs = np.empty_like(X)
    for j in range(X.shape[1]):
        Xs[:, j] = X[rng.permutation(X.shape[0]), j]
    return Xs


def run(dataset, models, dims, tag="", perm=0):
    t0 = time.time()
    Xtr_raw, Xte_raw, gid, _, _ = load_dataset(dataset, n_genes=None, log_transform=True, scale=None)
    gid = [str(g) for g in gid]
    rng = np.random.RandomState(777 + perm)   # the permutation replicate; independent of SEED
    Xtr_raw, Xte_raw = shuffle_genes(Xtr_raw, rng), shuffle_genes(Xte_raw, rng)
    # --- identical to pipeline/02_readouts/run_readouts.py from here on ---
    idx = np.random.RandomState(SEED).permutation(Xtr_raw.shape[0])
    n_val = max(1, int(Xtr_raw.shape[0] * 0.1))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw[idx[n_val:]]).astype(np.float32)
    Xval = scaler.transform(Xtr_raw[idx[:n_val]]).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)
    ng = Xtr.shape[1]
    Xtr_t, Xte_t = torch.tensor(Xtr, device=DEVICE), torch.tensor(Xte, device=DEVICE)
    nmf_min = Xtr.min(axis=0, keepdims=True)
    Xtr_nmf, Xte_nmf = Xtr - nmf_min + 1e-6, np.maximum(Xte - nmf_min + 1e-6, 1e-6)
    sub = np.random.RandomState(SEED + 999).choice(Xtr.shape[0], min(2000, Xtr.shape[0]), replace=False)
    sc = Scorer(gid, ng)
    for d in sc.dbs.values():
        d["univ"] = {"all": d["univ"]["all"]}
    print(f"[{dataset}] shuffled; train={Xtr.shape} test={Xte.shape} ({time.time()-t0:.0f}s)", flush=True)
    rows = []

    def emit(method, readout, D, mat):
        for (_, db), mp in sc.min_p(np.asarray(mat, dtype=np.float64)).items():
            rows.extend((dataset.upper(), method, readout, D, db, n, p) for n, p in zip(sc.dbs[db]["names"], mp))

    for D in dims:
        td = time.time()
        for m in models:
            try:
                if m == "PCA":
                    torch.manual_seed(SEED)
                    V, mu = _fit_pca_gpu(Xtr_t, D)
                    Z = ((Xte_t - mu) @ V).cpu().numpy()
                    class _P:  # noqa: E306
                        components_ = V.T.cpu().numpy()
                    rd = compute_readouts("PCA", Z, Xte, pca_model=_P())
                elif m == "ICA":
                    ica = FastICA(n_components=D, random_state=SEED, max_iter=300, tol=1e-4).fit(Xtr[sub])
                    if ica.n_iter_ >= 300:
                        print(f"  [WARN] ICA D={D}: did not converge, skipping (same rule as S2)", flush=True)
                        continue
                    rd = compute_readouts("ICA", ica.transform(Xte), Xte, ica_model=ica)
                elif m == "NMF":
                    nmf = NMF(n_components=D, random_state=SEED, max_iter=1000).fit(Xtr_nmf[sub])
                    rd = compute_readouts("NMF", nmf.transform(Xte_nmf), Xte, nmf_model=nmf)
                else:
                    model = train_model(NN[m](ng, D), Xtr, Xval, D, m, seed=SEED)
                    Z = extract_latent(model, Xte, m)
                    rd = compute_readouts(m, Z, Xte, **{f"{m.lower()}_model": model})
                    for x in XAI_METHODS:
                        rd[x] = compute_xai_readout(model, Xte, m, D, x, DEVICE)
                    del model
                    torch.cuda.empty_cache()
                for rname, mat in rd.items():
                    emit(m, rname, D, mat)
            except Exception as e:
                print(f"  [WARN] {m} D={D}: {e}", flush=True)
        print(f"  [{dataset}] D={D:3d} {','.join(models)} ({time.time()-td:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    out = RES / f"permutation_null_{dataset.upper()}_{'-'.join(models)}{tag}{SEED_SUFFIX}{perm_tag(perm)}_minp.csv.gz"
    pd.DataFrame(rows, columns=["dataset", "method", "readout", "dim", "db", "pathway", "min_p"]).to_csv(out, index=False)
    print(f"DONE {out.name} {time.time()-t0:.0f}s", flush=True)


def summarize(perm=0):
    """AUROC per cell against the pooled DE reference (k >= 1, i.e. the
    main-analysis reference), next to the real-model values from pipeline/03_evaluate.
    One (seed, permutation) replicate at a time."""
    from latent2genesets.evaluation import build_pooled_reference, replicate_shards
    rep = SEED_SUFFIX + perm_tag(perm)
    gt = build_pooled_reference("final_reference.csv")
    files = replicate_shards(RES, "permutation_null_", rep)
    print(f"seed {SEED}, permutation {perm}: {[f.name for f in files]}")
    sh = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = sh.merge(gt, on=["dataset", "db", "pathway"], how="inner")
    m["score"] = -np.log10((m.min_p * m.dim).clip(upper=1.0).clip(lower=1e-300))
    rows = []
    for key, s in m.groupby(["dataset", "db", "method", "readout", "dim"]):
        y = s.pooled_reference_binary.values
        rows.append((*key, roc_auc_score(y, s.score.values) if 0 < y.sum() < len(y) else np.nan))
    null = pd.DataFrame(rows, columns=["dataset", "db", "method", "readout", "dim", "auroc"])
    null.to_csv(RES / f"permutation_null_cells{rep}.csv", index=False)
    real = pd.read_csv(EVALUATION_DIR / f"per_dim_auroc_cells{SEED_SUFFIX}.csv")
    key = ["dataset", "db", "method", "readout"]
    t = pd.concat([real.groupby(key).auroc.mean().rename("real"),
                   null.groupby(key).auroc.mean().rename("permutation_null"),
                   null.groupby(key).dim.nunique().rename("null_dims")], axis=1).dropna(subset=["permutation_null"])
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tcga", "gtex"])
    ap.add_argument("--models", default="PCA,ICA,NMF")
    ap.add_argument("--dims", default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tag", default="", help="output filename suffix for split runs")
    ap.add_argument("--perm", type=int, default=0,
                    help="permutation replicate p (shuffle seed 777 + p); 0 .. config.N_PERMUTATIONS - 1")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.perm)
    else:
        dims = [int(x) for x in a.dims.split(",")] if a.dims else DIMENSIONS
        run(a.dataset, a.models.split(","), dims, a.tag, a.perm)
