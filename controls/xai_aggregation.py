"""
xai_aggregation.py -- sensitivity analysis for the sample aggregation of XAI
attributions.

Main pipeline (latent2genesets.readouts.compute_xai_readout): s_j = mean_i a_ij   (signed mean)
Here, from the SAME attribution pass:     s_j = mean_i |a_ij| (mean absolute)

A gene whose attribution is +5 in half the samples and -5 in the other half
scores ~0 under the signed mean but high under the mean absolute. Everything
downstream is unchanged: per-latent z-score, |z|>2 query, one-sided Fisher /
hypergeom ORA, min p over latents within a dim, AUROC vs the pooled DE reference
(main-analysis convention).

Models are re-trained exactly as in pipeline/02_readouts/run_readouts.py (same data prep, seed,
batch size, patience); the signed-mean readout of the re-trained model is kept
as a reproducibility check against pipeline/03_evaluate's values.

The model seed is config.SEED (LATENT2GENESETS_SEED); a seed other than the primary
one tags its files _seed<n>, and --summarize reads and compares within that seed only.

Output: <CONTROLS_DIR>/xai_aggregation_<DS><tag><seed>_minp.csv.gz
        (dataset, universe, method, readout, agg, dim, db, pathway, min_p)
        <CONTROLS_DIR>/xai_aggregation_cells<seed>.csv  (--summarize)
Usage:
  python controls/xai_aggregation.py --dataset tcga --dims 200,125,... --tag _partA
  python controls/xai_aggregation.py --summarize
  LATENT2GENESETS_SEED=43 python controls/xai_aggregation.py --dataset tcga
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import SEED, SEED_SUFFIX, DIMENSIONS, CONTROLS_DIR, EVALUATION_DIR, REFERENCE_DIR
from latent2genesets.models import AE as AEModel, DAE as DAEModel, VAE as VAEModel, train_model
from latent2genesets.readouts import xai_signed_and_abs, require_xai, XAI_METHODS
from latent2genesets.ora import Scorer
from latent2genesets.data import load_dataset
from latent2genesets.evaluation import build_pooled_reference, replicate_shards

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES = CONTROLS_DIR
NN = {"AE": AEModel, "DAE": DAEModel, "VAE": VAEModel}




def run(dataset, dims, tag):
    require_xai()
    t0 = time.time()
    Xtr_raw, Xte_raw, gid, _, _ = load_dataset(dataset, n_genes=None, log_transform=True, scale=None)
    gid = [str(g) for g in gid]
    idx = np.random.RandomState(SEED).permutation(Xtr_raw.shape[0])
    n_val = max(1, int(Xtr_raw.shape[0] * 0.1))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw[idx[n_val:]]).astype(np.float32)
    Xval = scaler.transform(Xtr_raw[idx[:n_val]]).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)
    ng = Xtr.shape[1]
    sc = Scorer(gid, ng)
    print(f"[{dataset}] train={Xtr.shape} test={Xte.shape} ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for D in dims:
        td = time.time()
        for m, Cls in NN.items():
            model = train_model(Cls(ng, D), Xtr, Xval, D, m, seed=SEED)
            for x in XAI_METHODS:
                signed, absval = xai_signed_and_abs(model, Xte, m, D, x, DEVICE)
                for agg, mat in (("signed", signed), ("abs", absval)):
                    for (mode, db), mp in sc.min_p(mat.astype(np.float64)).items():
                        rows.extend((dataset.upper(), mode, m, x, agg, D, db, n, p)
                                    for n, p in zip(sc.dbs[db]["names"], mp))
            del model
            torch.cuda.empty_cache()
        print(f"  [{dataset}] D={D:3d} ({time.time()-td:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    out = RES / f"xai_aggregation_{dataset.upper()}{tag}{SEED_SUFFIX}_minp.csv.gz"
    pd.DataFrame(rows, columns=["dataset", "universe", "method", "readout", "agg", "dim", "db", "pathway",
                                "min_p"]).to_csv(out, index=False)
    print(f"DONE {out.name} {time.time()-t0:.0f}s", flush=True)


def summarize():
    files = replicate_shards(RES, "xai_aggregation_", SEED_SUFFIX)
    print(f"seed {SEED}: {[f.name for f in files]}")
    mp = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cells = []
    for mode, suf in [("all", ""), ("annotated", "_annotU")]:
        if not (REFERENCE_DIR / f"final_reference{suf}.csv").exists():
            print(f"[skip universe={mode}] no reference for it")
            continue
        gt = build_pooled_reference(f"final_reference{suf}.csv")
        m = mp[mp.universe == mode].merge(gt, on=["dataset", "db", "pathway"], how="inner")
        m["score"] = -np.log10((m.min_p * m.dim).clip(upper=1.0).clip(lower=1e-300))
        for key, s in m.groupby(["dataset", "method", "readout", "agg", "dim", "db"]):
            y = s.pooled_reference_binary.values
            cells.append((mode, *key, roc_auc_score(y, s.score.values) if 0 < y.sum() < len(y) else np.nan))
    c = pd.DataFrame(cells, columns=["universe", "dataset", "method", "readout", "agg", "dim", "db", "auroc"])
    c.to_csv(RES / f"xai_aggregation_cells{SEED_SUFFIX}.csv", index=False)
    # main-analysis values for the same cells and the same seed (signed mean, main training run)
    for mode, suf in [("all", ""), ("annotated", "_annotU")]:
        pub_path = EVALUATION_DIR / f"per_dim_auroc_cells{suf}{SEED_SUFFIX}.csv"
        if not pub_path.exists():
            print(f"[skip universe={mode}] no {pub_path.name}")
            continue
        pub = pd.read_csv(pub_path)
        cc = c[c.universe == mode].pivot_table(index=["dataset", "method", "readout", "dim", "db"],
                                                columns="agg", values="auroc").reset_index()
        j = cc.merge(pub.rename(columns={"auroc": "published"}), on=["dataset", "method", "readout", "dim", "db"])
        pear = pub[pub.readout == "pearson"].groupby(["dataset", "db", "method"]).auroc.mean().rename("pearson_pub")
        t = j.groupby(["dataset", "db", "method", "readout"])[["published", "signed", "abs"]].mean()
        t = t.join(pear, on=["dataset", "db", "method"])
        t["abs-signed"] = t["abs"] - t["signed"]
        print(f"\n===== universe={mode}: AUROC mean over dims =====")
        print(t.round(3).to_string())
        print(f"  reproducibility |signed - published| per cell: median {np.median(np.abs(j.signed - j.published)):.3f}, "
              f"max {np.abs(j.signed - j.published).max():.3f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tcga", "gtex"])
    ap.add_argument("--dims", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.dataset, [int(x) for x in a.dims.split(",")] if a.dims else DIMENSIONS, a.tag)
