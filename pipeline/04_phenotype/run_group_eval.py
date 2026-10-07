"""
run_group_eval.py -- label-blind unsupervised eval for ONE group (TCGA cancer
type or GTEx tissue). Fits PCA/ICA/NMF/AE/DAE/VAE at every D in
the full 28-dim grid on that group's raw expression, computes all 6 readouts
(weights/pearson/spearman + ig/deeplift/inputxgrad for the 3 NN methods), ORA
against every pathway in hallmark AND kegg (scored separately, both written
in the same pass since the fit doesn't depend on db).

Same training/scoring loop as pipeline/02_readouts/run_readouts.py, on one
group's samples only -- the readout matrices produced here are reused by build_reference.py /
build_auroc.py for however many real-label case/control splits exist within
this group (e.g. BRCA's output serves both the ER and HER2 jobs).

The model seed is config.SEED (42; LATENT2GENESETS_SEED=<n> for a replicate, which
tags the output _seed<n>).

Output: results/phenotype/group_readout_ora_<TAG>_<db>[_seed<n>].csv
        (TAG = "TCGA-BRCA" / "TCGA-COAD" / ...)
    columns: dataset, group, db, method, dim, latent_idx, readout,
             eval_pathway, pathway_size, n_query, hit, p, neg_log_p

Usage:
    python run_group_eval.py --dataset tcga --group BRCA
    python run_group_eval.py --dataset tcga --group COAD --dims 2,8,32
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA, NMF
from sklearn.preprocessing import StandardScaler

from config import SEED, SEED_SUFFIX, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE, DIMENSIONS, GMTS, PHENOTYPE_DIR, group_readout_ora_path
from latent2genesets.models import AE as AEModel, DAE as DAEModel, VAE as VAEModel, train_model, extract_latent
from latent2genesets.models import fit_pca_gpu as _fit_pca_gpu
from latent2genesets.readouts import compute_readouts, compute_xai_readout, require_xai, XAI_METHODS
from latent2genesets.data import load_dataset, load_group, group_tag as tag  # noqa: F401
from latent2genesets import ora

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = PHENOTYPE_DIR

DIMENSIONS_FULL = DIMENSIONS

UNSUP_FIELDS = ["dataset", "group", "db", "method", "dim", "latent_idx", "readout",
                 "eval_pathway", "pathway_size", "n_query", "hit", "p", "neg_log_p"]








def run_group(dataset, group, dims=None, verbose=True):
    dims = dims if dims is not None else DIMENSIONS_FULL
    t0 = time.time()
    tg = tag(dataset, group)

    out_csvs = {db: group_readout_ora_path(tg, db, f"{ora.OUT_SUFFIX}{SEED_SUFFIX}") for db in GMTS}
    if all(p.exists() and p.stat().st_size > 100 for p in out_csvs.values()):
        if verbose:
            print(f"[SKIP] {tg} -- both {out_csvs['hallmark'].name}-style files exist", flush=True)
        return out_csvs

    require_xai()   # else ig/deeplift/inputxgrad silently vanish from the run
    X_tr_raw, X_te_raw, gene_ids, _, _ = load_group(dataset, group)
    pathways_by_db = {}
    for db, path in GMTS.items():
        pw, _ = ora.load_pathways([path], gene_ids, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE)
        pathways_by_db[db] = pw
    if verbose:
        print(f"[{tg}] train={X_tr_raw.shape[0]} test={X_te_raw.shape[0]} "
              f"hallmark={len(pathways_by_db['hallmark'])} kegg={len(pathways_by_db['kegg'])}",
              flush=True)

    # ---- standardize (fit on train) ----
    n_tr = X_tr_raw.shape[0]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n_tr)
    n_val = max(1, int(n_tr * 0.1))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr_raw[idx[n_val:]]).astype(np.float32)
    X_val_s = scaler.transform(X_tr_raw[idx[:n_val]]).astype(np.float32)
    X_test_s = scaler.transform(X_te_raw).astype(np.float32)
    n_genes_a = X_train_s.shape[1]
    # universe/query background per db -- see ora.UNIVERSE_MODE
    universe_by_db = {db: ora.universe_for(pw, n_genes_a) for db, pw in pathways_by_db.items()}
    X_train_t = torch.tensor(X_train_s, device=DEVICE)
    X_test_t = torch.tensor(X_test_s, device=DEVICE)
    _nmf_min = X_train_s.min(axis=0, keepdims=True)
    X_train_nmf = X_train_s - _nmf_min + 1e-6
    X_test_nmf = np.maximum(X_test_s - _nmf_min + 1e-6, 1e-6)

    _MAX = 2000
    if X_train_s.shape[0] > _MAX:
        sub = np.random.RandomState(SEED + 999).choice(X_train_s.shape[0], _MAX, replace=False)
        X_ica_fit, X_nmf_fit = X_train_s[sub], X_train_nmf[sub]
    else:
        X_ica_fit, X_nmf_fit = X_train_s, X_train_nmf

    def _train_ae(Cls, mt, n_comp):
        try:
            m = Cls(n_genes_a, n_comp)
            m = train_model(m, X_train_s, X_val_s, n_comp, mt, seed=SEED)
            return m, extract_latent(m, X_test_s, mt)
        except Exception as e:
            print(f"  [WARN] {mt} D={n_comp}: {e}", flush=True)
            return None, None

    writers = {}
    fhs = {}
    for db, path in out_csvs.items():
        need_hdr = not path.exists() or path.stat().st_size == 0
        fh = open(path, "a", newline="")
        fhs[db] = fh
        w = csv.DictWriter(fh, fieldnames=UNSUP_FIELDS)
        if need_hdr:
            w.writeheader()
        writers[db] = w

    def score_and_write(method, D, readout, mat, n_comp):
        if mat is None or mat.shape[0] != n_genes_a or mat.shape[1] < n_comp:
            return 0
        total = 0
        for li in range(n_comp):
            scores = mat[:, li]
            for db, pathways in pathways_by_db.items():
                rows = []
                U_db, restrict_db = universe_by_db[db]
                for r in ora.ora_one_latent(scores, pathways, U_db, restrict=restrict_db):
                    rows.append({
                        "dataset": dataset.upper(), "group": group, "db": db,
                        "method": method, "dim": D, "latent_idx": li, "readout": readout,
                        "eval_pathway": r["pathway"], "pathway_size": len(pathways[r["pathway"]]),
                        "n_query": r["n_query"], "hit": r["hit"],
                        "p": r["p"], "neg_log_p": r["neg_log_p"],
                    })
                writers[db].writerows(rows)
                total += len(rows)
        return total

    total_rows = 0
    for D in dims:
        n_comp = min(D, X_train_s.shape[0], X_train_s.shape[1])
        t_d = time.time()

        V_pca, pca_mu = _fit_pca_gpu(X_train_t, n_comp)
        Z_pca = ((X_test_t - pca_mu) @ V_pca).cpu().numpy()

        class _PCAModel:
            components_ = V_pca.T.cpu().numpy()
        pca_model = _PCAModel()

        try:
            ica = FastICA(n_components=n_comp, random_state=SEED, max_iter=300, tol=1e-4).fit(X_ica_fit)
            Z_ica, ica_m = ica.transform(X_test_s), ica
        except Exception as e:
            print(f"  [WARN] ICA D={D}: {e}", flush=True)
            Z_ica, ica_m = None, None

        try:
            nmf = NMF(n_components=n_comp, random_state=SEED, max_iter=1000).fit(X_nmf_fit)
            Z_nmf, nmf_m = nmf.transform(X_test_nmf), nmf
        except Exception as e:
            print(f"  [WARN] NMF D={D}: {e}", flush=True)
            Z_nmf, nmf_m = None, None

        ae_m, Z_ae = _train_ae(AEModel, "AE", n_comp)
        dae_m, Z_dae = _train_ae(DAEModel, "DAE", n_comp)
        vae_m, Z_vae = _train_ae(VAEModel, "VAE", n_comp)

        rows_d = 0
        for m_name, Z_m, pm, im, nm in [
            ("PCA", Z_pca, pca_model, None, None),
            ("ICA", Z_ica, None, ica_m, None),
            ("NMF", Z_nmf, None, None, nmf_m),
        ]:
            if Z_m is None:
                continue
            try:
                rd = compute_readouts(m_name, Z_m, X_test_s, pca_model=pm, ica_model=im, nmf_model=nm)
            except Exception as e:
                print(f"  [WARN] readouts {m_name} D={D}: {e}", flush=True)
                continue
            for rname, mat in rd.items():
                rows_d += score_and_write(m_name, D, rname, mat, n_comp)

        for m_name, Z_m, model, mtype in [
            ("AE", Z_ae, ae_m, "AE"),
            ("DAE", Z_dae, dae_m, "DAE"),
            ("VAE", Z_vae, vae_m, "VAE"),
        ]:
            if Z_m is None or model is None:
                continue
            kw = {"ae_model": model} if mtype == "AE" else \
                 {"dae_model": model} if mtype == "DAE" else {"vae_model": model}
            try:
                rd = compute_readouts(m_name, Z_m, X_test_s, **kw)
            except Exception as e:
                print(f"  [WARN] readouts {m_name} D={D}: {e}", flush=True)
                rd = {}
            for rname, mat in rd.items():
                rows_d += score_and_write(m_name, D, rname, mat, n_comp)
            for xname in XAI_METHODS:
                try:
                    xmat = compute_xai_readout(model, X_test_s, mtype, n_comp, xname, DEVICE)
                except Exception as e:
                    print(f"  [WARN] XAI {xname} {m_name} D={D}: {e}", flush=True)
                    continue
                rows_d += score_and_write(m_name, D, xname, xmat, n_comp)

        total_rows += rows_d
        if verbose:
            print(f"[{tg}] D={D:3d}: {rows_d} rows ({time.time()-t_d:.0f}s)", flush=True)

    for fh in fhs.values():
        fh.close()
    if verbose:
        print(f"[{tg}] DONE. total_rows={total_rows} ({time.time()-t0:.0f}s total)", flush=True)
    return out_csvs


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gtex"])
    ap.add_argument("--group", required=True)
    ap.add_argument("--dims", default=",".join(map(str, DIMENSIONS_FULL)),
                     help="comma-separated dimension list (default: full 28-dim grid)")
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    run_group(args.dataset, args.group, dims=dims)
