"""
run_readouts.py — fit every model x dimension x readout and score each latent with ORA.

Per docs/07_ORA_REDESIGN.md:
  - Train PCA/ICA/NMF/AE/DAE/VAE at all 15 dimensionalities on real TCGA/GTEx.
  - Readouts: linear methods -> weights/pearson/spearman; neural nets add the
    3 XAI attributions ig/deeplift/inputxgrad (6 total for AE/DAE/VAE).
  - Evaluate each latent's readout vector against MSigDB pathways with ORA
    (z-adaptive |z|>2 query set + Fisher exact), NOT GSEA.
  - NOTHING is aggregated: one output row per
    (pathway, method, dim, latent_idx, readout). No min-p, no correction,
    no collapsing.

The model seed is config.SEED (42; LATENT2GENESETS_SEED=<n> for a replicate, which
tags the output _seed<n>).

Output: results/readouts/readout_ora_<DS>[_seed<n>].csv
    columns: dataset, method, dim, latent_idx, readout, db, pathway,
             pathway_size, n_query, hit, p, neg_log_p

Usage:
    python pipeline/02_readouts/run_readouts.py --dataset tcga
    python pipeline/02_readouts/run_readouts.py --dataset gtex
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets


import os
import sys

# Single-threaded BLAS: R isn't involved here, but numpy/sklearn hit the same
# OpenBLAS tall-matrix cliff on 16148-row data.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA, NMF
from sklearn.preprocessing import StandardScaler

from config import SEED, SEED_SUFFIX, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE, DIMENSIONS, GMTS, READOUT_DIR, readout_ora_path
from latent2genesets.models import AE as AEModel, DAE as DAEModel, VAE as VAEModel, train_model, extract_latent
from latent2genesets.models import fit_pca_gpu as _fit_pca_gpu
from latent2genesets.readouts import compute_readouts, compute_xai_readout, require_xai, XAI_METHODS
from latent2genesets.data import load_dataset
from latent2genesets import ora
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = READOUT_DIR

FIELDS = ["dataset", "method", "dim", "latent_idx", "readout", "db",
          "pathway", "pathway_size", "n_query", "hit", "p", "neg_log_p"]




def run(dataset: str, n_genes=None, dims=None):
    dataset = dataset.lower()
    out_csv = readout_ora_path(dataset, f"{ora.OUT_SUFFIX}{SEED_SUFFIX}")
    if out_csv.exists() and out_csv.stat().st_size > 100:
        print(f"[SKIP] {out_csv.name} exists", flush=True)
        return

    require_xai()   # else ig/deeplift/inputxgrad silently vanish from the run
    print("=" * 72)
    print(f"  S2 {dataset.upper()} Baseline — ORA edition")
    print("=" * 72, flush=True)

    # ── data ──
    t0 = time.time()
    X_tr_raw, X_te_raw, gene_ids, _, _ = load_dataset(
        dataset, n_genes=n_genes, log_transform=True, scale=None)
    gene_ids = [str(g) for g in gene_ids]
    n_tr = X_tr_raw.shape[0]
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n_tr)
    n_val = max(1, int(n_tr * 0.1))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr_raw[idx[n_val:]]).astype(np.float32)
    X_val_s = scaler.transform(X_tr_raw[idx[:n_val]]).astype(np.float32)
    X_test_s = scaler.transform(X_te_raw).astype(np.float32)
    n_genes_a = X_train_s.shape[1]
    X_train_t = torch.tensor(X_train_s, device=DEVICE)
    X_test_t = torch.tensor(X_test_s, device=DEVICE)
    _nmf_min = X_train_s.min(axis=0, keepdims=True)
    X_train_nmf = X_train_s - _nmf_min + 1e-6
    X_test_nmf = np.maximum(X_test_s - _nmf_min + 1e-6, 1e-6)
    print(f"  data: train={X_train_s.shape} test={X_test_s.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ── pathways per db (kept separate so we record which db each came from) ──
    pathways_by_db = {}
    for db, path in GMTS.items():
        pw, _ = ora.load_pathways([path], gene_ids, GSEA_MIN_SET_SIZE, GSEA_MAX_SET_SIZE)
        pathways_by_db[db] = pw
        print(f"  {db}: {len(pw)} pathways", flush=True)
    # universe/query background per db -- see ora.UNIVERSE_MODE
    universe_by_db = {db: ora.universe_for(pw, n_genes_a) for db, pw in pathways_by_db.items()}
    for db, (u, _) in universe_by_db.items():
        print(f"  {db}: universe={u} (mode={ora.UNIVERSE_MODE})", flush=True)

    # ── ICA/NMF subsample (linear methods scale badly with 9000 samples) ──
    _MAX = 2000
    if X_train_s.shape[0] > _MAX:
        sub = np.random.RandomState(SEED + 999).choice(X_train_s.shape[0], _MAX, replace=False)
        X_ica_fit, X_nmf_fit = X_train_s[sub], X_train_nmf[sub]
    else:
        X_ica_fit, X_nmf_fit = X_train_s, X_train_nmf

    def _fit_ica(n_comp, res):
        try:
            ica = FastICA(n_components=n_comp, random_state=SEED, max_iter=300, tol=1e-4).fit(X_ica_fit)
            res[:] = ([ica, ica.transform(X_test_s)] if ica.n_iter_ < 300
                      else [None, np.zeros((X_test_s.shape[0], n_comp))])
        except Exception as e:
            print(f"  [WARN] ICA D={n_comp}: {e}", flush=True)
            res[:] = [None, np.zeros((X_test_s.shape[0], n_comp))]

    def _fit_nmf(n_comp, res):
        try:
            nmf = NMF(n_components=n_comp, random_state=SEED, max_iter=1000).fit(X_nmf_fit)
            res[:] = [nmf, nmf.transform(X_test_nmf)]
        except Exception as e:
            print(f"  [WARN] NMF D={n_comp}: {e}", flush=True)
            res[:] = [None, np.zeros((X_test_nmf.shape[0], n_comp))]

    def _train_ae(Cls, mt, n_comp):
        try:
            m = Cls(n_genes_a, n_comp)
            m = train_model(m, X_train_s, X_val_s, n_comp, mt, seed=SEED)
            return m, extract_latent(m, X_test_s, mt)
        except Exception as e:
            print(f"  [WARN] {mt} D={n_comp}: {e}", flush=True)
            return None, np.zeros((X_test_s.shape[0], n_comp))

    # ── thread-safe append writer ──
    _lock = threading.Lock()
    need_hdr = not out_csv.exists() or out_csv.stat().st_size == 0

    def write_rows(buf):
        nonlocal need_hdr
        with _lock:
            with open(out_csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if need_hdr:
                    w.writeheader()
                    need_hdr = False
                w.writerows(buf)

    def score_and_write(method, D, readout, mat, n_comp):
        """ORA every latent of one readout matrix against both dbs; write rows."""
        if mat is None or mat.shape[0] != n_genes_a or mat.shape[1] < n_comp:
            return 0
        buf = []
        for li in range(n_comp):
            scores = mat[:, li]
            for db, pathways in pathways_by_db.items():
                U_db, restrict_db = universe_by_db[db]
                for r in ora.ora_one_latent(scores, pathways, U_db, restrict=restrict_db):
                    buf.append({
                        "dataset": dataset.upper(), "method": method, "dim": D,
                        "latent_idx": li, "readout": readout, "db": db,
                        "pathway": r["pathway"],
                        "pathway_size": len(pathways[r["pathway"]]),
                        "n_query": r["n_query"], "hit": r["hit"],
                        "p": r["p"], "neg_log_p": r["neg_log_p"],
                    })
        write_rows(buf)
        return len(buf)

    total_rows = 0
    for D in tqdm(dims or DIMENSIONS, desc="Dimensions"):
        n_comp = min(D, X_train_s.shape[0], X_train_s.shape[1])
        t_d = time.time()

        # PCA (GPU)
        V_pca, pca_mu = _fit_pca_gpu(X_train_t, n_comp)
        Z_pca = ((X_test_t - pca_mu) @ V_pca).cpu().numpy()
        class _PCAModel:  # minimal shim so compute_readouts can read components_
            components_ = V_pca.T.cpu().numpy()
        pca_model = _PCAModel()

        # ICA + NMF in background threads while GPU trains AE/DAE/VAE
        ica_res = [None, np.zeros((X_test_s.shape[0], n_comp))]
        nmf_res = [None, np.zeros((X_test_nmf.shape[0], n_comp))]
        t_ica = threading.Thread(target=_fit_ica, args=(n_comp, ica_res), daemon=True)
        t_nmf = threading.Thread(target=_fit_nmf, args=(n_comp, nmf_res), daemon=True)
        t_ica.start(); t_nmf.start()
        ae_m, Z_ae = _train_ae(AEModel, "AE", n_comp)
        dae_m, Z_dae = _train_ae(DAEModel, "DAE", n_comp)
        vae_m, Z_vae = _train_ae(VAEModel, "VAE", n_comp)
        t_ica.join(); t_nmf.join()
        ica_m, Z_ica = ica_res
        nmf_m, Z_nmf = nmf_res

        rows_d = 0
        # linear methods: weights/pearson/spearman only
        for m_name, Z_m, pm, im, nm in [
            ("PCA", Z_pca, pca_model, None, None),
            ("ICA", Z_ica, None, ica_m, None),
            ("NMF", Z_nmf, None, None, nmf_m),
        ]:
            if Z_m is None or Z_m.ndim < 2:
                continue
            if m_name == "ICA" and im is None:
                print(f"  [WARN] ICA D={D}: did not converge, skipping", flush=True)
                continue
            if m_name == "NMF" and nm is None:
                print(f"  [WARN] NMF D={D}: fit failed, skipping", flush=True)
                continue
            try:
                rd = compute_readouts(m_name, Z_m, X_test_s,
                                      pca_model=pm, ica_model=im, nmf_model=nm)
            except Exception as e:
                print(f"  [WARN] readouts {m_name} D={D}: {e}", flush=True)
                continue
            for rname, mat in rd.items():
                rows_d += score_and_write(m_name, D, rname, mat, n_comp)

        # neural nets: weights/pearson/spearman + 3 XAI
        for m_name, Z_m, model, mtype in [
            ("AE", Z_ae, ae_m, "AE"),
            ("DAE", Z_dae, dae_m, "DAE"),
            ("VAE", Z_vae, vae_m, "VAE"),
        ]:
            if Z_m is None or Z_m.ndim < 2 or model is None:
                continue
            try:
                kw = {"ae_model": model} if mtype == "AE" else \
                     {"dae_model": model} if mtype == "DAE" else {"vae_model": model}
                rd = compute_readouts(m_name, Z_m, X_test_s, **kw)
            except Exception as e:
                print(f"  [WARN] readouts {m_name} D={D}: {e}", flush=True)
                rd = {}
            for rname, mat in rd.items():
                rows_d += score_and_write(m_name, D, rname, mat, n_comp)
            # XAI readouts
            for xname in XAI_METHODS:
                try:
                    xmat = compute_xai_readout(model, X_test_s, mtype, n_comp, xname, DEVICE)
                except Exception as e:
                    print(f"  [WARN] XAI {xname} {m_name} D={D}: {e}", flush=True)
                    continue
                rows_d += score_and_write(m_name, D, xname, xmat, n_comp)

        total_rows += rows_d
        print(f"  D={D:3d}: {rows_d} rows ({time.time()-t_d:.0f}s)", flush=True)

    print(f"\n  DONE. total rows={total_rows}  written {out_csv}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tcga", choices=["tcga", "gtex"])
    ap.add_argument("--n_genes", type=int, default=None)
    ap.add_argument("--dims", default=None, help="comma list; default = all 28 dims")
    args = ap.parse_args()
    run(args.dataset, args.n_genes, [int(x) for x in args.dims.split(",")] if args.dims else None)
