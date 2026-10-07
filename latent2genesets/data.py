"""
data.py -- BioBombe TCGA / GTEx expression loading and sample-group labels.

Pipeline: raw RSEM -> log2(x+1). StandardScaler is applied downstream in each
runner (fit on the training split, applied to validation and test). Train/test
partitions are BioBombe's predefined ones.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from config import BIOBOMBE_DATA, GTEX_SAMPLE_ATTR

# Dataset-specific file names
_FILES = {
    "tcga": {
        "mad":   "tcga_mad_genes.tsv",
        "train": "train_tcga_expression_matrix_processed.tsv.gz",
        "test":  "test_tcga_expression_matrix_processed.tsv.gz",
    },
    "gtex": {
        "mad":   "gtex_mad_genes.tsv",
        "train": "train_gtex_expression_matrix_processed.tsv.gz",
        "test":  "test_gtex_expression_matrix_processed.tsv.gz",
    },
}


def load_dataset(name: str, n_genes: int = None,
                 log_transform: bool = True, scale: str = "zero_one") -> tuple:
    """Load a BioBombe dataset (tcga or gtex).

    Args:
        n_genes: number of top-MAD genes to keep. None = all genes in matrix.

    Returns
    -------
    X_train : (n_train, n_genes) float32
    X_test  : (n_test,  n_genes) float32
    gene_ids : list of str  (Entrez gene IDs)
    train_samples : list of str
    test_samples  : list of str
    """
    name = name.lower()
    if name not in _FILES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_FILES)}")

    f = _FILES[name]
    gene_label = f"all" if n_genes is None else f"top-{n_genes}"
    print(f"Loading BioBombe {name.upper()} ({gene_label} MAD genes)...")

    print("  Reading train matrix...", flush=True)
    train_df = pd.read_csv(BIOBOMBE_DATA / f["train"], sep="\t", index_col=0)
    print("  Reading test matrix...", flush=True)
    test_df  = pd.read_csv(BIOBOMBE_DATA / f["test"],  sep="\t", index_col=0)

    if n_genes is not None:
        mad_df    = pd.read_csv(BIOBOMBE_DATA / f["mad"], sep="\t")
        top_genes = mad_df.head(n_genes)["gene_id"].astype(str).tolist()
        shared    = [g for g in top_genes if g in train_df.columns]
        train_df  = train_df[shared]
        test_df   = test_df[shared]
    else:
        shared = list(train_df.columns)
        test_df = test_df[[c for c in shared if c in test_df.columns]]

    print(f"  train shape: {train_df.shape}, test shape: {test_df.shape}")

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)
    gene_ids      = shared
    train_samples = train_df.index.tolist()
    test_samples  = test_df.index.tolist()

    if log_transform:
        X_train = np.log2(np.clip(X_train, 0, None) + 1)
        X_test  = np.log2(np.clip(X_test,  0, None) + 1)

    if scale == "zero_one":
        scaler  = MinMaxScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)
    elif scale == "zscore":
        mu      = X_train.mean(axis=0, keepdims=True)
        sd      = X_train.std(axis=0,  keepdims=True) + 1e-8
        X_train = ((X_train - mu) / sd).astype(np.float32)
        X_test  = ((X_test  - mu) / sd).astype(np.float32)
    elif scale is None:
        pass
    else:
        raise ValueError(f"Unknown scale: {scale}")

    print(f"  mean={X_train.mean():.4f}  std={X_train.std():.4f}  "
          f"range=[{X_train.min():.3f}, {X_train.max():.3f}]")

    return X_train, X_test, gene_ids, train_samples, test_samples




# ---------------------------------------------------------------- sample groups

def sample_group_map(dataset: str) -> dict:
    if dataset == "gtex":
        pheno = pd.read_csv(GTEX_SAMPLE_ATTR, sep="\t", usecols=["SAMPID", "SMTSD"], low_memory=False)
        return dict(zip(pheno["SAMPID"], pheno["SMTSD"]))
    ident = pd.read_csv(BIOBOMBE_DATA / "tcga_sample_identifiers.tsv", sep="\t")
    return dict(zip(ident["sample_id"], ident["cancer_type"]))


def group_counts(dataset: str) -> pd.Series:
    """Sample counts per group, from the *_sample_counts.tsv files (matches
    the counts already validated against the annotation files)."""
    fname = "tcga_sample_counts.tsv" if dataset == "tcga" else "gtex_sample_counts.tsv"
    df = pd.read_csv(BIOBOMBE_DATA / fname, sep="\t")
    col0, col1 = df.columns
    return df.set_index(col0)[col1]


def group_tag(dataset, group):
    g = group.replace(" - ", "-").replace(" ", "-").replace("(", "").replace(")", "")
    return f"{dataset.upper()}-{g}"


def load_group(dataset, group):
    """Load the full dataset, restrict to one group's samples, train/test kept
    separate for model fit/eval. Also returns the masked sample barcodes
    (train_s/test_s) so build_reference.py can join the labels in
    data/labels/ by sample_id."""
    X_tr, X_te, gene_ids, train_s, test_s = load_dataset(
        dataset, n_genes=None, log_transform=True, scale=None)
    gene_ids = [str(g) for g in gene_ids]
    gmap = sample_group_map(dataset)
    tr_mask = np.array([gmap.get(s) == group for s in train_s])
    te_mask = np.array([gmap.get(s) == group for s in test_s])
    train_s_g = np.array(train_s)[tr_mask]
    test_s_g = np.array(test_s)[te_mask]
    return X_tr[tr_mask], X_te[te_mask], gene_ids, train_s_g, test_s_g
