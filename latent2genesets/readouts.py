"""
readouts.py -- signed gene-level scores for every latent dimension

Readout strategies:
  1. Weights:  the model's own loading / weight matrix
  2. Pearson:  Pearson r between each latent dimension and every gene
  3. Spearman: Spearman rho between each latent dimension and every gene

plus three gradient-based attribution (XAI) readouts for AE/DAE/VAE (bottom of
this file). Every readout returns an (n_genes, D) matrix of signed gene scores.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr


# ============================================================
# Readout 1: weight matrix
# ============================================================

def readout_weights(method_name, pca_model=None, ica_model=None, nmf_model=None,
                    ae_model=None, dae_model=None, vae_model=None,
                    latent_dim=32):
    """Gene weights of every latent dimension.

    PCA / ICA / NMF: components_ (loading matrix)
    AE / DAE: product of the two encoder weight matrices (input -> latent)
    VAE: product of the shared-encoder and mu weight matrices (input -> latent mean)

    Returns:
        (n_genes, latent_dim) array, one column of signed weights per dimension
    """
    if method_name == "PCA":
        # PCA.components_: (n_components, n_genes) -> transpose to (n_genes, n_comp)
        weights = pca_model.components_.T

    elif method_name == "ICA":
        # FastICA.components_: (n_components, n_genes) -> transpose
        weights = ica_model.components_.T

    elif method_name == "NMF":
        # NMF.components_: (n_components, n_genes) -> transpose to (n_genes, n_comp)
        weights = nmf_model.components_.T

    elif method_name == "AE":
        # AE encoder is nn.Sequential: Linear(512) -> BN -> ReLU -> Linear(latent)
        # effective linear map = second-layer weight @ first-layer weight (ignoring the ReLU)
        # encoder[0].weight: (512, n_genes), encoder[3].weight: (latent, 512)
        # encoder[3].weight @ encoder[0].weight: (latent, n_genes) -> .T: (n_genes, latent)
        w1 = ae_model.encoder[0].weight.detach().cpu()
        w2 = ae_model.encoder[3].weight.detach().cpu()
        weights = (w2 @ w1).numpy().T

    elif method_name == "DAE":
        w1 = dae_model.encoder[0].weight.detach().cpu()
        w2 = dae_model.encoder[3].weight.detach().cpu()
        weights = (w2 @ w1).numpy().T

    elif method_name == "VAE":
        # VAE fc_mu is a separate Linear(512, latent_dim)
        # shared_encoder: Linear(512) -> BN -> ReLU
        # effective map = fc_mu.weight @ shared_encoder[0].weight
        w1 = vae_model.shared_encoder[0].weight.detach().cpu()
        w2 = vae_model.fc_mu.weight.detach().cpu()
        weights = (w2 @ w1).numpy().T

    else:
        raise ValueError(f"unknown method: {method_name}")

    # shape check
    assert weights.shape[1] == latent_dim, (
        f"expected {latent_dim} dimensions, got {weights.shape[1]} (method {method_name})"
    )

    return weights  # (n_genes, latent_dim)


# ============================================================
# Readout 2: Pearson correlation
# ============================================================

def readout_pearson(Z_latent, X):
    """Pearson correlation between every latent dimension and every gene (vectorised).

    Args:
        Z_latent: latent representation, (n_samples, n_dims)
        X: expression matrix, (n_samples, n_genes)

    Returns:
        (n_genes, n_dims) array of Pearson r
    """
    # centre and normalise (vectorised Pearson r)
    X_c = X - X.mean(axis=0, keepdims=True)
    Z_c = Z_latent - Z_latent.mean(axis=0, keepdims=True)
    X_norm = np.sqrt((X_c ** 2).sum(axis=0, keepdims=True))
    Z_norm = np.sqrt((Z_c ** 2).sum(axis=0, keepdims=True))
    X_c /= X_norm + 1e-10
    Z_c /= Z_norm + 1e-10
    # (n_genes, n_dims) correlation matrix
    return X_c.T @ Z_c


# ============================================================
# Readout 3: Spearman correlation (vectorised)
# ============================================================

def readout_spearman(Z_latent, X):
    """Spearman rank correlation between every latent dimension and every gene,
    computed as the Pearson correlation of ranks.

    Args:
        Z_latent: latent representation, (n_samples, n_dims)
        X: expression matrix, (n_samples, n_genes)

    Returns:
        (n_genes, n_dims) array of Spearman rho
    """
    # column-wise ranks (argsort of argsort)
    X_ranks = np.apply_along_axis(
        lambda c: np.argsort(np.argsort(c)), 0, X
    ).astype(float)
    Z_ranks = np.apply_along_axis(
        lambda c: np.argsort(np.argsort(c)), 0, Z_latent
    ).astype(float)

    # centre and normalise (Pearson on ranks = Spearman)
    X_c = X_ranks - X_ranks.mean(axis=0, keepdims=True)
    X_c /= np.sqrt((X_c ** 2).sum(axis=0, keepdims=True)) + 1e-10
    Z_c = Z_ranks - Z_ranks.mean(axis=0, keepdims=True)
    Z_c /= np.sqrt((Z_c ** 2).sum(axis=0, keepdims=True)) + 1e-10

    # (n_genes, n_dims) correlation matrix
    corr = X_c.T @ Z_c
    return corr  # (n_genes, n_dims)


# ============================================================
# Weights + Pearson + Spearman for one fitted model
# ============================================================

def compute_readouts(method_name, Z_latent, X,
                     pca_model=None, ica_model=None, nmf_model=None,
                     ae_model=None, dae_model=None, vae_model=None):
    """All three non-XAI readouts of one fitted model.

    Returns:
        dict: {readout name: (n_genes, n_dims) score matrix}
    """
    n_dims = Z_latent.shape[1]
    results = {}

    # weights readout
    weights = readout_weights(
        method_name,
        pca_model=pca_model, ica_model=ica_model, nmf_model=nmf_model,
        ae_model=ae_model, dae_model=dae_model, vae_model=vae_model,
        latent_dim=n_dims
    )
    results["weights"] = weights

    # Pearson readout
    results["pearson"] = readout_pearson(Z_latent, X)

    # Spearman readout
    results["spearman"] = readout_spearman(Z_latent, X)

    return results


# ============================================================
# XAI readouts (neural-net methods only): per-latent gene attribution.
# Our AE/DAE expose .encode(x) -> z; VAE exposes .encode_mu(x) -> mu.
# ----------------------------------------------------------------

class _DimSliceEncoder(torch.nn.Module):
    """Maps input genes -> a single latent dim's activation, so captum can
    attribute that one latent back onto the genes. Uses the real non-linear
    encoder path (Linear->BN->ReLU->Linear), which is what makes XAI differ
    from the linearized `weights` readout."""
    def __init__(self, model, mtype, dim_idx):
        super().__init__()
        self.model, self.mtype, self.dim_idx = model, mtype, dim_idx

    def forward(self, x):
        if self.mtype == "VAE":
            z = self.model.encode_mu(x)
        else:
            z = self.model.encode(x)
        return z[:, self.dim_idx:self.dim_idx + 1]


# The 3 XAI methods kept (all gradient-based, GPU-friendly, theory-distinct):
#   ig         - Integrated Gradients (path integral of gradients)
#   deeplift   - DeepLift (reference-point difference propagation)
#   inputxgrad - Input x Gradient (naivest gradient attribution; baseline
#                to gauge how much the fancier methods actually add)
# Excluded: ShapleyValueSampling (~78 min/latent = ~7 days for D=128, infeasible
# at 16148 genes), GradientShap (redundant with ig), LRP (no BatchNorm1d rule).
XAI_METHODS = ("ig", "deeplift", "inputxgrad")


def require_xai():
    """Fail fast if captum is missing.

    compute_xai_readout is called inside a per-(method,dim) try/except in the
    runners, so an ImportError there degrades into a [WARN] per latent and a
    result file silently missing ig/deeplift/inputxgrad -- i.e. half the
    readout comparison. Call this once before a run starts instead.
    """
    import captum  # noqa: F401


def compute_xai_readout(model, X_test, mtype, n_comp, method, device, batch=512):
    """Returns (n_genes, n_comp) attribution matrix for one XAI method,
    averaged over test samples. ig/deeplift use a zero baseline;
    inputxgrad takes no baseline."""
    from captum.attr import IntegratedGradients, DeepLift, InputXGradient
    attr_cls = {"ig": IntegratedGradients, "deeplift": DeepLift,
                "inputxgrad": InputXGradient}[method]
    model = model.to(device).eval()
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    baseline = torch.zeros((1, Xt.shape[1]), device=device)
    n_genes = Xt.shape[1]
    out = np.zeros((n_genes, n_comp), dtype=np.float32)
    for d in range(n_comp):
        wrapper = _DimSliceEncoder(model, mtype, d).to(device).eval()
        attributor = attr_cls(wrapper)
        acc = np.zeros(n_genes, dtype=np.float64)
        n_seen = 0
        for i in range(0, Xt.shape[0], batch):
            xb = Xt[i:i + batch]
            if method == "inputxgrad":
                a = attributor.attribute(xb)
            else:
                a = attributor.attribute(xb, baselines=baseline)
            acc += a.detach().cpu().numpy().sum(axis=0)
            n_seen += xb.shape[0]
        out[:, d] = acc / max(n_seen, 1)
    return out



def xai_signed_and_abs(model, X_test, mtype, n_comp, method, device, batch=512):
    """ora.compute_xai_readout, but accumulating sum(a) and sum(|a|) in one pass."""
    from captum.attr import IntegratedGradients, DeepLift, InputXGradient
    attr_cls = {"ig": IntegratedGradients, "deeplift": DeepLift, "inputxgrad": InputXGradient}[method]
    model = model.to(device).eval()
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    baseline = torch.zeros((1, Xt.shape[1]), device=device)
    ng = Xt.shape[1]
    signed = np.zeros((ng, n_comp), dtype=np.float32)
    absval = np.zeros((ng, n_comp), dtype=np.float32)
    for d in range(n_comp):
        attributor = attr_cls(_DimSliceEncoder(model, mtype, d).to(device).eval())
        acc, acc_abs, n_seen = np.zeros(ng), np.zeros(ng), 0
        for i in range(0, Xt.shape[0], batch):
            xb = Xt[i:i + batch]
            a = attributor.attribute(xb) if method == "inputxgrad" else attributor.attribute(xb, baselines=baseline)
            a = a.detach().cpu().numpy()
            acc += a.sum(0)
            acc_abs += np.abs(a).sum(0)
            n_seen += xb.shape[0]
        signed[:, d] = acc / max(n_seen, 1)
        absval[:, d] = acc_abs / max(n_seen, 1)
    return signed, absval
