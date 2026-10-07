"""
models.py -- AE, DAE and VAE models and their training loop

All three use one 512-unit hidden layer (ReLU + BatchNorm) in the encoder and decoder,
following BioBombe (whose ADAGE architecture is n_genes -> 100 (ReLU) -> latent -> 100 (ReLU) -> n_genes);
the wider 512-unit layer matches the ~16-18k-gene input.

Architectures:
  AE/DAE:  n_genes -> 512 (ReLU+BN) -> latent -> 512 (ReLU+BN) -> n_genes (linear)
  VAE:     n_genes -> 512 (ReLU+BN) -> linear (mu, logvar) -> sample -> 512 (ReLU+BN) -> n_genes (sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

from config import (
    AE_LR, AE_BATCH_SIZE, AE_EPOCHS, AE_PATIENCE,
    DAE_NOISE_PROB, VAE_KAPPA,
)


# ============================================================
# AE -- plain autoencoder
# ============================================================

class AE(nn.Module):
    """Autoencoder with one hidden layer (ReLU+BN) in encoder and decoder.
    Encoder: n_genes -> 512 (ReLU+BN) -> latent_dim (linear)
    Decoder: latent_dim -> 512 (ReLU+BN) -> n_genes (linear)
    Loss: MSE (inputs are StandardScaler-transformed, ~N(0,1))
    """

    def __init__(self, n_genes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_genes),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        """Latent representation z."""
        return self.encoder(x)


# ============================================================
# DAE -- denoising autoencoder
# ============================================================

class DAE(nn.Module):
    """Denoising autoencoder, same architecture as AE.
    During training a binomial mask sets DAE_NOISE_PROB (30%) of the input genes
    to zero; no noise at test time.
    Refs: Vincent et al. 2008 JMLR; BioBombe ADAGE.
    """

    def __init__(self, n_genes, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_genes),
        )
        self.noise_prob = DAE_NOISE_PROB

    def forward(self, x, corrupt=True):
        if corrupt and self.training:
            # binomial mask: zero out a random 30% of the genes
            mask = torch.bernoulli(
                torch.ones_like(x) * (1 - self.noise_prob)
            )
            x = x * mask
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        """Latent representation from the clean (unmasked) input, used at test time."""
        return self.encoder(x)


# ============================================================
# VAE -- variational autoencoder
# ============================================================

class VAE(nn.Module):
    """Variational autoencoder; same hidden layers as AE, encoder outputs mu and logvar.
    Loss: MSE_recon + kappa * KL(N(mu, sigma^2) || N(0, 1)), kappa warmed up to VAE_KAPPA.
    Refs: Way & Greene 2018 PSB (Tybalt); Kingma & Welling 2014 ICLR.

    NOTE: the decoder ends in a sigmoid (output in [0, 1]) although train_model()
    feeds StandardScaler-transformed data (~N(0,1)), so negative values cannot be
    reconstructed. Kept as-is because it is the configuration behind the reported
    results.
    """

    def __init__(self, n_genes, latent_dim):
        super().__init__()
        # shared hidden layer (encoder)
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)          # mean
        self.fc_logvar = nn.Linear(512, latent_dim)       # log variance

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_genes),
            nn.Sigmoid(),  # see class docstring: output in [0, 1]
        )

    def encode(self, x):
        """(mu, logvar)."""
        h = self.shared_encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterisation trick: z = mu + eps * sigma, eps ~ N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode_mu(self, x):
        """Posterior mean mu (deterministic encoding, used at test time)."""
        h = self.shared_encoder(x)
        return self.fc_mu(h)


# ============================================================
# Training (shared by AE / DAE / VAE)
# ============================================================

def train_model(model, X_train, X_val, latent_dim,
                model_type="AE", seed=42):
    """Train an AE / DAE / VAE with early stopping on the validation loss.

    Args:
        model: freshly initialised AE, DAE or VAE
        X_train: training matrix (n_train, n_genes)
        X_val: validation matrix (n_val, n_genes)
        latent_dim: number of latent dimensions
        model_type: "AE", "DAE" or "VAE"
        seed: random seed

    Returns:
        the model with the weights of the epoch with the lowest validation loss
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # standardise (fit on the training set, apply to training and validation sets)
    scaler = StandardScaler()
    train_s = scaler.fit_transform(X_train)
    val_s = scaler.transform(X_val)

    train_t = torch.tensor(train_s, dtype=torch.float32).to(device)
    val_t = torch.tensor(val_s, dtype=torch.float32).to(device)

    # move the model to GPU / CPU
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)

    # data loader
    loader = DataLoader(TensorDataset(train_t), batch_size=AE_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # VAE KL warm-up: kappa rises linearly from 0 to VAE_KAPPA over the first 10 epochs
    def kl_weight(epoch):
        return min(1.0, epoch / 10) * VAE_KAPPA

    for epoch in range(AE_EPOCHS):
        # --- training step ---
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()

            if model_type == "VAE":
                recon, mu, logvar = model(batch)
                recon_loss = F.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                beta = kl_weight(epoch)
                loss = (recon_loss + beta * kl_loss) / batch.size(0)

            elif model_type == "DAE":
                recon = model(batch, corrupt=True)
                loss = F.mse_loss(recon, batch)

            else:  # AE
                recon = model(batch)
                loss = F.mse_loss(recon, batch)

            loss.backward()
            optimizer.step()

        # --- validation step ---
        model.eval()
        with torch.no_grad():
            if model_type == "VAE":
                recon, mu, logvar = model(val_t)
                recon_loss = F.mse_loss(recon, val_t, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                beta = kl_weight(epoch)
                val_loss = (recon_loss + beta * kl_loss) / val_t.size(0)
            elif model_type == "DAE":
                recon = model(val_t, corrupt=False)
                val_loss = F.mse_loss(recon, val_t)
            else:
                recon = model(val_t)
                val_loss = F.mse_loss(recon, val_t)

        val_loss_value = val_loss.item()

        # did the validation loss improve?
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # early stopping: no improvement for AE_PATIENCE consecutive epochs
        if patience_counter >= AE_PATIENCE and epoch >= 10:
            break

    # restore the best state
    model.load_state_dict(best_state)
    model.eval()

    return model


# ============================================================
# Latent representation of a trained model
# ============================================================

def extract_latent(model, X, model_type="AE"):
    """Latent representation z of a trained model.
    VAE: the posterior mean mu (deterministic encoding).
    AE / DAE: the encoder output.
    """
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        if model_type == "VAE":
            z = model.encode_mu(X_t)
        else:
            z = model.encode(X_t)

    return z.cpu().numpy()


def fit_pca_gpu(X_t, n_comp):
    """Randomised low-rank PCA on GPU, as used throughout (torch.pca_lowrank,
    niter=4). Returns (V: n_genes x n_comp, mean: 1 x n_genes)."""
    _, _, V_t = torch.pca_lowrank(X_t, q=n_comp, center=True, niter=4)
    return V_t, X_t.mean(0, keepdim=True)
