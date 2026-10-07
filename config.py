"""
config.py -- every path and constant the pipeline uses, in one place.

Every path is relative to the repository root. The defaults can be overridden with
environment variables (see README); a relative value is also taken from the repository
root, so the scripts behave the same from any working directory:
  LATENT2GENESETS_BIOBOMBE_DATA     BioBombe processed TCGA/GTEx matrices + sample tables
                                    (default data/biobombe)
  LATENT2GENESETS_GTEX_SAMPLE_ATTR  GTEx v8 SampleAttributesDS.txt (tissue labels)
                                    (default data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt)
  LATENT2GENESETS_RESULTS           root of every output directory below (default results)
  LATENT2GENESETS_SEED              model seed of this run (default 42); see "replicates" below
  ORA_UNIVERSE                      "all" (default, as in the paper figures) or "annotated";
                                    read by latent2genesets/ora.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _path(env, default):
    """Path from environment variable `env`, else `default`; relative -> under ROOT."""
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- inputs
DATA = ROOT / "data"
BIOBOMBE_DATA = _path("LATENT2GENESETS_BIOBOMBE_DATA", "data/biobombe")
GTEX_SAMPLE_ATTR = _path("LATENT2GENESETS_GTEX_SAMPLE_ATTR",
                         "data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt")
GMT_DIR = DATA / "gmt"
GMTS = {
    "hallmark": GMT_DIR / "h.all.v2023.2.Hs.entrez.gmt",
    "kegg": GMT_DIR / "c2.cp.kegg_legacy.v2023.2.Hs.entrez.gmt",
}
LABELS_DIR = DATA / "labels"          # external clinical / GTEx covariate labels (within-group case studies)

# ---------------------------------------------------------------- outputs
RESULTS = _path("LATENT2GENESETS_RESULTS", "results")
REFERENCE_DIR = RESULTS / "reference"    # DE-derived reference gene sets (pipeline/01)
READOUT_DIR = RESULTS / "readouts"      # per-latent ORA of every model x readout (pipeline/02)
EVALUATION_DIR = RESULTS / "evaluation"  # AUROC / detection rate (pipeline/03)
PHENOTYPE_DIR = RESULTS / "phenotype"    # within-group label case studies (pipeline/04)
CONTROLS_DIR = RESULTS / "controls"      # permutation null, XAI aggregation (controls/)
for _d in (REFERENCE_DIR, READOUT_DIR, EVALUATION_DIR, PHENOTYPE_DIR, CONTROLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- replicates
# Model seeds: the same data, different model randomness (train/validation split, ICA/NMF
# initialisation and subsample, neural-network initialisation and batch order). The first
# seed is the primary run and its files carry no tag; any other seed tags every output
# that depends on the fitted models with _seed<n>. The DE reference does not depend on the
# fit and is shared by all seeds.
SEEDS = [42, 43, 44]
SEED = int(os.environ.get("LATENT2GENESETS_SEED", SEEDS[0]))
# Gene-wise permutations of the null control (controls/permutation_null.py --perm p,
# p = 0 .. N_PERMUTATIONS - 1). A permutation changes the DATA, not the model seed.
N_PERMUTATIONS = 10


def seed_tag(seed=SEED):
    """Filename tag of a model seed: "" for the primary seed, else "_seed<n>"."""
    return "" if seed == SEEDS[0] else f"_seed{seed}"


def perm_tag(perm):
    """Filename tag of a permutation replicate: "" for p = 0, else "_perm<p>"."""
    return f"_perm{perm}" if perm else ""


SEED_SUFFIX = seed_tag()


# ---------------------------------------------------------------- per-latent ORA files
# Earlier runs wrote these under older names; a file under the old name is used
# (read, resumed or skipped) whenever no file exists under the current name.
def _current_or_legacy(current, legacy):
    return current if current.exists() or not legacy.exists() else legacy


def readout_ora_path(dataset, suffix=""):
    """results/readouts/readout_ora_<DS><suffix>.csv (legacy: partial_S2_<DS>_BASELINE_ORA<suffix>.csv)."""
    ds = dataset.upper()
    return _current_or_legacy(READOUT_DIR / f"readout_ora_{ds}{suffix}.csv",
                              READOUT_DIR / f"partial_S2_{ds}_BASELINE_ORA{suffix}.csv")


def group_readout_ora_path(tg, db, suffix=""):
    """results/phenotype/group_readout_ora_<TAG>_<db><suffix>.csv (legacy: unsup_<TAG>_<db><suffix>.csv)."""
    return _current_or_legacy(PHENOTYPE_DIR / f"group_readout_ora_{tg}_{db}{suffix}.csv",
                              PHENOTYPE_DIR / f"unsup_{tg}_{db}{suffix}.csv")

# ---------------------------------------------------------------- study design
# BioBombe's 28-value latent-dimension sweep (z_parameter_sweep_*.tsv; identical for TCGA/GTEx)
DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50,
              60, 70, 80, 90, 100, 125, 150, 200]
GSEA_MIN_SET_SIZE = 15   # gene sets kept if 15-500 genes after intersection with the matrix
GSEA_MAX_SET_SIZE = 500
# Reference recurrence thresholds for the sensitivity analysis: a gene set is positive if it
# is differentially expressed in at least k cancer types / tissues. k = 1 is the primary analysis.
RECURRENCE_K = [1, 2, 3, 5]

# ---------------------------------------------------------------- AE / DAE / VAE training
AE_LR = 5e-4
AE_BATCH_SIZE = 512
AE_EPOCHS = 500
AE_PATIENCE = 10
DAE_NOISE_PROB = 0.3
VAE_KAPPA = 0.05
