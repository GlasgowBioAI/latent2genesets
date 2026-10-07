# latent2genesets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)

Code for **"Linking latent dimensions to gene sets: the role of gene scoring in gene set
enrichment analysis of transcriptomic data"** (Zheng and Bryson).

Gene set enrichment analysis of a latent dimension first requires a score for every gene; we
refer to the method used to derive these scores as a *gene-level readout*. This repository
compares weight-based, correlation-based and attribution-based gene-level readouts across six
dimensionality-reduction methods, using over-representation analysis against
differential-expression-derived reference gene sets.

| | |
|---|---|
| **Methods** | PCA, ICA, NMF, autoencoder (AE), denoising AE (DAE), variational AE (VAE) |
| **Gene-level readouts** | weights (linear methods), Pearson, Spearman, Integrated Gradients, DeepLIFT, Input×Gradient |
| **Data** | BioBombe-processed TCGA pan-cancer and GTEx multi-tissue RNA-seq; 28 latent dimensionalities (2–200) |
| **Gene sets** | MSigDB v2023.2.Hs Hallmark and KEGG legacy |
| **Reference** | gene sets detected by differential expression (tumour vs normal, one vs rest) |
| **Metric** | AUROC of per-gene-set enrichment scores against the reference |
| **Replicates** | 3 model seeds (42, 43, 44) for every model fit; 10 gene-wise permutations for the null control |

Plotting code is not included; every figure is drawn from the tables this code writes.

---

## Contents

- [Installation](#installation)
- [Input data](#input-data)
- [Repository layout](#repository-layout)
- [Running the pipeline](#running-the-pipeline)
- [Replicates: seeds and permutations](#replicates-seeds-and-permutations)
- [Outputs](#outputs)
- [Method summary](#method-summary)
- [Notes](#notes)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/GlasgowBioAI/latent2genesets.git
cd latent2genesets

conda create -n latent2genesets python=3.12
conda activate latent2genesets
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for the AE/DAE/VAE fits and the attribution readouts.

## Input data

| Input | Source | Location |
|---|---|---|
| Expression matrices `{train,test}_{tcga,gtex}_expression_matrix_processed.tsv.gz`, `tcga_sample_identifiers.tsv`, `*_sample_counts.tsv` | [BioBombe](https://github.com/greenelab/BioBombe) `0.expression-download/data` — not redistributed | `data/biobombe/` ([instructions](data/biobombe/README.md)) |
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | [GTEx Portal](https://www.gtexportal.org/home/datasets) — not redistributed | `data/gtex/` ([instructions](data/gtex/README.md)) |
| Hallmark and KEGG legacy GMT files (Entrez) | [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp) — not redistributed | `data/gmt/` ([instructions](data/gmt/README.md)) |
| Clinical and covariate labels for the within-group case studies | cBioPortal / GTEx, derived | `data/labels/` (included, [provenance](data/labels/README.md)) |

Every path is relative to the repository root, so nothing needs to be configured once the
files are in the folders above. To keep the data or the results somewhere else, override the
defaults — a relative value is taken from the repository root:

```bash
export LATENT2GENESETS_BIOBOMBE_DATA=data/biobombe        # default
export LATENT2GENESETS_GTEX_SAMPLE_ATTR=data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt   # default
export LATENT2GENESETS_RESULTS=results                    # default
```

No results are shipped with the repository; `results/` is created when the pipeline is run.

## Repository layout

```text
latent2genesets/
├── config.py               paths, study design, model hyperparameters
├── latent2genesets/        shared library
│   ├── data.py             data loading, sample groups
│   ├── models.py           AE / DAE / VAE, training loop, GPU PCA
│   ├── readouts.py         weights, Pearson, Spearman, attribution readouts
│   ├── ora.py              query sets and Fisher over-representation analysis
│   ├── de.py               differential expression for the reference gene sets
│   └── evaluation.py       pooled DE reference
├── pipeline/
│   ├── 01_reference/       DE-derived reference gene sets
│   ├── 02_readouts/        methods × dimensionalities × readouts
│   ├── 03_evaluate/        AUROC, detection rate, reference-threshold sensitivity
│   ├── 04_phenotype/       within-group phenotype case studies
│   └── 05_replicates/      mean ± SD over model seeds and over permutations
├── controls/               permutation null, attribution aggregation
└── data/
    ├── biobombe/           expression matrices (download)
    ├── gtex/               GTEx sample annotations (download)
    ├── gmt/                gene sets (download)
    └── labels/             case-study labels (included)
```

## Running the pipeline

All scripts resolve their paths from their own location, so they can be run from any
working directory. Commands below are run from the repository root.

### 1 · Reference gene sets

```bash
python pipeline/01_reference/run_pairwise_de.py --dataset both
python pipeline/01_reference/run_onevsrest_de.py --dataset tcga
python pipeline/01_reference/run_onevsrest_de.py --dataset gtex
python pipeline/01_reference/run_tumor_normal_de.py
python pipeline/01_reference/aggregate_pairwise.py --dataset both --tag full
python pipeline/01_reference/build_reference.py
```

### 2 · Models and readouts

Fits every model at every dimensionality, computes all readouts on the test split and runs
ORA for each latent dimension.

```bash
python pipeline/02_readouts/run_readouts.py --dataset tcga
python pipeline/02_readouts/run_readouts.py --dataset gtex
```

### 3 · Evaluation

```bash
python pipeline/03_evaluate/auroc_per_dim.py       # AUROC per method × readout × dimensionality
python pipeline/03_evaluate/detection_rate.py      # detection rate per gene set
python pipeline/03_evaluate/auroc_recurrence.py    # sensitivity to the reference threshold
```

`auroc_recurrence.py` repeats the AUROC evaluation against stricter references: a gene set is
positive only if it is differentially expressed in at least *k* cancer types or tissues
(*k* = 1, 2, 3, 5; `config.RECURRENCE_K`). *k* = 1 is the primary analysis.

### 4 · Within-group phenotype case studies

Labels are already provided in `data/labels/`; the `fetch_*` / `build_gtex_labels` scripts
regenerate them.

```bash
python pipeline/04_phenotype/run_group_eval.py --dataset tcga --group COAD             # one call per group
python pipeline/04_phenotype/build_reference.py --dataset tcga --group COAD --label msi     # one call per contrast
python pipeline/04_phenotype/build_auroc.py
python pipeline/04_phenotype/build_detection_bh.py
```

| Dataset | Group | Contrast |
|---|---|---|
| TCGA | BRCA | ER, PR, HER2 status |
| TCGA | COAD | MSI status |
| TCGA | HNSC | HPV status |
| TCGA | THCA | BRAF p.V600 status |
| TCGA | LUAD | EGFR mutation status |
| GTEx | Muscle – Skeletal | donor age, Hardy death classification |
| GTEx | Artery – Tibial | donor age |

### Controls

```bash
# gene-wise permutation null: each gene permuted across samples, all models re-fit
python controls/permutation_null.py --dataset tcga --models PCA,ICA,NMF
python controls/permutation_null.py --dataset tcga --models AE,DAE,VAE
python controls/permutation_null.py --summarize

# signed-mean vs mean-absolute aggregation of attributions across samples
python controls/xai_aggregation.py --dataset tcga
python controls/xai_aggregation.py --summarize
```

Both controls accept `--dims` (comma-separated) and `--tag` to split long runs.

## Replicates: seeds and permutations

The reported results are replicated in two separate ways. They answer different questions
and are never pooled.

| | Model seeds | Permutations |
|---|---|---|
| What changes | model randomness: train/validation split, ICA/NMF initialisation and subsample, neural-network initialisation and batch order | the data: each gene is re-shuffled across samples |
| What stays | the data | the model seed (42) |
| How many | 3 — seeds 42, 43, 44 (`config.SEEDS`) | 10 — `--perm 0` … `--perm 9` (`config.N_PERMUTATIONS`) |
| Selected by | `LATENT2GENESETS_SEED=<n>` | `controls/permutation_null.py --perm <p>` |
| Output tag | none for 42, `_seed<n>` otherwise | none for 0, `_perm<p>` otherwise |
| Reported as | mean ± SD over seeds | mean ± SD over permutations |

The reference gene sets (step 1 and `04_phenotype/build_reference.py`) do not depend on the model fit;
they are computed once and shared by all seeds.

### Model seeds

Steps 2–4 above, run without `LATENT2GENESETS_SEED`, are seed 42. Repeat every step that
depends on the fitted models for the other two seeds:

```bash
for seed in 43 44; do
    export LATENT2GENESETS_SEED=$seed
    python pipeline/02_readouts/run_readouts.py --dataset tcga
    python pipeline/02_readouts/run_readouts.py --dataset gtex
    python pipeline/03_evaluate/auroc_per_dim.py
    python pipeline/03_evaluate/detection_rate.py
    python pipeline/03_evaluate/auroc_recurrence.py
    python pipeline/04_phenotype/run_group_eval.py --dataset tcga --group COAD   # one call per group
    python pipeline/04_phenotype/build_auroc.py
    python controls/xai_aggregation.py --dataset tcga
    python controls/xai_aggregation.py --dataset gtex
    python controls/xai_aggregation.py --summarize
done
unset LATENT2GENESETS_SEED
```

### Permutations

`--perm 0` is the default. Each further permutation re-shuffles the data and re-fits every model:

```bash
for p in 1 2 3 4 5 6 7 8 9; do
    for ds in tcga gtex; do
        python controls/permutation_null.py --dataset $ds --models PCA,ICA,NMF --perm $p
        python controls/permutation_null.py --dataset $ds --models AE,DAE,VAE --perm $p
    done
    python controls/permutation_null.py --summarize --perm $p
done
```

### Summary

```bash
python pipeline/05_replicates/summarize_replicates.py
```

writes the mean, the SD and the per-replicate values for the AUROC, its
sensitivity to the reference recurrence threshold, the case studies and the permutation null (see [Outputs](#outputs)). Each replicate is first reduced
as a single run is reported — the mean AUROC over the 28 dimensionalities — and the mean
and SD are then taken across replicates.

ICA is skipped at a dimensionality where FastICA does not converge. Such a cell is absent
from that replicate and is averaged over the replicates that have it; the counts are written
out (`n_seeds`, `min_dims` / `max_dims`). On permuted data ICA rarely converges (on TCGA,
1–4 of the 28 dimensionalities per permutation), so its permutation-null values rest on few
dimensionalities and vary more between permutations than those of any other model.

## Outputs

Everything is written under `results/` (or `$LATENT2GENESETS_RESULTS`).

| Folder | Main files |
|---|---|
| `reference/` | `final_reference.csv` — DE status of every gene set in every cancer type / tissue |
| `readouts/` | `readout_ora_{TCGA,GTEX}.csv` — ORA p-value per method × dimension × latent × readout × gene set |
| `evaluation/` | `per_dim_auroc_cells.csv` — AUROC per model × readout × dimension; `detection_*_BH.csv`; `auroc_seed_summary.csv` — mean ± SD over seeds; `per_dim_auroc_cells_seedmean.csv` — per-dimension mean over seeds; `auroc_recurrence_cells.csv` and `auroc_recurrence_seed_summary.csv` — AUROC by reference recurrence threshold *k*, per run and mean ± SD over seeds |
| `phenotype/` | `auroc_<group>_<label>.csv`, `detection_bh_<group>_<db>.csv`; `auroc_seed_summary.csv` — mean ± SD over seeds per case study |
| `controls/` | `permutation_null_cells.csv`, `xai_aggregation_cells.csv`; `permutation_null_summary.csv` — mean ± SD over permutations |

Files of seed 43 / 44 carry `_seed43` / `_seed44` before the extension, and files of
permutation *p* > 0 carry `_perm<p>`; the untagged files are seed 42 and permutation 0.

## Method summary

1. Expression is log2(x + 1) transformed and standardised on the training split, using the
   BioBombe train/test partitions.
2. Dimensionality-reduction methods are fitted without phenotype labels on the training split;
   readouts are computed on the test split. Reference gene sets use the training split only.
3. For each latent dimension, gene scores are z-scored and genes with |z| > 2 form the
   query set, which is tested against every gene set with a one-sided Fisher exact test.
4. At dimensionality *D*, a gene set is scored by −log10 of its minimum p-value over the
   *D* latent dimensions (Bonferroni × *D*).
5. AUROC is computed against the reference: a gene set is positive if it is differentially
   expressed (BH < 0.05, tumour vs normal or one vs rest) in at least one cancer type or tissue.

The ORA background defaults to all measured genes. Setting `ORA_UNIVERSE=annotated` restricts
it to the genes annotated in each gene-set collection and suffixes every output with `_annotU`.

## Notes

- **VAE decoder.** The decoder ends in a sigmoid, so its output lies in [0, 1], while the
  model is trained on standardised data. This configuration is kept to reproduce the reported
  results; see the `VAE` docstring in [`latent2genesets/models.py`](latent2genesets/models.py).
- **Randomness.** PCA uses randomised low-rank SVD and the neural networks train on GPU, so
  re-running one seed reproduces its results closely but not bit-for-bit; ICA and NMF are
  exact. Results are therefore reported as the mean ± SD over three seeds rather than from a
  single run.

## License

Code is released under the [MIT License](LICENSE). MSigDB gene sets are not included and
remain subject to their own terms of use.
