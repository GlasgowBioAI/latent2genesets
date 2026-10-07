# BioBombe expression matrices (not included)

Download the processed matrices and sample tables from
[BioBombe](https://github.com/greenelab/BioBombe) (`0.expression-download/data`) and place
them in this folder:

| File | Content |
|---|---|
| `train_tcga_expression_matrix_processed.tsv.gz`, `test_tcga_expression_matrix_processed.tsv.gz` | TCGA pan-cancer, BioBombe train / test partition |
| `train_gtex_expression_matrix_processed.tsv.gz`, `test_gtex_expression_matrix_processed.tsv.gz` | GTEx multi-tissue, BioBombe train / test partition |
| `tcga_sample_identifiers.tsv` | TCGA sample -> cancer type and sample type |
| `tcga_sample_counts.tsv`, `gtex_sample_counts.tsv` | samples per cancer type / tissue |
| `tcga_mad_genes.tsv`, `gtex_mad_genes.tsv` | genes ranked by median absolute deviation (only read when a gene subset is requested) |

This is the default location (`config.BIOBOMBE_DATA`). To keep the files elsewhere, set
`LATENT2GENESETS_BIOBOMBE_DATA` to that folder.
