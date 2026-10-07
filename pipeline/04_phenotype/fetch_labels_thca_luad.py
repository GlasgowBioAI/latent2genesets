"""
fetch_labels_thca_luad.py -- data/labels/ pull for the THCA BRAF and LUAD
EGFR case studies from cBioPortal (joined to the TCGA sample cohort by
15-character sample barcode, one row per cohort sample, no imputation).

Labels:
  THCA-BRAF -- thca_braf.csv  (source: thca_tcga_pan_can_atlas_2018 mutations
               endpoint, gene BRAF/entrez 673, restricted to the
               thca_tcga_pan_can_atlas_2018_sequenced sample list:
               braf_binary = "Mutant" if the sample has a BRAF p.V600 hotspot
               mutation, "WT" if sequenced with no BRAF mutation at all, NaN
               (dropped) if some other non-hotspot BRAF mutation. Papillary
               thyroid carcinoma's BRAF V600E "BRAF-like" transcriptional
               signature is one of TCGA's own flagship subtype calls (Cancer
               Cell 2014).

  LUAD-EGFR -- luad_egfr.csv  (source: luad_tcga_pan_can_atlas_2018 mutations
               endpoint, gene EGFR/entrez 1956, restricted to the
               luad_tcga_pan_can_atlas_2018_sequenced sample list. Unlike
               BRAF in melanoma, EGFR in lung adenocarcinoma doesn't have one
               dominant hotspot codon -- the two classic sensitizing classes
               (exon19 in-frame deletions and the exon21 L858R point mutant)
               together cover ~85-90% of clinically-actionable EGFR-mutant
               NSCLC; here any reported EGFR mutation is treated as "Mutant": egfr_binary = "Mutant" if the sample has ANY
               EGFR mutation call, "WT" if sequenced with none.

Usage:
    python fetch_labels_thca_luad.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_labels import OUT, get, cohort  # noqa: E402


def fetch_thca_braf():
    study = "thca_tcga_pan_can_atlas_2018"
    samples = cohort("THCA")
    sequenced = set(get(f"/sample-lists/{study}_sequenced")["sampleIds"])
    muts = get(f"/molecular-profiles/{study}_mutations/mutations"
               f"?sampleListId={study}_sequenced&entrezGeneId=673&projection=SUMMARY")
    v600, other = {}, set()
    for m in muts:
        pc = m.get("proteinChange") or ""
        if pc.startswith("V600"):
            v600[m["sampleId"]] = pc
        else:
            other.add(m["sampleId"])

    rows = []
    for s in samples:
        if s in v600:
            status, protein = "Mutant", v600[s]
        elif s in other:
            status, protein = None, None
        elif s in sequenced:
            status, protein = "WT", None
        else:
            status, protein = None, None
        rows.append({"sample_id": s, "BRAF_PROTEIN_CHANGE": protein, "braf_binary": status})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "thca_braf.csv", index=False)
    print(f"thca_braf.csv: {len(df)} rows, {df.braf_binary.value_counts().to_dict()}")


def fetch_luad_egfr():
    study = "luad_tcga_pan_can_atlas_2018"
    samples = cohort("LUAD")
    sequenced = set(get(f"/sample-lists/{study}_sequenced")["sampleIds"])
    muts = get(f"/molecular-profiles/{study}_mutations/mutations"
               f"?sampleListId={study}_sequenced&entrezGeneId=1956&projection=SUMMARY")
    mutated = {m["sampleId"]: (m.get("proteinChange") or "") for m in muts}

    rows = []
    for s in samples:
        if s in mutated:
            status, protein = "Mutant", mutated[s]
        elif s in sequenced:
            status, protein = "WT", None
        else:
            status, protein = None, None
        rows.append({"sample_id": s, "EGFR_PROTEIN_CHANGE": protein, "egfr_binary": status})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "luad_egfr.csv", index=False)
    print(f"luad_egfr.csv: {len(df)} rows, {df.egfr_binary.value_counts().to_dict()}")


if __name__ == "__main__":
    fetch_thca_braf()
    fetch_luad_egfr()
