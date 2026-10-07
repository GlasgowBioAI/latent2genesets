"""
fetch_labels.py -- pulls the HNSC HPV label (data/labels/hnsc_hpv.csv) from
cBioPortal's public REST API: study hnsc_tcga_pub, HPV_STATUS (PATIENT-level,
joined to the TCGA sample cohort by 12-character patient barcode, so the same
call is applied to every sample of that patient). One row per cohort sample
(NaN where the source study has no call), no imputation.

The THCA BRAF and LUAD EGFR labels are pulled by fetch_labels_thca_luad.py;
the BRCA ER/PR/HER2 and COAD MSI labels are documented in data/labels/README.md.

Usage:
    python fetch_labels.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
from config import LABELS_DIR, BIOBOMBE_DATA
OUT = LABELS_DIR
API = "https://www.cbioportal.org/api"
COHORT_TSV = BIOBOMBE_DATA / "tcga_sample_identifiers.tsv"


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as r:
        return json.load(r)


def cohort(cancer_type):
    ids = pd.read_csv(COHORT_TSV, sep="\t")
    return ids[ids.cancer_type == cancer_type]["sample_id"].tolist()


def fetch_hnsc_hpv():
    samples = cohort("HNSC")
    data = get("/studies/hnsc_tcga_pub/clinical-data?clinicalDataType=PATIENT&pageSize=20000")
    hpv = {d["patientId"]: d["value"] for d in data if d["clinicalAttributeId"] == "HPV_STATUS"}
    rows = [{"sample_id": s, "HPV_STATUS": hpv.get(s[:12])} for s in samples]
    df = pd.DataFrame(rows)
    df["hpv_binary"] = df["HPV_STATUS"]  # already "HPV+" / "HPV-"
    df.to_csv(OUT / "hnsc_hpv.csv", index=False)
    print(f"hnsc_hpv.csv: {len(df)} rows, {df.hpv_binary.value_counts().to_dict()}")


if __name__ == "__main__":
    fetch_hnsc_hpv()
