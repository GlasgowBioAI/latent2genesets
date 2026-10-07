"""
build_gtex_labels.py -- builds the GTEx within-tissue labels (AGE, DTHHRDY)
from GTEx's public annotation files:

  - GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt (SUBJID/AGE/DTHHRDY,
    public download, no dbGaP) -- downloaded by this script.
  - GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt (config.GTEX_SAMPLE_ATTR).

Joined to the GTEx sample cohort (BioBombe train+test matrices) by donor ID
(SAMPID's first two dash-separated fields, e.g. "GTEX-1117F").

Two label files, one row per classifiable sample (unclassifiable samples are
absent; build_reference.py treats "not in the lookup" as excluded):

  gtex_age.csv       -- age_binary: Young/Old, AGE bracket 20-29/30-39 vs
                        60-69/70-79 only (40-49/50-59 dropped)
  gtex_dthhrdy.csv   -- dthhrdy_binary: Fast/Slow, Hardy scale 0/1 (violent
                        or ventilator/fast death) vs 4 (slow death from
                        illness) only (2/3 dropped)

Usage:
    python build_gtex_labels.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repository root: config + latent2genesets

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from config import LABELS_DIR, BIOBOMBE_DATA
OUT = LABELS_DIR
OUT.mkdir(parents=True, exist_ok=True)

SUBJ_URL = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

YOUNG = {"20-29", "30-39"}
OLD = {"60-69", "70-79"}
DTH_FAST = {0, 1}
DTH_SLOW = {4}

def project_gtex_samples():
    train = pd.read_csv(BIOBOMBE_DATA / "train_gtex_expression_matrix_processed.tsv.gz",
                         sep="\t", index_col=0, usecols=[0])
    test = pd.read_csv(BIOBOMBE_DATA / "test_gtex_expression_matrix_processed.tsv.gz",
                        sep="\t", index_col=0, usecols=[0])
    return list(train.index) + list(test.index)


def main():
    samples = project_gtex_samples()
    print(f"project GTEx cohort: {len(samples)} samples")

    subj = pd.read_csv(SUBJ_URL, sep="\t")
    age_map = dict(zip(subj.SUBJID, subj.AGE))
    dth_map = dict(zip(subj.SUBJID, subj.DTHHRDY))

    donors = {s: "-".join(s.split("-")[:2]) for s in samples}

    # ---- AGE ----
    rows = []
    for s in samples:
        v = age_map.get(donors[s])
        if v in YOUNG:
            rows.append({"sample_id": s, "age_binary": "Young"})
        elif v in OLD:
            rows.append({"sample_id": s, "age_binary": "Old"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gtex_age.csv", index=False)
    print(f"gtex_age.csv: {len(df)} rows -- "
          f"{(df.age_binary=='Young').sum()} Young / {(df.age_binary=='Old').sum()} Old")

    # ---- DTHHRDY ----
    rows = []
    for s in samples:
        v = dth_map.get(donors[s])
        if v in DTH_FAST:
            rows.append({"sample_id": s, "dthhrdy_binary": "Fast"})
        elif v in DTH_SLOW:
            rows.append({"sample_id": s, "dthhrdy_binary": "Slow"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gtex_dthhrdy.csv", index=False)
    print(f"gtex_dthhrdy.csv: {len(df)} rows -- "
          f"{(df.dthhrdy_binary=='Fast').sum()} Fast / {(df.dthhrdy_binary=='Slow').sum()} Slow")


if __name__ == "__main__":
    main()
