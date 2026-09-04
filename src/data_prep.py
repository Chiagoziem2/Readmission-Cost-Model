"""
Data preparation for the diabetes 130-hospitals readmission task.

Key decisions (each defensible and documented):

1. TARGET: binary — readmitted within 30 days (`<30`) vs not (`>30` or `NO`).
   The 30-day window is the one that carries financial/regulatory weight
   (US CMS Hospital Readmissions Reduction Program).

2. EXCLUDE encounters ending in death or hospice (discharge_disposition_id in
   {11,13,14,19,20,21}). These patients CANNOT be readmitted, so leaving them in
   pollutes the negative class and inflates apparent performance. This follows
   the original Strack et al. (2014) analysis.

3. PATIENT-LEVEL SPLIT. The same patient_nbr appears in multiple encounters
   (71.5k patients / 101.8k encounters). A random row split leaks a patient into
   both train and test. We split by patient so no patient crosses the boundary.
   (A true *temporal* split is not possible — the dataset has no admission dates
   — so patient-grouped splitting is the correct available safeguard.)

4. ICD-9 DIAGNOSIS GROUPING. diag_1/2/3 have ~700+ codes each. We map them to
   9 clinically standard categories (circulatory, respiratory, ... , diabetes,
   other), the grouping used in the source literature.

All modelling avoids leakage columns: encounter_id and patient_nbr are dropped
from features (patient_nbr is used only to define the split).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEATH_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DROP_COLS = [
    "encounter_id",     # identifier
    "weight",           # ~97% missing
    "payer_code",       # ~40% missing, administrative, not clinically predictive here
]

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def _icd9_group(code):
    """Map a single ICD-9 diagnosis code to a broad category."""
    if pd.isna(code) or code == "?":
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 390 <= num <= 459 or int(num) == 785:
        return "Circulatory"
    if 460 <= num <= 519 or int(num) == 786:
        return "Respiratory"
    if 520 <= num <= 579 or int(num) == 787:
        return "Digestive"
    if int(num) == 250:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or int(num) == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def load_and_clean(path):
    df = pd.read_csv(path)

    # 1. remove death/hospice encounters (cannot be readmitted)
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_DISPOSITIONS)].copy()

    # 2. drop invalid gender rows (a few 'Unknown/Invalid')
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    # 3. binary target
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 4. ICD-9 grouping
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c + "_grp"] = df[c].apply(_icd9_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # 5. engineered features
    df["n_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    change_map = {"No": 0, "Steady": 0, "Up": 1, "Down": 1}
    df["n_med_changes"] = df[DRUG_COLS].apply(
        lambda col: col.map(change_map)
    ).sum(axis=1)
    df["on_insulin"] = (df["insulin"] != "No").astype(int)

    # 6. tidy missing markers
    df["race"] = df["race"].replace("?", "Missing")
    df["medical_specialty"] = df["medical_specialty"].replace("?", "Missing")

    # 7. drop unhelpful / leakage columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # keep patient_nbr for splitting; it is removed from the feature matrix later
    return df


def make_feature_lists(df):
    """Return (numeric_cols, categorical_cols) excluding target and id."""
    exclude = {"target", "patient_nbr"}
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def patient_level_split(df, test_size=0.2, seed=42):
    """Split so that no patient_nbr appears in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    # sanity: zero patient overlap
    overlap = set(train["patient_nbr"]) & set(test["patient_nbr"])
    assert not overlap, "Patient leakage between train and test!"
    return train, test
