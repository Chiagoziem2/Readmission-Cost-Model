"""
Two models, deliberately:

  - Logistic regression: interpretable baseline. Reasonable probabilities.
  - HistGradientBoosting: stronger non-linear model, native categorical support.

Design choices tied to the downstream cost decision:
  - We DO NOT resample (no SMOTE/undersampling). Resampling distorts the
    predicted probabilities, and the cost layer needs *calibrated* probabilities,
    not just a good ranking. Imbalance is handled at the decision stage via a
    cost-sensitive threshold, not by rebalancing the data.
  - Probabilities are calibrated on a held-out slice (see calibration.py).
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_logistic(numeric_cols, categorical_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
         categorical_cols),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_gbm(numeric_cols, categorical_cols):
    # ordinal-encode categoricals and tell HGB they are categorical
    cat_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                              encoded_missing_value=-1)
    pre = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    n_num = len(numeric_cols)
    n_cat = len(categorical_cols)
    categorical_mask = [False] * n_num + [True] * n_cat
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
