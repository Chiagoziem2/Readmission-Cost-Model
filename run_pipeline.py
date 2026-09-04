"""
End-to-end pipeline:
  load -> clean/engineer -> patient-level split -> train LR + GBM ->
  calibrate -> evaluate (discrimination + calibration) -> cost decision layer.

Usage:
    python run_pipeline.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_prep import (load_and_clean, make_feature_lists, patient_level_split)  # noqa
from models import build_logistic, build_gbm  # noqa
import cost_analysis as ca  # noqa
import plots  # noqa

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "diabetic_data.csv")
FIGS = os.path.join(HERE, "figures"); os.makedirs(FIGS, exist_ok=True)


def fit_calibrated(pipeline, X_tr, y_tr, groups_tr):
    """Calibrate on a held-out slice of the training set (patient-grouped)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    fit_idx, cal_idx = next(gss.split(X_tr, y_tr, groups=groups_tr))
    pipeline.fit(X_tr.iloc[fit_idx], y_tr.iloc[fit_idx])
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
    calibrated.fit(X_tr.iloc[cal_idx], y_tr.iloc[cal_idx])
    return calibrated


def main():
    print("Loading & cleaning ...")
    df = load_and_clean(DATA)
    print(f"  after cleaning: {df.shape[0]:,} encounters, "
          f"{df['patient_nbr'].nunique():,} patients, "
          f"positive rate {df['target'].mean():.3f}")

    train, test = patient_level_split(df, test_size=0.2, seed=42)
    numeric_cols, categorical_cols = make_feature_lists(df)
    feat = numeric_cols + categorical_cols

    X_tr, y_tr, g_tr = train[feat], train["target"], train["patient_nbr"]
    X_te, y_te = test[feat], test["target"]
    print(f"  train {len(train):,} / test {len(test):,} encounters "
          f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical features)")

    preds, metrics = {}, {}
    for name, builder in [("Logistic regression", build_logistic),
                          ("Gradient boosting", build_gbm)]:
        print(f"Training: {name} ...")
        model = fit_calibrated(builder(numeric_cols, categorical_cols),
                               X_tr, y_tr, g_tr)
        p = model.predict_proba(X_te)[:, 1]
        preds[name] = p
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_te, p)),
            "pr_auc": float(average_precision_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
        }
        print(f"  ROC-AUC={metrics[name]['roc_auc']:.3f}  "
              f"PR-AUC={metrics[name]['pr_auc']:.3f}  "
              f"Brier={metrics[name]['brier']:.4f}")

    # pick the better model by PR-AUC (appropriate for imbalance) for the cost layer
    best = max(preds, key=lambda k: metrics[k]["pr_auc"])
    p_best = preds[best]
    print(f"\nCost decision layer using: {best}")

    pol = ca.evaluate_policies(p_best)
    print(f"  cost-optimal threshold p* = {pol['threshold']:.3f}")
    print(f"  flagged {pol['n_flagged']:,} / {pol['n_patients']:,} test patients")
    print(f"  expected cost — no intervention: ${pol['cost_no_intervention']:,.0f}")
    print(f"  expected cost — treat all:       ${pol['cost_treat_all']:,.0f}")
    print(f"  expected cost — model-guided:    ${pol['cost_model_guided']:,.0f}")
    print(f"  savings vs no intervention: ${pol['savings_vs_none']:,.0f} "
          f"(${pol['savings_per_1000']:,.0f} per 1,000 patients)")
    print(f"  savings vs treat-all:       ${pol['savings_vs_treat_all']:,.0f}")

    grid, costs = ca.cost_vs_threshold(p_best, y_te)

    # persist test-set predictions + outcomes for downstream analyses (e.g. sensitivity)
    pd.DataFrame({"y_true": y_te.values, "p": p_best}).to_csv(
        os.path.join(HERE, "test_predictions.csv"), index=False)

    # figures
    plots.roc_pr(preds, y_te.values, os.path.join(FIGS, "roc_pr.png"))
    plots.calibration(preds, y_te.values, os.path.join(FIGS, "calibration.png"))
    plots.cost_curve(grid, costs, pol["threshold"],
                     os.path.join(FIGS, "cost_threshold.png"), len(y_te))

    # persist metrics
    out = {"metrics": metrics, "cost_policy": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                               for k, v in pol.items()},
           "economic_inputs": {"C_readmit": ca.C_READMIT,
                               "C_intervention": ca.C_INTERVENTION, "RRR": ca.RRR}}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nFigures -> figures/ | metrics -> results.json")


if __name__ == "__main__":
    main()
