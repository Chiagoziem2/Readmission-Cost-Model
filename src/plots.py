"""Figures: ROC + PR, calibration curves, and realised cost vs threshold."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

BLUE, RED, GREY = "#2c6fbb", "#c0392b", "#7f8c8d"


def roc_pr(results, y_true, path):
    """results: dict name -> predicted prob."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for name, p in results.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax1.plot(fpr, tpr, lw=2, label=name)
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax2.plot(rec, prec, lw=2, label=name)
    ax1.plot([0, 1], [0, 1], ls="--", color=GREY)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC"); ax1.legend()
    base = y_true.mean()
    ax2.axhline(base, ls="--", color=GREY, label=f"prevalence = {base:.2f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def calibration(results, y_true, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="perfect")
    for name, p in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction readmitted")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def cost_curve(grid, costs, opt_thr, path, n):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, costs / n, lw=2, color=BLUE, label="realised cost / patient")
    ax.axvline(opt_thr, color=RED, ls="--",
               label=f"cost-optimal threshold = {opt_thr:.2f}")
    # treat-none = cost at threshold 1.0 (nobody enrolled); treat-all at 0.0
    ax.scatter([1.0], [costs[-1] / n], color="black", zorder=5)
    ax.annotate("treat none", (1.0, costs[-1] / n),
                textcoords="offset points", xytext=(-60, 0), fontsize=9)
    ax.set_xlabel("Enrolment threshold on predicted probability")
    ax.set_ylabel("Realised expected cost per patient ($)")
    ax.set_title("Cost vs decision threshold (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
