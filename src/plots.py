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


def sensitivity_heatmap(savings_grid, rrr_grid, c_int_grid, path):
    """
    Savings are floored at $0 by construction (when the cost-optimal threshold
    exceeds the model's achievable probabilities, the policy enrols nobody,
    which is identical in cost to no intervention — never worse). So this is a
    one-sided quantity: a sequential colormap is used, not diverging, and a
    power-law normalisation is used to keep the region near breakeven visible
    despite a few large values in the cheap/high-RRR corner.
    """
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = savings_grid.max()
    norm = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=vmax)
    im = ax.imshow(savings_grid, aspect="auto", origin="lower",
                   cmap="Blues", norm=norm,
                   extent=[c_int_grid[0], c_int_grid[-1], rrr_grid[0], rrr_grid[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Savings vs no-intervention ($ per 1,000 patients)")
    cbar.set_ticks([0, vmax * 0.05, vmax * 0.2, vmax * 0.5, vmax])
    cbar.set_ticklabels([f"{t:,.0f}" for t in
                         [0, vmax * 0.05, vmax * 0.2, vmax * 0.5, vmax]])

    # breakeven contour: savings == 0 (i.e. the policy enrols nobody beyond this line)
    C, R = np.meshgrid(c_int_grid, rrr_grid)
    cs = ax.contour(C, R, savings_grid, levels=[1e-6], colors="black", linewidths=2)
    ax.clabel(cs, fmt="breakeven", fontsize=9)

    ax.scatter([1200], [0.25], color="red", marker="*", s=250, zorder=5,
              edgecolors="black", linewidths=0.5,
              label="Base case ($1,200, RRR=0.25)")
    ax.set_xlabel("Intervention cost, $")
    ax.set_ylabel("Relative risk reduction (RRR)")
    ax.set_title("Sensitivity: savings from targeting vs no intervention")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def breakeven_plot(rrr_grid, breakeven_costs, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rrr_grid, breakeven_costs, color=BLUE, lw=2)
    ax.fill_between(rrr_grid, 0, breakeven_costs, color=BLUE, alpha=0.15,
                    label="Targeting beats doing nothing")
    ax.scatter([0.25], [1200], color=RED, marker="*", s=200, zorder=5,
              label="Base case assumption")
    ax.set_xlabel("Relative risk reduction (RRR)")
    ax.set_ylabel("Break-even intervention cost, $")
    ax.set_title("Maximum affordable intervention cost, by effect size")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


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
