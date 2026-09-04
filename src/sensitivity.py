"""
Economic sensitivity analysis for the readmission cost-decision layer.

The single base-case answer in run_pipeline.py ("targeting saves ~$5k per 1,000
patients vs doing nothing") is only as good as three assumed numbers:
C_readmit, C_intervention, and RRR (relative risk reduction from the
intervention). This module asks the honest follow-up question: **for which
combinations of these inputs does model-guided targeting actually beat doing
nothing, and by how much?**

Method: for each (RRR, C_intervention) combination on a grid, recompute the
cost-optimal threshold p* = C_intervention / (RRR * C_readmit), then evaluate
REALISED savings (using actual observed outcomes, not the modelled expectation)
of the model-guided policy vs doing nothing, on the held-out test set. This
reuses the already-calibrated predicted probabilities; the model itself is not
retrained for each grid cell, only the decision threshold and policy costing.

C_readmit is held fixed at its base-case value; RRR and C_intervention are
varied because they are the two most uncertain and most decision-relevant
inputs (RRR because it is not estimated from this data at all, and
C_intervention because real programs vary enormously in cost).
"""

import numpy as np

import cost_analysis as ca


def savings_per_1000(p, y_true, rrr, c_intervention, c_readmit=ca.C_READMIT):
    """Realised savings of model-guided targeting vs no-intervention, per 1,000
    patients, for one (RRR, C_intervention) combination."""
    p = np.asarray(p)
    y = np.asarray(y_true)
    n = len(p)

    thr = ca.optimal_threshold(c_intervention, rrr, c_readmit)
    enrolled = p >= thr

    # realised cost, no intervention: every actual readmission costs C_readmit
    cost_none = (y * c_readmit).sum()

    # realised cost, model-guided: enrolled patients get the intervention
    # (cost C_intervention) and, if they were actually going to be readmitted,
    # a fraction (1-RRR) still are; non-enrolled patients follow the no-intervention cost
    cost_guided = (
        enrolled.sum() * c_intervention
        + (y[enrolled] * (1 - rrr) * c_readmit).sum()
        + (y[~enrolled] * c_readmit).sum()
    )

    savings = cost_none - cost_guided
    return {
        "threshold": thr,
        "n_flagged": int(enrolled.sum()),
        "pct_flagged": float(enrolled.mean()),
        "savings_total": float(savings),
        "savings_per_1000": float(savings / n * 1000),
    }


def sensitivity_grid(p, y_true, rrr_grid, c_intervention_grid,
                     c_readmit=ca.C_READMIT):
    """
    Build a 2D grid of realised savings-per-1,000 over RRR x C_intervention.

    Returns:
        savings_grid : (len(rrr_grid), len(c_intervention_grid)) array
        flagged_grid : same shape, % of cohort flagged at that combination
    """
    savings_grid = np.zeros((len(rrr_grid), len(c_intervention_grid)))
    flagged_grid = np.zeros_like(savings_grid)

    for i, rrr in enumerate(rrr_grid):
        for j, c_int in enumerate(c_intervention_grid):
            res = savings_per_1000(p, y_true, rrr, c_int, c_readmit)
            savings_grid[i, j] = res["savings_per_1000"]
            flagged_grid[i, j] = res["pct_flagged"] * 100

    return savings_grid, flagged_grid


def breakeven_curve(p, y_true, rrr_grid, c_readmit=ca.C_READMIT,
                    c_int_bounds=(50, 5000), tol=1.0):
    """
    For each RRR, find the C_intervention at which savings-per-1,000 crosses
    zero (binary search). Points above this curve (higher C_intervention) mean
    targeting no longer beats doing nothing.
    """
    lo_bound, hi_bound = c_int_bounds
    breakevens = []
    for rrr in rrr_grid:
        lo, hi = lo_bound, hi_bound
        s_lo = savings_per_1000(p, y_true, rrr, lo, c_readmit)["savings_per_1000"]
        s_hi = savings_per_1000(p, y_true, rrr, hi, c_readmit)["savings_per_1000"]
        if s_lo < 0:
            # even the cheapest intervention doesn't pay off at this RRR
            breakevens.append(np.nan)
            continue
        if s_hi > 0:
            # even the most expensive intervention in range still pays off
            breakevens.append(hi_bound)
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            s_mid = savings_per_1000(p, y_true, rrr, mid, c_readmit)["savings_per_1000"]
            if abs(s_mid) < tol:
                break
            if s_mid > 0:
                lo = mid
            else:
                hi = mid
        breakevens.append(0.5 * (lo + hi))
    return np.array(breakevens)
