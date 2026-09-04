"""
The cost-decision layer — the part that turns a classifier into a resource
allocation tool, and the reason this project has a financial angle.

Setup: a preventive intervention (e.g. a transitional-care / follow-up program)
can be offered to patients flagged as high readmission risk. For a patient with
(calibrated) readmission probability p:

    expected cost if NOT enrolled : p * C_readmit
    expected cost if enrolled     : C_intervention + p * (1 - RRR) * C_readmit

Enrolling is worthwhile when the second is smaller, i.e.

    C_intervention < p * RRR * C_readmit
    => p > C_intervention / (RRR * C_readmit)   ==   p*  (the optimal threshold)

So the cost-optimal decision threshold is a CLOSED FORM in the economic inputs —
no need to sweep. The classifier's job is purely to *target*: identify which
patients clear p*. Good targeting is what the ML adds value on.

>>> HONESTY <<<
- C_readmit, C_intervention and RRR are ILLUSTRATIVE. Real values must be sourced
  (reference costs for C_readmit; program budgets for C_intervention; an RCT /
  meta-analysis for RRR — transitional-care trials suggest ~15-25% relative
  reduction, but this is NOT estimable from this observational dataset and must
  come from external evidence).
- Because expected cost is linear in p, MISCALIBRATED probabilities bias the cost
  estimates directly. That is why calibration (calibration.py) is not optional
  window-dressing here — it feeds straight into the money.
"""

import numpy as np

# ---- illustrative economic inputs (USD) ------------------------------------
C_READMIT = 15000.0        # cost of a 30-day readmission
C_INTERVENTION = 1200.0    # cost of enrolling one patient in the program
RRR = 0.25                 # relative risk reduction from the intervention


def optimal_threshold(c_intervention=C_INTERVENTION, rrr=RRR, c_readmit=C_READMIT):
    return c_intervention / (rrr * c_readmit)


def policy_expected_cost(p, enrolled, c_intervention=C_INTERVENTION,
                         rrr=RRR, c_readmit=C_READMIT):
    """Total expected cost over a cohort given a boolean enrolment vector."""
    base = p * c_readmit
    treated = c_intervention + p * (1 - rrr) * c_readmit
    return np.where(enrolled, treated, base).sum()


def evaluate_policies(p, c_intervention=C_INTERVENTION, rrr=RRR,
                      c_readmit=C_READMIT):
    """Compare no-intervention, treat-all, and model-guided policies."""
    p = np.asarray(p)
    n = len(p)
    thr = optimal_threshold(c_intervention, rrr, c_readmit)

    none = policy_expected_cost(p, np.zeros(n, bool), c_intervention, rrr, c_readmit)
    all_ = policy_expected_cost(p, np.ones(n, bool), c_intervention, rrr, c_readmit)
    guided_mask = p > thr
    guided = policy_expected_cost(p, guided_mask, c_intervention, rrr, c_readmit)

    return {
        "threshold": thr,
        "n_patients": n,
        "n_flagged": int(guided_mask.sum()),
        "cost_no_intervention": none,
        "cost_treat_all": all_,
        "cost_model_guided": guided,
        "savings_vs_none": none - guided,
        "savings_vs_treat_all": all_ - guided,
        "savings_per_1000": (none - guided) / n * 1000,
    }


def cost_vs_threshold(p, y_true, grid=None, c_intervention=C_INTERVENTION,
                      rrr=RRR, c_readmit=C_READMIT):
    """
    REALISED cost across candidate thresholds using observed outcomes y_true
    (not the modelled expected cost). This is the honest check: does enrolling by
    predicted risk actually reduce realised cost? Assumes the intervention
    prevents a fraction RRR of readmissions among the enrolled.
    """
    p = np.asarray(p); y = np.asarray(y_true)
    if grid is None:
        grid = np.linspace(0, 1, 201)
    costs = []
    for thr in grid:
        enrolled = p >= thr
        # realised readmissions among not-enrolled: full cost
        cost = (y[~enrolled] * c_readmit).sum()
        # enrolled: program cost + residual readmissions (fraction (1-RRR) still occur)
        cost += enrolled.sum() * c_intervention
        cost += (y[enrolled] * (1 - rrr) * c_readmit).sum()
        costs.append(cost)
    return grid, np.array(costs)
