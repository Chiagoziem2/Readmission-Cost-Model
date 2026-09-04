"""
Economic sensitivity analysis for the readmission cost-decision layer.

Answers the question the base-case result (run_pipeline.py) cannot: for which
combinations of intervention cost and effect size (RRR) does model-guided
targeting actually beat doing nothing, and by how much? Uses the same
calibrated test-set predictions already produced by run_pipeline.py.

Requires: run_pipeline.py has already been run (produces test_predictions.csv).

Usage:
    python run_sensitivity.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cost_analysis as ca  # noqa: E402
import sensitivity as sens  # noqa: E402
import plots  # noqa: E402

HERE = os.path.dirname(__file__)
PRED_FILE = os.path.join(HERE, "test_predictions.csv")
FIGS = os.path.join(HERE, "figures"); os.makedirs(FIGS, exist_ok=True)


def main():
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"{PRED_FILE} not found. Run `python run_pipeline.py` first — "
            "it now saves test-set predictions that this script reuses."
        )

    df = pd.read_csv(PRED_FILE)
    p, y_true = df["p"].values, df["y_true"].values
    print(f"Loaded {len(p):,} test-set predictions.")

    rrr_grid = np.linspace(0.05, 0.50, 19)
    c_int_grid = np.linspace(100, 3000, 20)

    print("Computing sensitivity grid (RRR x intervention cost) ...")
    savings_grid, flagged_grid = sens.sensitivity_grid(p, y_true, rrr_grid, c_int_grid)

    # base case for reference
    base = sens.savings_per_1000(p, y_true, ca.RRR, ca.C_INTERVENTION)
    print(f"Base case (RRR={ca.RRR}, C_intervention=${ca.C_INTERVENTION:.0f}): "
          f"${base['savings_per_1000']:,.0f} per 1,000 patients "
          f"({base['pct_flagged']:.1%} flagged)")

    frac_positive = float((savings_grid > 0).mean())
    print(f"Targeting beats doing nothing in {frac_positive:.0%} of the "
          f"{savings_grid.size} grid combinations tested.")

    plots.sensitivity_heatmap(savings_grid, rrr_grid, c_int_grid,
                              os.path.join(FIGS, "sensitivity_heatmap.png"))

    print("Computing break-even intervention cost by RRR ...")
    rrr_grid_fine = np.linspace(0.05, 0.50, 30)
    breakevens = sens.breakeven_curve(p, y_true, rrr_grid_fine,
                                      c_int_bounds=(50, 3000))
    plots.breakeven_plot(rrr_grid_fine, breakevens,
                         os.path.join(FIGS, "breakeven_curve.png"))

    # save the grid itself for transparency / reuse
    out = pd.DataFrame(savings_grid, index=rrr_grid, columns=c_int_grid)
    out.index.name = "RRR"
    out.columns.name = "C_intervention"
    out.to_csv(os.path.join(HERE, "sensitivity_grid.csv"))

    print("\nFigures -> figures/sensitivity_heatmap.png, "
          "figures/breakeven_curve.png")
    print("Grid data -> sensitivity_grid.csv")


if __name__ == "__main__":
    main()
