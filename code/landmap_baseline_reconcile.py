# -*- coding: utf-8 -*-
"""One canonical baseline VOI estimate, not two. landmap_econ.py's headline VOI ($729,
N_REP_ECON=20 repeats, n_mc=3000, seeds SEED+1000+rep) and landmap_robustness.py's
undegraded baseline ($1012, 12 repeats, n_mc=1500, seeds SEED+rep) are both correct,
independent Monte Carlo estimates of the same underlying quantity, computed under an
otherwise identical protocol (same classifier, same block, same undegraded assumptions).
Rather than assert they are "close enough," this pools both batches' raw per-repeat VOI
means into one 40-repeat estimate with a correctly quantified standard error, and reports
that pooled number as the paper's single canonical baseline VOI.

Requires landmap_econ_results.json (with voi_averaged_raw) and a fresh reconciled-protocol
run of the undegraded scenario at the same n_rep/n_mc as landmap_econ.py.

    python landmap_baseline_reconcile.py
"""
import json

import numpy as np

from landmap_econ import N_REP_ECON
from landmap_robustness import eval_scenario

E = json.load(open("landmap_econ_results.json"))
batch_a = E["voi_averaged_raw"]                          # seeds SEED+1000..SEED+1019
batch_b_full = eval_scenario(1, 1, 1, 1, 0, n_rep=N_REP_ECON, n_mc=3000)  # seeds SEED..SEED+19
batch_b = batch_b_full["voi_raw"]

pooled = np.array(batch_a + batch_b)
res = {
    "batch_a": {"label": "landmap_econ.py headline (seeds SEED+1000..)", "n": len(batch_a),
                "mean": round(float(np.mean(batch_a)), 1), "sd": round(float(np.std(batch_a, ddof=1)), 1)},
    "batch_b": {"label": "landmap_robustness.py reconciled-protocol baseline (seeds SEED..)",
                "n": len(batch_b), "mean": round(float(np.mean(batch_b)), 1),
                "sd": round(float(np.std(batch_b, ddof=1)), 1)},
    "diff_se": round(float(np.sqrt(np.var(batch_a, ddof=1) / len(batch_a) +
                                   np.var(batch_b, ddof=1) / len(batch_b))), 1),
    "pooled": {"n": len(pooled), "mean": round(float(np.mean(pooled)), 1),
              "median": round(float(np.median(pooled)), 1),
              "sd": round(float(np.std(pooled, ddof=1)), 1),
              "se": round(float(np.std(pooled, ddof=1) / np.sqrt(len(pooled))), 1),
              "p_positive": round(float(np.mean(pooled > 0)), 3)},
}
res["diff"] = round(res["batch_a"]["mean"] - res["batch_b"]["mean"], 1)
res["diff_in_se_units"] = round(abs(res["diff"]) / res["diff_se"], 2)
print(json.dumps(res, indent=1))
json.dump(res, open("landmap_baseline_reconcile_results.json", "w"), indent=1)
print("\nsaved landmap_baseline_reconcile_results.json")
