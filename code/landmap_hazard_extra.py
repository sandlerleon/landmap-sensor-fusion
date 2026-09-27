# -*- coding: utf-8 -*-
"""Extra gas-hazard operating-point metrics (precision, specificity, PR-AUC, fraction of
cells flagged) that the headline AUC/recall pair in landmap_model_results.json does not
show on its own -- a classifier can reach high recall by flagging almost everything.
Reuses the same spatial-split protocol and hazard_eval() as landmap_model.py, but only
re-runs the hazard classifier (not the material classifiers), so it is much cheaper than
a full landmap_model.py rerun.

    python landmap_hazard_extra.py
"""
import json

import numpy as np

import landmap_synth as LS
from landmap_model import N_REPEATS, SEED, hazard_eval, spatial_split


def main():
    keys = ["auc", "pr_auc", "recall", "precision", "specificity", "frac_flagged"]
    out = {k: [] for k in keys}
    totals = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    n_used = 0
    for rep in range(N_REPEATS):
        surv = LS.generate(seed=SEED + rep)
        grid = surv["grid"]
        mask_test = spatial_split(grid, block=8, seed=SEED + rep)
        Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
        hr = hazard_eval(Xf, haz, mask_test, SEED + rep)
        if hr is None:
            continue
        n_used += 1
        for k in keys:
            out[k].append(hr[k])
        for k in totals:
            totals[k] += hr[k]

    def summarize(vals):
        a = np.array(vals)
        return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
                "p05": round(float(np.percentile(a, 5)), 4), "p95": round(float(np.percentile(a, 95)), 4)}

    res = {"seed": SEED, "n_repeats_used": n_used,
           "metrics": {k: summarize(v) for k, v in out.items()},
           "pooled_confusion": totals,
           "pooled_precision": totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else None,
           "pooled_specificity": totals["tn"] / (totals["tn"] + totals["fp"]) if (totals["tn"] + totals["fp"]) else None,
           "note": "hazard ground truth is a logistic-thresholded function of the pre-noise gas "
                   "signal from the same generative model as the deployed gas sensor; these "
                   "metrics are an upper bound on independently ground-truthed performance."}
    for k in keys:
        print(k, res["metrics"][k])
    print("pooled confusion:", totals, "pooled precision", res["pooled_precision"],
          "pooled specificity", res["pooled_specificity"])
    json.dump(res, open("landmap_hazard_extra_results.json", "w"), indent=1)
    print("\nsaved landmap_hazard_extra_results.json")


if __name__ == "__main__":
    main()
