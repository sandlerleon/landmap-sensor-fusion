# -*- coding: utf-8 -*-
"""Does the classifier used for the economic layer matter? The main VOI analysis
(landmap_econ.py) uses random forest rather than the classification benchmark's best
performer (XGBoost) because XGBoost's raw probabilities are worse calibrated (Brier 0.070
vs 0.059 for RF; landmap_model_results.json), and VOI consumes those probabilities
directly. This script tests whether that choice actually changes the economic conclusion,
by comparing mean VOI (same protocol as landmap_econ.py's N_REP_ECON-averaged headline
number) from: (a) random forest [the paper's default], (b) raw XGBoost, (c) XGBoost with
Platt-scaled (sigmoid) probability calibration.

    python landmap_calib_voi.py
"""
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import landmap_synth as LS
from landmap_econ import ECON, N_REP_ECON, breakeven_survey_cost, monte_carlo_voi, sample_econ, voi_for_survey
from landmap_model import SEED, make_classifier, spatial_split

N_MC = 3000


def make_xgb_calibrated(seed):
    base = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                           subsample=0.85, colsample_bytree=0.85,
                                           eval_metric="mlogloss", random_state=seed, verbosity=0))])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def brier_multiclass(y_true, proba, n_classes):
    briers = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        briers.append(brier_score_loss(yb, proba[:, c]))
    return float(np.mean(briers))


def holdout_proba(kind, seed, block=8):
    surv = LS.generate(seed=seed)
    grid = surv["grid"]
    Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
    mask = spatial_split(grid, block=block, seed=seed)
    if kind == "xgb_calibrated":
        clf = make_xgb_calibrated(seed)
    else:
        clf = make_classifier(kind)
    clf.fit(Xf[~mask], y[~mask])
    proba = clf.predict_proba(Xf[mask])
    return proba, y[mask]


def main():
    kinds = ["rf", "xgb", "xgb_calibrated"]
    voi_means, be_means, briers = {k: [] for k in kinds}, {k: [] for k in kinds}, {k: [] for k in kinds}
    for rep in range(N_REP_ECON):
        seed_r = SEED + 1000 + rep
        for kind in kinds:
            proba, y_r = holdout_proba(kind, seed_r)
            briers[kind].append(brier_multiclass(y_r, proba, LS.N_CLASSES))
            v, _, _, _ = monte_carlo_voi(proba, y_r, n_mc=N_MC, seed=seed_r)
            voi_means[kind].append(float(np.mean(v)))
            d = breakeven_survey_cost(proba, y_r, n_mc=N_MC, seed=seed_r)
            be_means[kind].append(float(np.mean(d)))
        print("repeat %d/%d done" % (rep + 1, N_REP_ECON))

    def summarize(vals):
        a = np.array(vals)
        return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
                "sd": round(float(a.std(ddof=1)), 2), "p_positive": round(float(np.mean(a > 0)), 4)}

    res = {"seed": SEED, "n_rep": N_REP_ECON, "n_mc": N_MC,
           "brier": {k: {"mean": round(float(np.mean(briers[k])), 4),
                        "sd": round(float(np.std(briers[k], ddof=1)), 4)} for k in kinds},
           "voi": {k: summarize(voi_means[k]) for k in kinds},
           "breakeven": {k: summarize(be_means[k]) for k in kinds},
           "voi_diff_xgb_minus_rf": {"mean": round(float(np.mean(np.array(voi_means["xgb"]) -
                                                                 np.array(voi_means["rf"]))), 2)},
           "voi_diff_xgbcal_minus_xgb": {"mean": round(float(np.mean(np.array(voi_means["xgb_calibrated"]) -
                                                                     np.array(voi_means["xgb"]))), 2)}}
    print("\nBrier (macro one-vs-rest, mean over repeats):")
    for k in kinds:
        print(" ", k, res["brier"][k])
    print("\nVOI (mean over %d repeats):" % N_REP_ECON)
    for k in kinds:
        print(" ", k, res["voi"][k])
    print("\nDiff XGB-raw minus RF:", res["voi_diff_xgb_minus_rf"])
    print("Diff XGB-calibrated minus XGB-raw:", res["voi_diff_xgbcal_minus_xgb"])
    json.dump(res, open("landmap_calib_voi_results.json", "w"), indent=1)
    print("\nsaved landmap_calib_voi_results.json")


if __name__ == "__main__":
    main()
