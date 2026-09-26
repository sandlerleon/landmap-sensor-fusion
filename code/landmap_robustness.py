# -*- coding: utf-8 -*-
"""Synthetic-to-real stress test: degrade the simulation along the axes the white paper's
own risk section names (sensor noise, dropout, GPR penetration, moisture, signature overlap)
and find where classification and, more importantly, economic value (VOI) collapse. This is
the falsification boundary the manuscript reports rather than asserting.

    python landmap_robustness.py
"""
import copy
import json

import numpy as np

import landmap_synth as LS
from landmap_econ import breakeven_survey_cost, monte_carlo_voi, sample_econ, voi_for_survey
from landmap_model import make_classifier, spatial_split

SEED = 20260926
N_REP = 12


def degraded_assumptions(noise_mult=1.0, dropout_mult=1.0, gpr_atten_mult=1.0,
                         moisture_mult=1.0, overlap_shrink=0.0):
    A = copy.deepcopy(LS.ASSUMPTIONS)
    for k in A["noise_sd"]:
        A["noise_sd"][k] = A["noise_sd"][k] * noise_mult
    for k in A["dropout_p"]:
        A["dropout_p"][k] = min(0.9, A["dropout_p"][k] * dropout_mult)
    A["depth_atten_per_m"]["gpr"] = A["depth_atten_per_m"]["gpr"] * gpr_atten_mult
    A["moisture_gpr_atten"] = A["moisture_gpr_atten"] * moisture_mult
    return A


def degraded_response(overlap_shrink):
    """Shrink every material's response toward the six-material mean by 'overlap_shrink' (0-1),
    modelling real, weathered, heterogeneous waste being less distinctive than the idealised
    synthetic signatures the white paper's own risk section flags as its main limitation."""
    R = copy.deepcopy(LS.RESPONSE)
    if overlap_shrink <= 0:
        return R
    n_ch = len(next(iter(R.values())))
    mean_vec = np.mean([np.array(v) for v in R.values()], axis=0)
    for cls in R:
        v = np.array(R[cls])
        R[cls] = (v * (1 - overlap_shrink) + mean_vec * overlap_shrink).tolist()
    return R


def eval_scenario(noise_mult, dropout_mult, gpr_atten_mult, moisture_mult, overlap_shrink,
                  n_rep=N_REP, seed=SEED):
    A = degraded_assumptions(noise_mult, dropout_mult, gpr_atten_mult, moisture_mult)
    R = degraded_response(overlap_shrink)
    macro_f1s, vois, bes = [], [], []
    for rep in range(n_rep):
        s = seed + rep
        surv = LS.generate(seed=s, response=R, assumptions=A)
        grid = surv["grid"]
        Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
        mask = spatial_split(grid, block=8, seed=s)
        clf = make_classifier("xgb")
        clf.fit(Xf[~mask], y[~mask])
        pred = clf.predict(Xf[mask])
        proba = clf.predict_proba(Xf[mask])
        from sklearn.metrics import f1_score
        macro_f1s.append(f1_score(y[mask], pred, average="macro", zero_division=0))
        v, _, _, _ = monte_carlo_voi(proba, y[mask], n_mc=1500, seed=s)
        vois.append(float(np.mean(v)))
        d = breakeven_survey_cost(proba, y[mask], n_mc=1500, seed=s)
        bes.append(float(np.mean(d)))
    return {"macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
            "macro_f1_sd": round(float(np.std(macro_f1s, ddof=1)), 4),
            "voi_mean": round(float(np.mean(vois)), 1), "voi_sd": round(float(np.std(vois, ddof=1)), 1),
            "voi_p_positive": round(float(np.mean(np.array(vois) > 0)), 3),
            "breakeven_mean": round(float(np.mean(bes)), 1)}


def main():
    res = {"seed": SEED, "n_rep": N_REP, "baseline": eval_scenario(1, 1, 1, 1, 0)}
    print("baseline:", res["baseline"])

    # ---- one-axis sweeps
    sweeps = {
        "noise_mult": [1.0, 1.5, 2.0, 3.0, 4.0],
        "dropout_mult": [1.0, 2.0, 4.0, 8.0],
        "gpr_atten_mult": [1.0, 1.5, 2.0, 3.0],       # >1 = shallower effective GPR penetration
        "moisture_mult": [1.0, 1.5, 2.0, 3.0],
        "overlap_shrink": [0.0, 0.15, 0.30, 0.45, 0.60],
    }
    res["sweeps"] = {}
    for axis, values in sweeps.items():
        res["sweeps"][axis] = []
        for v in values:
            kw = {"noise_mult": 1.0, "dropout_mult": 1.0, "gpr_atten_mult": 1.0,
                  "moisture_mult": 1.0, "overlap_shrink": 0.0}
            kw[axis] = v
            r = eval_scenario(**kw)
            r["value"] = v
            res["sweeps"][axis].append(r)
            print(axis, v, "-> F1", r["macro_f1_mean"], "VOI", r["voi_mean"], "P(VOI>0)", r["voi_p_positive"])

    # ---- combined "realistic degradation" scenario (all axes moved together)
    res["combined_degradation"] = []
    for level, (nm, dm, gm, mm, ov) in enumerate([
            (1.0, 1.0, 1.0, 1.0, 0.0), (1.3, 1.5, 1.3, 1.3, 0.15),
            (1.6, 2.0, 1.6, 1.6, 0.30), (2.0, 3.0, 2.0, 2.0, 0.45),
            (2.5, 4.0, 2.5, 2.5, 0.60)]):
        r = eval_scenario(nm, dm, gm, mm, ov)
        r["level"] = level
        res["combined_degradation"].append(r)
        print("combined level", level, "-> F1", r["macro_f1_mean"], "VOI", r["voi_mean"],
              "P(VOI>0)", r["voi_p_positive"])

    # ---- locate the VOI=0 boundary along the combined axis by finer search
    lo, hi = 0.0, 1.0
    for _ in range(6):
        mid = (lo + hi) / 2
        nm = 1.0 + mid * 1.5; dm = 1.0 + mid * 3.0; gm = 1.0 + mid * 1.5
        mm = 1.0 + mid * 1.5; ov = mid * 0.6
        r = eval_scenario(nm, dm, gm, mm, ov, n_rep=8)
        if r["voi_mean"] > 0:
            lo = mid
        else:
            hi = mid
    res["voi_zero_boundary"] = {"combined_severity_fraction": round((lo + hi) / 2, 3),
                                "note": "fraction of the way from baseline (0) to the most severe "
                                        "combined scenario (1) at which mean VOI crosses zero"}
    print("\nVOI=0 boundary at combined severity fraction:", res["voi_zero_boundary"])

    json.dump(res, open("landmap_robustness_results.json", "w"), indent=1)
    print("\nsaved landmap_robustness_results.json")


if __name__ == "__main__":
    main()
