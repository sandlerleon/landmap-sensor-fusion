# -*- coding: utf-8 -*-
"""Two-dimensional decision-viability map: mean VOI(alpha, beta) over a sensing-degradation
axis (alpha, reusing landmap_robustness.py's combined noise/dropout/attenuation/overlap
parametrization) and an independent economic-adversity axis (beta: weaker secondary-material
prices and a relatively costlier survey). The 93%-boundary reported from the one-dimensional
combined sweep is the beta=baseline slice of this surface; the full surface gives the
VOI(alpha,beta)=0 contour instead of a single point estimate, replacing "the conclusion
reverses around 93% combined severity" with an actual decision-viability region.

Classifier training depends only on alpha, not beta, so it is cached once per alpha value
and reused across every beta on that row -- the expensive part (survey generation +
classifier fit) runs only n_alpha x n_rep times, not n_alpha x n_beta x n_rep times.

    python landmap_voi_map.py
"""
import json

import numpy as np

import landmap_synth as LS
from landmap_econ import ECON, SEED
from landmap_econ_fast import mean_voi_fast
from landmap_model import make_classifier, spatial_split
from landmap_robustness import degraded_assumptions, degraded_response

N_REP = 10
N_MC = 2500
ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0]     # sensing-degradation severity
BETA = [0.0, 0.25, 0.5, 0.75, 1.0]      # economic-adversity severity


def combined_degradation_params(alpha):
    """Same combined parametrization as landmap_robustness.py's binary-search axis."""
    return dict(noise_mult=1.0 + alpha * 1.5, dropout_mult=1.0 + alpha * 3.0,
                gpr_atten_mult=1.0 + alpha * 1.5, moisture_mult=1.0 + alpha * 1.5,
                overlap_shrink=alpha * 0.6)


def econ_degraded(beta):
    import copy
    e = copy.deepcopy(ECON)
    price_shrink = 1.0 - 0.5 * beta
    for cls in e["price_usd_per_t"]:
        e["price_usd_per_t"][cls] = tuple(v * price_shrink for v in e["price_usd_per_t"][cls])
    cost_grow = 1.0 + 1.5 * beta
    e["survey_cost_usd_total"] = tuple(v * cost_grow for v in e["survey_cost_usd_total"])
    return e


def main():
    cache = {}
    for alpha in ALPHA:
        p = combined_degradation_params(alpha)
        A = degraded_assumptions(p["noise_mult"], p["dropout_mult"], p["gpr_atten_mult"], p["moisture_mult"])
        R = degraded_response(p["overlap_shrink"])
        reps = []
        for rep in range(N_REP):
            seed_r = SEED + 9000 + rep
            surv = LS.generate(seed=seed_r, response=R, assumptions=A)
            grid = surv["grid"]
            mask = spatial_split(grid, block=8, seed=seed_r)
            Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
            clf = make_classifier("rf")
            clf.fit(Xf[~mask], y[~mask])
            proba = clf.predict_proba(Xf[mask])
            reps.append((proba, y[mask]))
        cache[alpha] = reps
        print("cached alpha=%.2f (%d repeats)" % (alpha, N_REP))

    grid_out = []
    for alpha in ALPHA:
        row = []
        for beta in BETA:
            econ = econ_degraded(beta)
            vois = []
            for rep_i, (proba, y_r) in enumerate(cache[alpha]):
                seed_r = SEED + 9500 + rep_i
                v = mean_voi_fast(proba, y_r, econ, N_MC, seed_r)
                vois.append(float(np.mean(v)))
            cell = {"alpha": alpha, "beta": beta, "voi_mean": float(np.mean(vois)),
                   "voi_sd": float(np.std(vois, ddof=1)),
                   "p_positive": float(np.mean(np.array(vois) > 0))}
            row.append(cell)
            print("alpha=%.2f beta=%.2f -> mean VOI %.0f (P>0 across repeat-means=%.2f)"
                  % (alpha, beta, cell["voi_mean"], cell["p_positive"]))
        grid_out.append(row)

    # ---- zero-contour: for each alpha row, linearly interpolate the beta at which mean VOI crosses 0
    zero_crossings = []
    for i, alpha in enumerate(ALPHA):
        vals = [c["voi_mean"] for c in grid_out[i]]
        crossing = None
        for j in range(len(BETA) - 1):
            if (vals[j] > 0) != (vals[j + 1] > 0):
                t = vals[j] / (vals[j] - vals[j + 1])
                crossing = BETA[j] + t * (BETA[j + 1] - BETA[j])
                break
        zero_crossings.append({"alpha": alpha, "beta_at_voi_zero": crossing})

    res = {"seed": SEED, "n_rep": N_REP, "n_mc": N_MC, "alpha_grid": ALPHA, "beta_grid": BETA,
           "grid": grid_out, "zero_crossings": zero_crossings}
    json.dump(res, open("landmap_voi_map_results.json", "w"), indent=1)
    print("\nzero crossings (beta at which VOI=0, per alpha row):")
    for z in zero_crossings:
        print(" ", z)
    print("saved landmap_voi_map_results.json")


if __name__ == "__main__":
    main()
