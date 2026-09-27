# -*- coding: utf-8 -*-
"""Sensor-selection map: which sensor channel carries the greatest incremental value of
information as a function of the site's resource-value structure? The single-regime result
(only the magnetometer has reliably positive incremental VOI) could be an artifact of one
assumed price table (Table 2). This grids over relative metal and plastic value (price x
recovery fraction, scaled from the baseline ECON assumptions) and finds, at every grid
point, argmax_j incremental-VOI_j across the five sensors -- the same ablation logic as
landmap_econ.py's sensor_value(), just repeated across an economic parameter grid instead
of at one fixed point. Classifier training (the expensive part) is cached once per repeat
and reused across the whole economic grid, since the economics do not affect the trained
classifier or its held-out probabilities.

    python landmap_sensor_map.py
"""
import copy
import json

import numpy as np

import landmap_synth as LS
from landmap_econ import ECON, SEED
from landmap_econ_fast import mean_voi_fast
from landmap_model import make_classifier, spatial_split

N_REP = 15
N_MC = 2500
METAL_MULT = [0.5, 0.8, 1.0, 1.3, 1.6]
PLASTIC_MULT = [0.4, 0.8, 1.3, 2.0, 3.0]


def scaled_econ(metal_mult, plastic_mult):
    e = copy.deepcopy(ECON)
    e["price_usd_per_t"]["metal"] = tuple(v * metal_mult for v in ECON["price_usd_per_t"]["metal"])
    e["price_usd_per_t"]["plastic"] = tuple(v * plastic_mult for v in ECON["price_usd_per_t"]["plastic"])
    return e


def main():
    ablation_sets = {"all": LS.SENSORS}
    for s in LS.SENSORS:
        ablation_sets["minus_" + s] = [x for x in LS.SENSORS if x != s]

    # ---- cache held-out (proba, y_true) once per repeat per sensor-set; reused for every
    # economic grid point below, since the classifier does not depend on the economics.
    cache = {name: [] for name in ablation_sets}
    for rep in range(N_REP):
        seed_r = SEED + 2000 + rep
        surv = LS.generate(seed=seed_r)
        grid = surv["grid"]
        mask = spatial_split(grid, block=8, seed=seed_r)
        for name, sensors in ablation_sets.items():
            Xf, y, haz, feat = LS.to_table(surv, sensors=sensors)
            clf = make_classifier("rf")
            clf.fit(Xf[~mask], y[~mask])
            proba = clf.predict_proba(Xf[mask])
            cache[name].append((proba, y[mask]))
        print("cached repeat %d/%d" % (rep + 1, N_REP))

    grid_out = []
    for metal_mult in METAL_MULT:
        row = []
        for plastic_mult in PLASTIC_MULT:
            econ = scaled_econ(metal_mult, plastic_mult)
            incr = {s: [] for s in LS.SENSORS}
            for rep in range(N_REP):
                seed_r = SEED + 5000 + rep
                proba_all, y_r = cache["all"][rep]
                voi_all = float(np.mean(mean_voi_fast(proba_all, y_r, econ, N_MC, seed_r)))
                for s in LS.SENSORS:
                    proba_wo, y_wo = cache["minus_" + s][rep]
                    assert np.array_equal(y_wo, y_r)
                    voi_wo = float(np.mean(mean_voi_fast(proba_wo, y_wo, econ, N_MC, seed_r)))
                    incr[s].append(voi_all - voi_wo)
            means = {s: float(np.mean(v)) for s, v in incr.items()}
            p_pos = {s: float(np.mean(np.array(v) > 0)) for s, v in incr.items()}
            winner = max(means, key=means.get)
            cell = {"metal_mult": metal_mult, "plastic_mult": plastic_mult,
                   "incremental_voi_mean": means, "p_positive": p_pos, "winner": winner,
                   "winner_margin": means[winner] - sorted(means.values())[-2]}
            row.append(cell)
            print("metal_mult=%.1f plastic_mult=%.1f -> winner=%s means=%s"
                  % (metal_mult, plastic_mult, winner, {k: round(v, 0) for k, v in means.items()}))
        grid_out.append(row)

    res = {"seed": SEED, "n_rep": N_REP, "n_mc": N_MC, "metal_mult": METAL_MULT,
           "plastic_mult": PLASTIC_MULT, "grid": grid_out,
           "baseline_price_metal": ECON["price_usd_per_t"]["metal"],
           "baseline_price_plastic": ECON["price_usd_per_t"]["plastic"]}
    json.dump(res, open("landmap_sensor_map_results.json", "w"), indent=1)
    winners = sorted(set(cell["winner"] for row in grid_out for cell in row))
    print("\ndistinct winning sensors across the grid:", winners)
    print("saved landmap_sensor_map_results.json")


if __name__ == "__main__":
    main()
