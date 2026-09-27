# -*- coding: utf-8 -*-
"""Follow-up to landmap_sensor_map.py: the first grid (metal price 0.5-1.6x, plastic price
0.4-3.0x baseline) never flipped the winner away from the magnetometer. Before reporting
"the magnetometer wins everywhere tested" as if that were the whole story, push to
deliberately extreme, still-nameable site archetypes -- scaling recovery fraction as well
as price for the challenger class, and testing construction & demolition debris (GPR's own
strongest classification signature) and glass, not just plastic -- to see whether any
economically nameable regime actually flips the ranking, or whether magnetometer dominance
is robust even under scenarios designed to break it.

    python landmap_sensor_map_extreme.py
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


def regime_econ(metal_mult, challenger_class, challenger_price_mult, challenger_rec_mult):
    e = copy.deepcopy(ECON)
    e["price_usd_per_t"]["metal"] = tuple(v * metal_mult for v in ECON["price_usd_per_t"]["metal"])
    e["recovery_fraction"]["metal"] = tuple(min(0.95, v * metal_mult) for v in ECON["recovery_fraction"]["metal"])
    e["price_usd_per_t"][challenger_class] = tuple(v * challenger_price_mult
                                                    for v in ECON["price_usd_per_t"][challenger_class])
    e["recovery_fraction"][challenger_class] = tuple(min(0.95, v * challenger_rec_mult)
                                                      for v in ECON["recovery_fraction"][challenger_class])
    return e


ARCHETYPES = {
    "baseline": dict(metal_mult=1.0, challenger_class="plastic", challenger_price_mult=1.0, challenger_rec_mult=1.0),
    "metal_starved": dict(metal_mult=0.3, challenger_class="plastic", challenger_price_mult=1.0, challenger_rec_mult=1.0),
    "polymer_max": dict(metal_mult=0.3, challenger_class="plastic", challenger_price_mult=5.0, challenger_rec_mult=4.0),
    "cd_max": dict(metal_mult=0.3, challenger_class="cd", challenger_price_mult=6.0, challenger_rec_mult=2.5),
    "glass_max": dict(metal_mult=0.3, challenger_class="glass", challenger_price_mult=8.0, challenger_rec_mult=3.5),
}


def main():
    ablation_sets = {"all": LS.SENSORS}
    for s in LS.SENSORS:
        ablation_sets["minus_" + s] = [x for x in LS.SENSORS if x != s]
    cache = {name: [] for name in ablation_sets}
    for rep in range(N_REP):
        seed_r = SEED + 2000 + rep     # same seeds as landmap_sensor_map.py -> identical classifiers
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

    res = {}
    for arch_name, params in ARCHETYPES.items():
        econ = regime_econ(**params)
        incr = {s: [] for s in LS.SENSORS}
        for rep in range(N_REP):
            seed_r = SEED + 5000 + rep
            proba_all, y_r = cache["all"][rep]
            voi_all = float(np.mean(mean_voi_fast(proba_all, y_r, econ, N_MC, seed_r)))
            for s in LS.SENSORS:
                proba_wo, y_wo = cache["minus_" + s][rep]
                voi_wo = float(np.mean(mean_voi_fast(proba_wo, y_wo, econ, N_MC, seed_r)))
                incr[s].append(voi_all - voi_wo)
        means = {s: float(np.mean(v)) for s, v in incr.items()}
        p_pos = {s: float(np.mean(np.array(v) > 0)) for s, v in incr.items()}
        winner = max(means, key=means.get)
        res[arch_name] = {"params": params, "incremental_voi_mean": means, "p_positive": p_pos,
                          "winner": winner}
        print("%-15s -> winner=%s means=%s" % (arch_name, winner, {k: round(v, 0) for k, v in means.items()}))

    json.dump(res, open("landmap_sensor_map_extreme_results.json", "w"), indent=1)
    winners = sorted(set(v["winner"] for v in res.values()))
    print("\ndistinct winners across archetypes:", winners)
    print("saved landmap_sensor_map_extreme_results.json")


if __name__ == "__main__":
    main()
